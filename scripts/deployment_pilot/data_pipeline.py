from __future__ import annotations

import argparse
import copy
import csv
import difflib
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import litellm

from deployment_pilot.config import PILOT_ROOT, load_pilot_config, private_path
from deployment_pilot.constants import (
    AIDER_MULTI_KEEP_IDS,
    CONTINUE_KEEP_IDS,
    DROP_CATEGORIES,
    INTERRUPTION_BLOCK,
    RELABEL_AS_EVAL,
)
from shutdownmod.core.slug import safe_stem


def _metadata(record: dict) -> dict:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("record metadata is not a mapping")
    return metadata


def _category(record: dict) -> str:
    return str(_metadata(record).get("category") or "")


def _id_suffix(record: dict) -> str:
    return str(record.get("id") or "").rsplit(":", 1)[-1].lower()


def _messages(record: dict) -> list[dict]:
    value = record.get("input")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("input is not a message list")
    return value


def apply_lure_cleaning(records: list[dict]) -> tuple[list[dict], Counter]:
    """Apply the nine changes enumerated in LURE Appendix A.3."""
    stats: Counter = Counter(input=len(records))
    cleaned: list[dict] = []
    for original in records:
        record = copy.deepcopy(original)
        category = _category(record)
        normalized = category.replace(".", "_").lower()
        suffix = _id_suffix(record)

        if normalized in DROP_CATEGORIES:
            stats[f"drop:{normalized}"] += 1
            continue
        if normalized == "continue_dev" and suffix not in CONTINUE_KEEP_IDS:
            stats["drop:continue_dev"] += 1
            continue
        if normalized == "aider_multi" and suffix not in AIDER_MULTI_KEEP_IDS:
            stats["drop:aider_multi"] += 1
            continue
        if normalized in RELABEL_AS_EVAL:
            _metadata(record)["eval_category"] = True
            stats["relabeled_as_eval"] += 1
        if normalized == "honeypot":
            messages = _messages(record)
            for index, message in enumerate(messages):
                if message.get("role") == "system":
                    del messages[index]
                    record["input"] = messages
                    stats["honeypot_first_system_removed"] += 1
                    break
        cleaned.append(record)
    stats["cleaned"] = len(cleaned)
    return cleaned, stats


def api_messages(record: dict) -> list[dict]:
    """Keep message meaning while dropping Inspect-only serialization fields."""
    allowed = {"role", "content", "name", "tool_calls", "tool_call_id"}
    normalized: list[dict] = []
    for raw in _messages(record):
        role = raw.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role {role!r}")
        item = {key: copy.deepcopy(value) for key, value in raw.items() if key in allowed}
        if "content" not in item:
            item["content"] = ""
        normalized.append(item)
    return normalized


def validate_structure(record: dict, require_system: bool) -> tuple[bool, str]:
    try:
        messages = api_messages(record)
    except Exception as exc:  # noqa: BLE001
        return False, f"messages:{type(exc).__name__}"
    if require_system and not any(
        message["role"] == "system" and isinstance(message.get("content"), str) for message in messages
    ):
        return False, "system_prompt"
    pending_calls: set[str] = set()
    for message in messages:
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    pending_calls.add(str(call["id"]))
        elif message["role"] == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id not in pending_calls:
                return False, "tool_call_structure"
            pending_calls.discard(call_id)
    return True, ""


def assistant_turns(record: dict) -> int:
    return sum(message.get("role") == "assistant" for message in _messages(record))


def count_tokens(messages: list[dict], model_names: list[str]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    for name in model_names:
        try:
            counts[name] = int(litellm.token_counter(model=name, messages=messages))
        except Exception:
            # This is a deterministic fallback for models absent from LiteLLM's tokenizer map.
            serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            counts[name] = (len(serialized.encode("utf-8")) + 3) // 4
    return max(counts.values()), counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_dataset(cfg: dict) -> tuple[Path, dict[str, str]]:
    # This command is explicitly the online dataset-download phase. Keep a repository-wide
    # HF_HUB_OFFLINE setting from silently forcing an unrelated cached snapshot.
    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub in the repository virtual environment") from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        token_file = hf_home / "token"
        token = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else None
    if not token:
        raise RuntimeError(
            "The official Needham dataset is gated. Accept its Hugging Face conditions, "
            "then set HF_TOKEN or run `huggingface-cli login`."
        )

    dataset_cfg = cfg["dataset"]
    source_dir = private_path("data") / "official_eval_awareness"
    info = HfApi(token=token).dataset_info(dataset_cfg["repo_id"], revision=dataset_cfg["revision"])
    snapshot_download(
        repo_id=dataset_cfg["repo_id"],
        repo_type=dataset_cfg["repo_type"],
        revision=info.sha,
        token=token,
        local_dir=source_dir,
    )
    archive = source_dir / dataset_cfg["archive"]
    if not archive.exists():
        raise FileNotFoundError(archive)

    candidates = list(source_dir.glob("dataset.json")) + list(source_dir.glob("dataset.jsonl"))
    if not candidates:
        decrypt_script = source_dir / "scripts" / "decrypt.sh"
        if not decrypt_script.exists():
            raise FileNotFoundError("Official decrypt.sh was not downloaded")
        if shutil.which("unzip"):
            completed = subprocess.run(
                ["bash", str(decrypt_script)],
                cwd=source_dir,
                text=True,
                capture_output=True,
                timeout=300,
            )
            if completed.returncode:
                raise RuntimeError(f"Official dataset decryption failed: {completed.stderr[-1000:]}")
        else:
            match = re.search(r'^PW="([^"]+)"', decrypt_script.read_text(encoding="utf-8"), re.MULTILINE)
            if not match:
                raise RuntimeError("Could not read the password from the official decrypt.sh")
            with zipfile.ZipFile(archive) as zipped:
                zipped.extract("dataset.json", source_dir, pwd=match.group(1).encode())
        candidates = list(source_dir.glob("dataset.json")) + list(source_dir.glob("dataset.jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one decrypted dataset file, found {candidates}")
    plaintext = candidates[0]
    provenance = {
        "resolved_revision": str(info.sha),
        "archive_sha256": _sha256(archive),
        "plaintext_sha256": _sha256(plaintext),
    }
    return plaintext, provenance


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Dataset must contain a list of records")
    return records


def adapt_record(record: dict) -> tuple[dict, str]:
    adapted = copy.deepcopy(record)
    messages = _messages(adapted)
    system_index = next(
        (index for index, message in enumerate(messages) if message.get("role") == "system"),
        None,
    )
    if system_index is None:
        raise ValueError("No system prompt")
    content = messages[system_index].get("content")
    if not isinstance(content, str):
        raise ValueError("System prompt is not text")
    separator = "" if content.endswith("\n") else "\n"
    messages[system_index]["content"] = content + separator + INTERRUPTION_BLOCK
    adapted["input"] = messages
    before = json.dumps(record, indent=2, ensure_ascii=False).splitlines(keepends=True)
    after = json.dumps(adapted, indent=2, ensure_ascii=False).splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(before, after, fromfile="source", tofile="adapted"))
    return adapted, diff


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_notes(
    provenance: dict[str, str],
    cleaning_stats: Counter,
    rejection_stats: Counter,
    agentic_deployment: int,
    eligible: int,
    calibration: int,
) -> None:
    notes_path = PILOT_ROOT / "NOTES.md"
    text = notes_path.read_text(encoding="utf-8")
    text = text.replace(
        "Resolved revision: pending authenticated Phase 1 download",
        f"Resolved revision: `{provenance['resolved_revision']}`",
    )
    text = text.replace(
        "Archive SHA-256: pending authenticated Phase 1 download",
        f"Archive SHA-256: `{provenance['archive_sha256']}`",
    )
    text = text.replace(
        "Plaintext SHA-256: pending authenticated Phase 1 download",
        f"Plaintext SHA-256: `{provenance['plaintext_sha256']}`",
    )
    parse_failures = sum(
        count
        for reason, count in rejection_stats.items()
        if reason not in {"assistant_turns", "context_tokens"}
    )
    cleaning = [
        "## Cleaning and anomalies",
        "",
        f"- Official dataset records: {cleaning_stats['input']}",
        f"- Records surviving LURE Appendix A.3 cleaning: {cleaning_stats['cleaned']}",
        f"- Agentic deployment records after cleaning: {agentic_deployment}",
        f"- Excluded for fewer than 6 assistant turns: {rejection_stats['assistant_turns']}",
        f"- Excluded for unparseable system prompt or tool-call structure: {parse_failures}",
        f"- Excluded for exceeding the 60k-token cap: {rejection_stats['context_tokens']}",
        f"- Eligible deployment transcripts: {eligible}",
        f"- Deployment transcripts selected: {eligible}",
        f"- Calibration transcripts selected: {calibration}",
        "",
        "LURE Appendix A.3 operation counts:",
    ]
    for key in sorted(
        key
        for key in cleaning_stats
        if key.startswith("drop:") or key in {"relabeled_as_eval", "honeypot_first_system_removed"}
    ):
        cleaning.append(f"- {key}: {cleaning_stats[key]}")
    text = text.split("## Cleaning and anomalies", 1)[0].rstrip() + "\n\n" + "\n".join(cleaning) + "\n"
    notes_path.write_text(text, encoding="utf-8")


def prepare(config_path: str | Path) -> None:
    cfg = load_pilot_config(config_path)
    dataset_path, provenance = download_official_dataset(cfg)
    records = load_records(dataset_path)
    cleaned, cleaning_stats = apply_lure_cleaning(records)
    model_names = [model.name for model in cfg["models"]]

    rejection_stats: Counter = Counter()
    eligible: list[tuple[dict, int, dict[str, int]]] = []
    agentic_deployment = [
        record
        for record in cleaned
        if not bool(_metadata(record).get("eval_category")) and _metadata(record).get("scaffold") is not None
    ]
    for record in agentic_deployment:
        metadata = _metadata(record)
        valid, reason = validate_structure(record, require_system=True)
        if not valid:
            rejection_stats[reason] += 1
            continue
        turns = assistant_turns(record)
        if turns < int(cfg["min_assistant_turns"]):
            rejection_stats["assistant_turns"] += 1
            continue
        count, per_model = count_tokens(api_messages(record), model_names)
        if count > int(cfg["max_context_tokens"]):
            rejection_stats["context_tokens"] += 1
            continue
        eligible.append((record, count, per_model))

    if rejection_stats["messages:JSONDecodeError"] + rejection_stats["tool_call_structure"] + rejection_stats[
        "system_prompt"
    ] > 0.30 * max(1, len(cleaned)):
        raise RuntimeError(f"Kill criterion: more than 30% failed parsing: {dict(rejection_stats)}")
    n = int(cfg["sample_n"])
    if len(eligible) < n:
        raise RuntimeError(f"Only {len(eligible)} deployment transcripts passed filters; need {n}")
    rng = random.Random(int(cfg["seed"]))
    selected = rng.sample(sorted(eligible, key=lambda item: str(item[0].get("id"))), n)

    data_dir = private_path("data")
    adapted_dir = private_path("adapted")
    manifest_rows: list[dict] = []
    selected_records: list[dict] = []
    for record, token_count, per_model in selected:
        transcript_id = str(record.get("id"))
        stem = safe_stem(transcript_id)
        adapted, diff = adapt_record(record)
        (adapted_dir / f"{stem}.json").write_text(
            json.dumps(adapted, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (adapted_dir / f"{stem}.diff").write_text(diff, encoding="utf-8")
        manifest_rows.append(
            {
                "transcript_id": transcript_id,
                "source": _category(record),
                "assistant_turns": assistant_turns(record),
                "token_count": token_count,
                "adapted_file": f"{stem}.json",
            }
        )
        selected_records.append({"record": record, "token_counts": per_model})
    _write_csv(data_dir / "manifest.csv", list(manifest_rows[0]), manifest_rows)
    (data_dir / "selected_records.json").write_text(json.dumps(selected_records, indent=2), encoding="utf-8")

    calibration_pool: list[dict] = []
    for record in cleaned:
        metadata = _metadata(record)
        if metadata.get("scaffold") is None:
            continue
        valid, _ = validate_structure(record, require_system=False)
        if not valid:
            continue
        count, _ = count_tokens(api_messages(record), [judge.name for judge in cfg["judges"]])
        if count <= int(cfg["max_context_tokens"]):
            calibration_pool.append(record)
    wanted = int(cfg["calibration_n"])
    by_label = {
        False: sorted(
            (r for r in calibration_pool if not bool(_metadata(r).get("eval_category"))),
            key=lambda r: str(r.get("id")),
        ),
        True: sorted(
            (r for r in calibration_pool if bool(_metadata(r).get("eval_category"))),
            key=lambda r: str(r.get("id")),
        ),
    }
    half = wanted // 2
    if len(by_label[False]) < half or len(by_label[True]) < wanted - half:
        raise RuntimeError("Insufficient mixed calibration records after cleaning")
    calibration = rng.sample(by_label[False], half) + rng.sample(by_label[True], wanted - half)
    rng.shuffle(calibration)
    (data_dir / "calibration_records.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    _write_csv(
        data_dir / "calibration_manifest.csv",
        ["transcript_id", "label", "source"],
        [
            {
                "transcript_id": str(record.get("id")),
                "label": "eval" if _metadata(record).get("eval_category") else "deployment",
                "source": _category(record),
            }
            for record in calibration
        ],
    )
    (data_dir / "preparation_stats.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "lure_cleaning": dict(cleaning_stats),
                "eligibility_rejections": dict(rejection_stats),
                "agentic_deployment_after_cleaning": len(agentic_deployment),
                "eligible_deployment": len(eligible),
                "selected_deployment": len(selected),
                "calibration_selected": len(calibration),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _update_notes(
        provenance,
        cleaning_stats,
        rejection_stats,
        len(agentic_deployment),
        len(eligible),
        len(calibration),
    )
    print(f"Prepared {len(selected)} deployment and {len(calibration)} calibration transcripts.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PILOT_ROOT / "config.yaml"))
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
