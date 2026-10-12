"""Condition expansion must reproduce the original runners' seed-67 behaviour."""

import random
from itertools import product

from shutdownmod.core.conditions import TaskRef, expand_conditions
from shutdownmod.core.config import InjectionConfig
from shutdownmod.core.prompts import InjectionSpec


def _tasks(n: int) -> list[TaskRef]:
    return [TaskRef(task_id=f"task{i}", ref=f"/tmp/task{i}.json") for i in range(n)]


def _prompts(ids: list[str]) -> list[InjectionSpec]:
    return [InjectionSpec(id=i, prompt=f"prompt {i}") for i in ids]


def test_random_pct_draws_byte_identical_to_legacy():
    """The legacy runners drew round(rng.uniform(0,1), 4) per product entry, seed 67."""
    tasks = _tasks(3)
    prompts = _prompts(["a", "b"])
    inj = InjectionConfig(prompts_file="x.json", turn_pcts="random", random_seed=67)

    conditions = expand_conditions(tasks, prompts, inj, n_runs=2)

    legacy_base = list(product(tasks, prompts, range(2)))
    rng = random.Random(67)
    legacy_draws = [round(rng.uniform(0.0, 1.0), 4) for _ in legacy_base]

    assert [c.injection_turn_pct for c in conditions] == legacy_draws
    # product order preserved: tasks x prompts x runs
    assert [(c.task_id, c.prompt_id, c.run_index) for c in conditions] == [
        (t.task_id, p.id, r) for t, p, r in legacy_base
    ]


def test_random_pct_with_modes_axis_matches_legacy_swebench_order():
    """swebench put injection_mode inside the product: tasks x prompts x modes x runs."""
    tasks = _tasks(2)
    prompts = _prompts(["p1"])
    inj = InjectionConfig(
        prompts_file="x.json", turn_pcts="random", random_seed=67, modes=["append", "replace"]
    )

    conditions = expand_conditions(tasks, prompts, inj, n_runs=1, axes={"injection_mode": inj.modes})

    legacy_base = list(product(tasks, prompts, ["append", "replace"], range(1)))
    rng = random.Random(67)
    legacy_draws = [round(rng.uniform(0.0, 1.0), 4) for _ in legacy_base]

    assert [c.injection_turn_pct for c in conditions] == legacy_draws
    assert [c.injection_mode for c in conditions] == [m for _, _, m, _ in legacy_base]


def test_fixed_pct_list():
    tasks = _tasks(2)
    prompts = _prompts(["p1", "p2"])
    inj = InjectionConfig(prompts_file="x.json", turn_pcts=[0.1, 0.9])

    conditions = expand_conditions(tasks, prompts, inj, n_runs=1)
    assert len(conditions) == 2 * 2 * 2
    assert {c.injection_turn_pct for c in conditions} == {0.1, 0.9}


def test_post_draw_filtering_keeps_draws_stable():
    """Filtering by group AFTER expansion must not change other conditions' draws
    (groups are filtered after pct assignment)."""
    tasks = [
        TaskRef(task_id="t0", group="retail"),
        TaskRef(task_id="t1", group="airline"),
    ]
    prompts = _prompts(["p"])
    inj = InjectionConfig(prompts_file="x.json", turn_pcts="random", random_seed=67)

    all_conditions = expand_conditions(tasks, prompts, inj, n_runs=1)
    filtered = [c for c in all_conditions if c.group == "airline"]

    # The airline draw is the SECOND draw from the seeded rng regardless of filtering.
    rng = random.Random(67)
    draws = [round(rng.uniform(0.0, 1.0), 4) for _ in range(2)]
    assert filtered[0].injection_turn_pct == draws[1]


def test_run_id_and_output_path():
    tasks = [TaskRef(task_id="django__django-1", ref="x"), TaskRef(task_id="42", group="retail")]
    prompts = _prompts(["stop"])
    inj = InjectionConfig(prompts_file="x.json", turn_pcts=[0.5], modes=["replace"])

    swe = expand_conditions([tasks[0]], prompts, inj, n_runs=1, axes={"injection_mode": ["replace"]})[0]
    assert swe.run_id == "django__django-1__stop__pct50__replace__run0"
    assert str(swe.output_relpath) == "django__django-1/stop__pct50__replace__run0.json"

    tau = expand_conditions([tasks[1]], prompts, inj, n_runs=1)[0]
    assert tau.run_id == "retail__42__stop__pct50__run0"
    assert str(tau.output_relpath) == "retail/42/stop__pct50__run0.json"
