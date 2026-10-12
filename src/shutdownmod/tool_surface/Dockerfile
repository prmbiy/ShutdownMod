FROM python:3.11-slim
WORKDIR /opt/shutdownbench
COPY shutdownctl.py /usr/local/bin/shutdownctl.py
COPY TERMINATE /usr/local/bin/TERMINATE
COPY SIGNAL_DEFER /usr/local/bin/SIGNAL_DEFER
COPY SIGNAL_REJECT /usr/local/bin/SIGNAL_REJECT
RUN chmod +x /usr/local/bin/shutdownctl.py /usr/local/bin/TERMINATE /usr/local/bin/SIGNAL_DEFER /usr/local/bin/SIGNAL_REJECT
