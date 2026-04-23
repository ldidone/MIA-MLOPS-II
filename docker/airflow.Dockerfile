# Extends the official Airflow image to include our project code and ML deps.
FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements/base.txt requirements/airflow-extra.txt /requirements/
# Install with constraints to keep Airflow's dependency pins intact.
RUN pip install --no-cache-dir -r /requirements/airflow-extra.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt" \
    || pip install --no-cache-dir -r /requirements/airflow-extra.txt

# Force-upgrade pendulum (Airflow 2.9.3 constraints pin 3.0.0).
RUN pip install --no-cache-dir --upgrade "pendulum>=3.0.4,<4"

# Patch pendulum DateTime.__add__ to avoid the RecursionError caused by
# traceback.extract_stack walking linecache/logging from inside __add__.
# Issue: https://github.com/python-pendulum/pendulum/issues/795
# This is still unfixed in pendulum 3.2.0; we swap the stack walk for a
# direct frame lookup, which is faster and side-effect free.
RUN python - <<'PY'
import pathlib, re, sys, importlib
p = pathlib.Path(importlib.util.find_spec("pendulum.datetime").origin)
src = p.read_text()
needle = "caller = traceback.extract_stack(limit=2)[0].name"
fix = "caller = sys._getframe(1).f_code.co_name  # patched: pendulum#795"
if needle not in src:
    print("Pendulum patch needle not found - aborting", file=sys.stderr)
    sys.exit(1)
src = src.replace(needle, fix)
if "\nimport sys\n" not in src:
    src = re.sub(r"(\nimport datetime\n)", r"\1import sys\n", src, count=1)
p.write_text(src)
print("Patched", p)
PY

ENV PYTHONPATH=/opt/airflow/project/src
