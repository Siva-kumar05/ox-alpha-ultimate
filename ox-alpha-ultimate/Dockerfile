# ox-alpha-ultimate runtime image.
#
# The agent defaults to PAPER mode from config.yaml.  Live Dhan execution is
# only possible with explicit operator affirmation and credentials supplied as
# environment variables at run time - never baked into the image:
#
#   docker build -t ox-alpha-ultimate .
#   docker run --rm -v "$PWD/state:/srv/oxalpha/state" \
#     -e OX_AUDIT_KEY=... -e DHAN_CLIENT_ID=... -e DHAN_TOKEN=... \
#     ox-alpha-ultimate live-test 300
#
# Dashboard (paper or live-read-only views):
#   docker run --rm -p 8501:8501 ox-alpha-ultimate streamlit run dashboard.py
FROM python:3.12-slim-bookworm

WORKDIR /srv/oxalpha
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install the (lean) base requirements first so dependency layers cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The codebase itself.
COPY . .

# Non-root operator.  State (databases/logs/track-record) is written under the
# workdir, so give the runtime user the tree it needs.
RUN useradd --create-home --uid 1000 ox && chown -R ox:ox /srv/oxalpha
USER ox

# Paper-mode state and backups live here by default; mount a volume for
# persistence across containers:
#   -v "$PWD/state:/srv/oxalpha/state" -v "$PWD/backups:/srv/oxalpha/backups"
VOLUME ["/srv/oxalpha/state", "/srv/oxalpha/backups"]

# Default: autonomous paper run.  Override with any run.py command, e.g.
#   docker run ... ox-alpha-ultimate track-record
#   docker run ... ox-alpha-ultimate smoketest
CMD ["python", "run.py", "run"]
