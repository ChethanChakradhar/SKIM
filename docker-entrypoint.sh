#!/bin/sh
# ---------------------------------------------------------------------
# Runs as root, then hands off to an unprivileged user.
#
# Why this exists: the Dockerfile creates /data and gives it to the
# `skim` user at BUILD time. Railway then mounts a persistent volume
# over /data at RUN time, and that mount arrives owned by root --
# replacing the directory and the permissions set during the build.
#
# The app then cannot create its database and every page that touches
# storage returns 500, while /health (which touches nothing) stays
# green. That combination is the fingerprint of this bug.
#
# So: fix the mount's ownership as root, then drop to `skim` before
# exec'ing the server. The app never serves a request as root.
# ---------------------------------------------------------------------
set -e

DATA_DIR="${SKIM_DATA_DIR:-/data}"
mkdir -p "$DATA_DIR"
chown -R skim:skim "$DATA_DIR" || echo "warning: could not chown $DATA_DIR"

# `exec` replaces this shell with the server, so the app becomes PID 1's
# child directly and receives stop signals instead of them being
# swallowed by a lingering shell.
#
# setpriv comes from util-linux, already in the base image. `su` is the
# fallback if a future base image drops it.
if command -v setpriv >/dev/null 2>&1; then
  exec setpriv --reuid=skim --regid=skim --init-groups \
    uvicorn web.app:app --host 0.0.0.0 --port "${PORT:-8000}"
else
  exec su skim -c "uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}"
fi
