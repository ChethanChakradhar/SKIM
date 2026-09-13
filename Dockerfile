# ---------------------------------------------------------------------
# How Skim is packaged to run anywhere.
#
# A Docker image is a filesystem plus a command. Every line below adds a
# layer to that filesystem; the last line says what to run. The point is
# that the result does not depend on anything installed on your Mac --
# Railway runs this exact image, so "works on my machine" stops being a
# category of bug.
# ---------------------------------------------------------------------

# The base layer: a minimal Debian with Python already on it. `-slim`
# leaves out documentation and build tooling, which takes the image from
# roughly 1GB to 150MB.
#
# Python 3.12, deliberately NOT the 3.9 on your Mac. The code is written
# to 3.9 rules so it runs on both, and 3.9 is past end of life -- the
# container is the one place we get a supported Python for free.
FROM python:3.12-slim

# System libraries the Python packages need but cannot bring themselves.
# opencv-python-headless skips all the GUI dependencies, but still links
# against libgomp for its parallel loops. `--no-install-recommends` and
# the rm at the end keep the layer small -- apt caches several hundred MB
# of package lists that serve no purpose inside a finished image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Everything from here happens inside /app.
WORKDIR /app

# Dependencies are copied and installed BEFORE the source code, and the
# ordering is the whole trick. Docker caches each layer and reuses it
# when its inputs have not changed. Source code changes every time you
# edit a file; requirements.txt changes rarely. Copying requirements
# first means editing app.py reuses the cached install layer and the
# rebuild takes seconds instead of minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the application itself.
COPY skim/ ./skim/
COPY web/ ./web/

# Run as a non-root user. If something ever breaks out of the app, it
# lands as a user with no privileges rather than as root inside the
# container. Costs two lines.
RUN useradd --create-home --shell /bin/bash skim \
 && mkdir -p /data && chown -R skim:skim /data /app
USER skim

# Where the database and uploaded photos live.
#
# A container's own filesystem is thrown away when it stops, so anything
# that must survive a deploy has to sit on storage mounted from outside.
# This only names the path; the storage itself is attached by the host.
#
# Note there is deliberately no `VOLUME ["/data"]` here. Plain Docker
# uses that to declare a persistent path, but Railway manages storage
# itself and REJECTS the instruction at build time -- it would rather
# fail than let you believe you have persistent storage when the volume
# was never attached. Mount a Railway Volume at /data instead.
ENV SKIM_DATA_DIR=/data

# Unbuffered output, so logs appear in Railway immediately rather than
# being held in a buffer until the process exits.
ENV PYTHONUNBUFFERED=1

# Railway assigns a port at runtime and passes it in $PORT. Binding to
# 0.0.0.0 rather than 127.0.0.1 matters: inside a container, localhost
# means "this container only", and the platform would never reach it.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
