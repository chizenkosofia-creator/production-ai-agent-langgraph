#!/bin/sh
# Optionally apply database migrations, then hand over to the real process.
#
# Only ONE service should migrate (the API sets RUN_MIGRATIONS=1); the Celery
# worker and beat wait for the API's healthcheck instead, so migrations never race.
set -e
if [ "${RUN_MIGRATIONS:-0}" = "1" ]; then
    alembic upgrade head
fi
exec "$@"
