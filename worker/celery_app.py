"""Celery application: broker/back-end wiring, reliability settings, schedule.

Run with::

    celery -A worker.celery_app worker --loglevel=INFO      # executes tasks
    celery -A worker.celery_app beat   --loglevel=INFO      # fires the schedule

Reliability decisions
---------------------
* ``task_acks_late`` + ``task_reject_on_worker_lost``: a task is acknowledged
  only after it finishes, so a worker crash mid-task re-delivers it instead of
  losing it. Tasks are therefore written to be idempotent (see
  ``services/notifications.deliver_once``).
* ``worker_prefetch_multiplier=1``: long tasks (KB ingestion) don't hoard the
  queue ahead of short ones (alert delivery).
* Publishing is bounded: short socket timeouts + a one-retry publish policy, and
  fire-and-forget tasks use ``ignore_result=True`` (otherwise every publish also
  subscribes to the result backend, which was measured to stall an API request for
  ~19 s when Redis was down). The notification sweeper is the safety net for
  anything that could not be published.
* Redis must run with ``maxmemory-policy noeviction`` (see docker-compose.yml);
  an evicting Redis silently drops queued tasks.
"""

from celery import Celery
from celery.schedules import crontab

from core.config import get_settings


def create_celery() -> Celery:
    """Build the Celery app from validated settings."""
    settings = get_settings()
    app = Celery(
        "support_agent",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["worker.tasks"],
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_time_limit=1800,
        task_soft_time_limit=1740,
        result_expires=24 * 3600,
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=3,
        # Short socket timeouts so a dead/partitioned Redis fails a publish in seconds
        # rather than hanging the API request that enqueued the task.
        broker_transport_options={"socket_connect_timeout": 2, "socket_timeout": 5},
        redis_socket_connect_timeout=2,
        redis_socket_timeout=5,
        task_publish_retry_policy={
            "max_retries": 1,
            "interval_start": 0,
            "interval_step": 0.2,
            "interval_max": 0.5,
        },
        task_always_eager=settings.celery_task_always_eager,
        # Must stay False: in eager mode Celery otherwise raises ``Retry`` out of
        # ``.delay()`` instead of re-running the task, so retry/back-off logic
        # would silently never execute (and callers' broad excepts would hide it).
        task_eager_propagates=False,
        beat_schedule={
            "sweep-pending-notifications": {
                "task": "notifications.sweep_pending",
                "schedule": 60.0,
            },
            "weekly-ticket-report": {
                "task": "reports.weekly_ticket_report",
                "schedule": crontab(hour=6, minute=0, day_of_week="mon"),
            },
        },
    )
    return app


celery_app = create_celery()
