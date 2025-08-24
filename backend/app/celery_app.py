import os
from celery import Celery
from kombu import Queue, Exchange

# Create Celery instance
celery_app = Celery(
    "verixai",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["app.tasks.document_tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes hard limit
    task_soft_time_limit=1500,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time per worker
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks to prevent memory leaks
    task_acks_late=True,  # Tasks acknowledged after completion
    task_reject_on_worker_lost=True,  # Reject tasks if worker dies
    broker_connection_retry_on_startup=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Redis specific settings
    broker_transport_options={
        'visibility_timeout': 3600,
        'fanout_prefix': True,
        'fanout_patterns': True
    },
    result_backend_transport_options={
        'master_name': 'mymaster',
    },
)

# Task routing for different queues
celery_app.conf.task_routes = {
    "app.tasks.document_tasks.process_document": {"queue": "documents"},
    "app.tasks.document_tasks.process_document_batch": {"queue": "documents"},
    "app.tasks.document_tasks.process_large_document": {"queue": "large_documents"},
    "app.tasks.document_tasks.process_with_multi_agent": {"queue": "documents"},
}

# Queue configuration using kombu
celery_app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("documents", Exchange("documents"), routing_key="documents"),
    Queue("large_documents", Exchange("large_documents"), routing_key="large_documents"),
)

# Default queue
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_default_exchange = "default"
celery_app.conf.task_default_routing_key = "default"