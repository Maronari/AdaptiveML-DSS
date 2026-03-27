from __future__ import annotations

import logging
import time

from backend.services.job_service import JobService


LOGGER = logging.getLogger("adaptiveml.worker.training")
POLL_INTERVAL_SECONDS = 2.0


def run_training_worker() -> None:
    """Poll the shared job queue and execute queued training jobs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = JobService()
    worker_name = "training-worker"
    LOGGER.info("Training worker started.")

    while True:
        job = service.claim_next_job(job_types=["training_dataset"], worker_name=worker_name)
        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        LOGGER.info("Claimed job %s for project %s.", job["job_id"], job["project_id"])
        service.process_job(job_id=job["job_id"], worker_name=worker_name)


if __name__ == "__main__":
    run_training_worker()
