"""Run long research steps in background threads so the UI stays usable. Progress lives in SQLite."""
import threading
import traceback

_cancel: set[int] = set()
_lock = threading.Lock()


def start(library, kind: str, ref_id: int | None, fn) -> int:
    """fn(report, cancelled) -> final message. report(progress 0..1 or None, message)."""
    job_id = library.create_job(kind, ref_id)

    def report(progress, message):
        values = {"message": message[:300]}
        if progress is not None:
            values["progress"] = max(0.0, min(1.0, float(progress)))
        library.update_job(job_id, **values)

    def cancelled() -> bool:
        return job_id in _cancel

    def run():
        try:
            message = fn(report, cancelled) or "Done"
            status = "cancelled" if cancelled() else "done"
            library.update_job(job_id, status=status, progress=1.0,
                               message="Stopped" if status == "cancelled" else message[:300])
        except Exception as e:
            traceback.print_exc()
            library.update_job(job_id, status="failed", error=str(e)[:500], message=f"Failed: {str(e)[:200]}")
        finally:
            with _lock:
                _cancel.discard(job_id)

    threading.Thread(target=run, daemon=True, name=f"job-{kind}-{job_id}").start()
    return job_id


def cancel(job_id: int) -> None:
    with _lock:
        _cancel.add(job_id)


def active(library, kind_prefix: str, ref_id: int) -> dict | None:
    job = library.latest_job(kind_prefix, ref_id)
    return job if job and job["status"] == "running" else None
