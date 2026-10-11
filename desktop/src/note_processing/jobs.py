import secrets
from concurrent.futures import ThreadPoolExecutor
from threading import RLock


class Cancelled(Exception):
    pass


class Jobs:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.items = {}
        self.lock = RLock()

    def start(self, operation):
        with self.lock:
            if any(j['status'] in {'running', 'queued'} for j in self.items.values()):
                raise ValueError('A job is already running; wait or cancel it first')
            identifier = secrets.token_hex(8)
            job = {'id': identifier, 'status': 'queued', 'percent': 0, 'message': 'Queued', 'cancel': False}
            self.items[identifier] = job
            while len(self.items) > 20:
                self.items.pop(next(iter(self.items)))

        def progress(percent, message):
            with self.lock:
                if job['cancel']:
                    raise Cancelled()
                job.update(percent=int(percent), message=message)

        def run():
            with self.lock:
                job['status'] = 'running'
            try:
                result = operation(progress)
                with self.lock:
                    job.update(status='complete', percent=100, result=result, message='Complete')
            except Cancelled:
                with self.lock:
                    job.update(status='cancelled', message='Cancelled; already completed batches were kept')
            except Exception as error:
                with self.lock:
                    job.update(status='failed', error=str(error), message=str(error))
        self.executor.submit(run)
        return identifier

    def get(self, identifier):
        with self.lock:
            if identifier not in self.items:
                raise ValueError('Job not found')
            return dict(self.items[identifier])

    def cancel(self, identifier):
        with self.lock:
            if identifier not in self.items:
                raise ValueError('Job not found')
            self.items[identifier]['cancel'] = True

    def close(self):
        with self.lock:
            for job in self.items.values():
                job['cancel'] = True
        self.executor.shutdown(wait=True)
