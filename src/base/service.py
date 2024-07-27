from base.constants import *
import threading
import asyncio

def pretty_print_semaphore(semaphore):
    """Print a semaphore in better format."""
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

class BaseService(object):
    def __init__(self, model_names="", worker_addr="", worker_id="", no_register=False):
        self.controller_addr = AddressWorker.CONTROLLER_URL
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        
        self.global_counter = 0
        self.model_semaphore = None
        self.speed = 1
        self.limit_model_concurrency = 5
        
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()
            
            
    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(self.model_semaphore)}. "
            f"global_counter: {self.global_counter}. "
            f"worker_id: {self.worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "speed": self.speed,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                self.speed = int(1/ret.elapsed.total_seconds())
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)
        #print("---exist: ", exist)
        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            self.model_semaphore is None
            or self.model_semaphore._value is None
            or self.model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.limit_model_concurrency
                - self.model_semaphore._value
                + len(self.model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": self.speed,
            "queue_length": self.get_queue_length(),
        }
        
    def release_model_semaphore(self,):
        self.model_semaphore.release()

    def acquire_model_semaphore(self,):
        self.global_counter += 1
        if self.model_semaphore is None:
            self.model_semaphore = asyncio.Semaphore(self.limit_model_concurrency)
        return self.model_semaphore.acquire()
