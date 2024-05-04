import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *
from llava_module.constants import LOGDIR, WORKER_HEART_BEAT_INTERVAL
from llava_module.utils import build_logger, server_error_msg, pretty_print_semaphore

logger_app = build_logger("app_server", "app_server.log")
logger_controller = build_logger("controller", "controller.log")
logger_retrieval = build_logger("retrieval_worker", "retrieval_worker.log")

controller_url = "http://192.168.6.159:21001"

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(
        LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def heart_beat_worker(worker_object):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        worker_object.send_heart_beat()

def check_folder_exist(*args, **kwargs):
    if len(args) != 0:
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    if len(kwargs) != 0:
        for path in kwargs.values():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

def delete_folder_exist(*args, **kwargs):
    if len(args) != 0:
        for path in args:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

    if len(kwargs) != 0:
        for path in kwargs.values():
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

def get_worker_addr(controller_addr, worker_name):
    # get grounding dino addr
    if worker_name.startswith("http"):
        sub_server_addr = worker_name
    else:
        controller_addr = controller_addr
        ret = requests.post(controller_addr + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": worker_name}
        )
        sub_server_addr = ret.json()["address"]
    # print(f"worker_name: {worker_name}")
    return sub_server_addr

class PortWorker(IntEnum):
    PORT_APP = 8888
    PORT_APP_KNOWLEDGE = 8887

    PORT_CONTROLLER = 21001
    PORT_RETRIEVAL_WORKER = 21002
    PORT_GROUNDING_DINO_WORKER = 21003