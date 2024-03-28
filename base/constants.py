import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *
from llava.constants import LOGDIR, WORKER_HEART_BEAT_INTERVAL
from llava.utils import build_logger, server_error_msg, pretty_print_semaphore

logger_app = build_logger("app_server", "app_server.log")
logger_controller = build_logger("controller", "controller.log")
logger_retrieval = build_logger("retrieval_worker", "retrieval_worker.log")

controller_url = "http://localhost:21001"

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(
        LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def heart_beat_worker(worker_object):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        worker_object.send_heart_beat()