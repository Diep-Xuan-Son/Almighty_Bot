import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

# from base.libs import *
import os
import shutil 
from enum import IntEnum
from pydantic import BaseModel
from typing import Optional
from loguru import logger
import logging 
import datetime
import time
import requests
import json
from urllib.parse import urlparse

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
	 
class AddressWorker():
	AP_URL = "http://192.168.6.161:8888"
	APP_KNOWLEDGE_URL = "http://192.168.6.161:8887"
	CONTROLLER_URL = "http://192.168.6.161:21001"
	RETRIEVAL_WORKER_URL = "http://192.168.6.161:21002"
	GROUNDING_DINO_WORKER_URL = "http://192.168.6.161:21003"
	SKINLESION_WORKER_URL = "http://192.168.6.161:21004"
controller_url = AddressWorker.CONTROLLER_URL

class ModelName():
    SKINLESION_WORKER = "skinlesion"

class PathDefault(BaseModel):
	PATH_CONVER: Optional[str] = os.path.abspath(f"{ROOT}/history")
	PATH_IMAGE: Optional[str] = os.path.abspath(os.path.join(PATH_CONVER, "images"))
	PATH_VOICES: Optional[str] = os.path.abspath(os.path.join(PATH_CONVER, "voices"))
	LOGDIR: Optional[str] = f"{str(ROOT)}/logs"
	
	def check_exist(self):
		check_folder_exist(**self.__dict__)
		print("----Check finished!")
PATH_DEFAULT = PathDefault()
PATH_DEFAULT.check_exist()

class Configuration():
	CONTROLLER_HEART_BEAT_EXPIRATION = 90
	WORKER_HEART_BEAT_INTERVAL = 60
	tritonserver_url = "192.168.6.161:8001"
	server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
	moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

class ErrorCode(IntEnum):
    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006
    
#---------------------------log---------------------------
logger.level("INFO", color="<light-green><dim>")
logger.level("DEBUG", color="<cyan><bold><italic>")
logger.level("WARNING", color="<yellow><bold><italic>")
logger.level("ERROR", color="<red><bold>")

# logger_app = build_logger("app_server", "app_server.log")
# logger_controller = build_logger("controller", "controller.log")
# logger_retrieval = build_logger("retrieval_worker", "retrieval_worker.log")

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, log_level=logging.INFO):
		self.terminal = sys.stdout
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def __getattr__(self, attr):
		return getattr(self.terminal, attr)

	def write(self, buf):
		temp_linebuf = self.linebuf + buf
		self.linebuf = ''
		for line in temp_linebuf.splitlines(True):
			# From the io.TextIOWrapper docs:
			#   On output, if newline is None, any '\n' characters written
			#   are translated to the system default line separator.
			# By default sys.stdout.write() expects '\n' newlines and then
			# translates them so this is still cross platform.
			if line[-1] == '\n':
				self.logger.log(self.log_level, line.rstrip())
			else:
				self.linebuf += line

	def flush(self):
		if self.linebuf != '':
			self.logger.log(self.log_level, self.linebuf.rstrip())
		self.linebuf = ''

formatter = logging.Formatter(
	# fmt='\033[0;32m'+"%(asctime)s.%(msecs)03d | %(levelname)s    | %(name)s | %(message)s",
	fmt="%(asctime)s.%(msecs)03d | %(levelname)s    | %(name)s | %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S"
)

if not logging.getLogger().handlers:
	logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(formatter)
    
stdout_logger = logging.getLogger("stdout")
stdout_logger.setLevel(logging.INFO)
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger("stderr")
stderr_logger.setLevel(logging.ERROR)
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl
#////////////////////////////////////////////////////////////

def get_conv_log_filename():
	t = datetime.datetime.now()
	name = os.path.join(
		PATH_DEFAULT.LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
	return name

def heart_beat_worker(worker_object):
	while True:
		time.sleep(Configuration.WORKER_HEART_BEAT_INTERVAL)
		worker_object.send_heart_beat()
        
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
		print(f"----Models: {models}")
		if worker_name not in models:
			return -1
		ret = requests.post(
			controller_addr + "/get_worker_address", json={"model": worker_name}
		)
		sub_server_addr = ret.json()["address"]
	# print(f"worker_name: {worker_name}")
	return sub_server_addr

def read_jsonline(address):
	not_mark = []
	with open(address, 'r', encoding="utf-8") as f:
		for jsonstr in f.readlines():
			jsonstr = json.loads(jsonstr)
			not_mark.append(jsonstr)
	return not_mark

def read_json(address):
	with open(address, 'r', encoding='utf-8') as json_file:
		json_data = json.load(json_file)
	return json_data

def estimate_execute_time(name_func: str, logger_: object):
	def decorator(func):
		def inner(*args, **kwargs):
			logger_.debug(name_func)
			st_time = time.time()
			value = func(*args, **kwargs)
			logger_.info("----Duration: {}", time.time()-st_time)
			return value
		return inner
	return decorator