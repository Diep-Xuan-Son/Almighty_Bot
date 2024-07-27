import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
    
from base.service import BaseService
from base.constants import *
from typing import List, Union, Tuple, Optional, Type
from PIL import Image
from io import BytesIO
import cv2 
import tritonclient.grpc as grpcclient
import uvicorn
import numpy as np
from loguru import logger
import uuid
import socket
import argparse
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse

logger_skinlesion = logger.bind(name="logger_skinlesion")
logger_skinlesion.add(os.path.join(PATH_DEFAULT.LOGDIR, f"skinlesion_worker.{datetime.date.today()}.log"), mode='w')
    
class SkinLesionWorker(BaseService):
    def __init__(self, worker_addr, model_names, worker_id, limit_model_concurrency, client):
        logger_skinlesion.debug("Init skin lesion worker")
        super().__init__(model_names=model_names, worker_addr=worker_addr, worker_id=str(uuid.uuid4())[:6])
        # self.worker_addr = worker_addr
        # self.model_names = model_names
        self.client = client
        self.limit_model_concurrency = limit_model_concurrency
    
    def load_image(self, image_path: str) -> Tuple[np.array]:
        if os.path.exists(image_path):
            image_source = Image.open(image_path).convert("RGB")
        else:
            # base64 coding
            image_source = Image.open(BytesIO(image_path)).convert("RGB")
        image = np.asarray(image_source)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    @estimate_execute_time("Run skin lesion worker", logger_skinlesion)
    def run(self, image):
        try:
            image_data = self.load_image(image)
            image_data = np.expand_dims(image_data, axis=0)
            input_tensors = [grpcclient.InferInput("imgs", image_data.shape, "UINT8")]
            input_tensors[0].set_data_from_numpy(image_data)
            results = self.client.infer(model_name="skinlesion_recognition", inputs=input_tensors)
            diseases = results.as_numpy("class_top5").squeeze(0)
            diseases = [x.decode("utf-8") for x in diseases]
            conf = results.as_numpy("conf_top5")
            ret = {"success": True, "information": f"The disease in this image is {diseases[0]} or {diseases[1]}", "result": {"disease": diseases, "confident": conf.tolist()}}
        except Exception as e:
            logger_skinlesion.exception(e)
            ret = {
                "success": False,
                "error": f"{Configuration.server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret
    
app = FastAPI()


@app.post("/worker_generate")
async def api_generate(image: UploadFile = File(...), id_thread: int=1):
    # params = await request.json()
    image = await image.read()
    await worker.acquire_model_semaphore()
    output = worker.run(image)
    worker.release_model_semaphore()
    output["id_thread"] = id_thread
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=str(urlparse(AddressWorker.SKINLESION_WORKER_URL).hostname))
    parser.add_argument("--port", type=int, default=int(urlparse(AddressWorker.SKINLESION_WORKER_URL).port))
    parser.add_argument("--worker-address", type=str, default=AddressWorker.SKINLESION_WORKER_URL)

    parser.add_argument(
        "--model-names",
        default=ModelName.SKINLESION_WORKER,
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=10)
    parser.add_argument("--worker-id", type=str, default=str(uuid.uuid4())[:6])
    args = parser.parse_args()
    logger.info(f"args: {args}")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_real = s.getsockname()[0]
        args.worker_address = f"http://{ip_real}:{args.port}"
    except:
        pass

    client = grpcclient.InferenceServerClient(url=Configuration.tritonserver_url)

    worker = SkinLesionWorker(
        args.worker_address,
        args.model_names,
        args.worker_id,
        args.limit_model_concurrency,
        client
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")