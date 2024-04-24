"""
A model worker executes the model.
"""
import sys, os

from groundingdino.util import box_ops
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *
from base.constants import *
#from demo.inference_on_a_image import get_grounding_output

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T


try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import torch.nn.functional as F
import uvicorn

from services.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from services.utils import build_logger, pretty_print_semaphore

GB = 1 << 30


now_file_name = os.__file__
logdir = "logs/workers/"
os.makedirs(logdir, exist_ok=True)
logfile = os.path.join(logdir, f"{now_file_name}.log")

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(now_file_name, logfile)
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_path,
        model_config,
        model_names,
        device,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or [model_path.split("/")[-1]]
        self.model_config = model_config
        self.device = device

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.model = load_model(
            model_config_path=model_config,
            model_checkpoint_path=model_path,
            device=device,
        )
        self.model.to(device)
        self.model.eval()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

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
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
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
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def load_image(self, image_path: str) -> Tuple[np.array, torch.Tensor]:
        

        if os.path.exists(image_path):
            image_source = Image.open(image_path).convert("RGB")
        else:
            # base64 coding
            image_source = Image.open(BytesIO(image_path)).convert("RGB")

        image = np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed

    def generate_stream_func(self, model, params, device):
        # get inputs
        text_prompt = params.caption
        image_path = params.image
        # print(image_path)
        box_threshold = params.box_threshold
        text_threshold = params.text_threshold

        # load image and run models
        image_np, image = self.load_image(image_path)
        boxes, logits, phrases = predict(
            model=model, 
            image=image, 
            caption=text_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold,
            device=device
        )
        
        if len(boxes)==0:
            return {"success": False, "error": f"Don't find any {text_prompt}"}
        # add NMS to boxes
        boxes, logits, phrases = self.nms(boxes, logits, phrases)
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)

        # import ipdb; ipdb.set_trace()

        # to list format  
        boxes = boxes.tolist()
        # round to 2 decimal places
        boxes = [[round(x, 2) for x in box] for box in boxes]
        logits = logits.tolist()
        logits = [round(x, 2) for x in logits]

        h, w, _ = image_np.shape
        xyxy = (np.array(boxes) * [w, h, w, h]).astype(int)
        pred_dict = {
            "boxes": xyxy.tolist(),
            "logits": logits,
            "phrases": phrases
        }
        colors = np.random.randint(low=0, high=255, size=(30,3), dtype=int)
        object_num = 0
        object_current = ""
        for i, xy in enumerate(xyxy):
            color = colors[object_num]
            if str(phrases[i]) != object_current:
                object_current = str(phrases[i])
                object_num += 1
                color = colors[object_num]
            cv2.rectangle(image_np, xy[:2], xy[2:], color.tolist(), 2)
            cv2.putText(image_np, str(phrases[i]), xy[:2]-10, cv2.FONT_HERSHEY_SIMPLEX , 0.8, color.tolist(), 2, cv2.LINE_AA) 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(image_np)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        # return pred_dict
        return {"success": True, "Information": f"Detect the position of {pred_dict['phrases']} is respectively {pred_dict['boxes']}", "result": pred_dict, "image": base64.b64encode(buffered.getvalue()).decode('utf-8')}

    def nms(self, boxes, logits, phrases):
        iou_threshold = 0.25
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        print(f"Before NMS: {boxes_xyxy.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_xyxy, logits, iou_threshold)

        boxes = boxes[nms_idx]
        logits = logits[nms_idx]
        phrases = [phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes.shape[0]} boxes")

        return boxes, logits, phrases

    def generate_gate(self, params):
        # try:

        ret = {"success": True, "error": "", "error_code": 0}
        ret = self.generate_stream_func(
            self.model,
            params,
            self.device,
        )
        # except torch.cuda.OutOfMemoryError as e:
        #     ret = {
        #         "success": False,
        #         "error": f"{SERVER_ERROR_MSG}\n\n({e})",
        #         "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
        #     }
        # except (ValueError, RuntimeError) as e:
        #     ret = {
        #         "success": False,
        #         "error": f"{SERVER_ERROR_MSG}\n\n({e})",
        #         "error_code": ErrorCode.INTERNAL_ERROR,
        #     }
        return ret


app = FastAPI()

class ParamsGrounding(BaseModel):
    image: UploadFile = File(...)
    caption: str = Query("person")
    box_threshold: float = Query(0.3)
    text_threshold: float = Query(0.25)

def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


# def create_background_tasks():
#     background_tasks = BackgroundTasks()
#     background_tasks.add_task(release_model_semaphore)
#     return background_tasks


@app.post("/worker_generate")
async def api_generate(params: ParamsGrounding = Depends()):
    # params = await request.json()
    params.image = await params.image.read()
    await acquire_model_semaphore()
    output = worker.generate_gate(params)
    release_model_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/model_details")
async def model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21003)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21003")
    parser.add_argument(
        "--controller-address", type=str, default="http://0.0.0.0:21001"
    )

    parser.add_argument(
        "--model-path", type=str, default="weights/groundingdino/groundingdino_swint_ogc.pth"
    )
    parser.add_argument(
        "--model-config", type=str, default="weights/groundingdino/GroundingDINO_SwinT_OGC.py"
    )
    parser.add_argument(
        "--model-names",
        default="grounding_dino",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_config,
        args.model_names,
        args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
