import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *
from base.constants import *
from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = "AIzaSyDE_QjNXwSYh_uwc4LqBhmf-1Q_U1lZ7RQ"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

worker_id = str(uuid.uuid4())[:6]
gemini_logger = build_logger("gemini_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

class ModelWorker:
    def __init__(self, controller_addr, worker_addr, worker_id, no_register, model_path, model_names, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or model_path.split("/")[-1]
        self.device = device   
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()
        self.llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")

    def register_to_controller(self):
        gemini_logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        # print(url)
        # print(data)
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        gemini_logger.info(f"Send heart beat. Models: {[self.model_names]}"
                              f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                              f"global_counter: {global_counter}")

        url = self.controller_addr  + "/receive_heart_beat"
        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()
                }, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                gemini_logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)
    
    def get_status(self):
        return {
            "model_names": [self.model_names],
            "speed": 1,
            "queue_length": self.get_queue_length()
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        prompt = params.prompt
        images = params.files
        
        if images is not None:
            img_pil = Image.open(BytesIO(base64.b64decode(images[0])))
            llm = self.llm_vision
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt
                    },  # You can optionally provide text parts
                    {
                        "type": "image_url",
                        "image_url": img_pil
                     },
                ]
            )
            print(img_pil)
        else:
            llm = self.llm
            message = prompt
        
        generated_text = "Answer:"
        for chunk in llm.stream([message]):
            text_result = chunk.content.split(" ")
            for text in text_result:
                generated_text += " " + str(text)
                yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

        print(generated_text)

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"

def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()

app = FastAPI()

class ParamsChatbot(BaseModel):
    files: List[UploadFile] = File(None) # File(...) for required
    prompt: str = Query("Hãy miêu tả bức ảnh này", description="Write prompt to ask chatbot")

@app.post("/worker_generate_stream")
async def generate_stream(params: ParamsChatbot = Depends()):
    global model_semaphore, global_counter
    global_counter += 1
    images = []

    if params.files is not None:
        for file in params.files:
            f = await file.read()
            images.append(f)
        params.files = images

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)

@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21008)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21008")
    parser.add_argument("--controller-address", type=str,
        default=controller_url)
    parser.add_argument("--model-path", type=str, default="gemini-pro-vision")
    parser.add_argument("--model-name", type=str, default="gemini")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-register", action="store_true", default=False)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    args = parser.parse_args()
    gemini_logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_name,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")