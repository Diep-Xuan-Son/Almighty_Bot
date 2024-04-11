import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *
from base.constants import *
# from pdfminer.high_level import extract_text
# from pdfminer.pdfparser import PDFParser
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
os.environ["OPENAI_API_KEY"] = ""

class ModelWorker:
    def __init__(self, controller_addr, worker_addr, worker_id, no_register, model_path, model_config, model_names, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_config = model_config
        self.model_names = model_names or model_path.split("/")[-1]
        self.device = device   
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()
        # Retrieval
        self.model = SentenceTransformer("../weights/all-mpnet-base-v2")
        self.model.max_seq_length = 512
        self.cross_encoder = CrossEncoder("../weights/ms-marco-MiniLM-L-6-v2")
        self.embeddings_topic = None
        self.paragraphs_topic = None
        self.topics = None
        self.embeddings_doc = None
        self.paragraphs_doc = None

        self.chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo", streaming= True), chain_type="stuff")
    
    def register_to_controller(self):
        logger_retrieval.info("Register to controller")

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
        logger_retrieval.info(f"Send heart beat. Models: {[self.model_names]}"
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
                logger_retrieval.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return limit_model_concurrency - model_semaphore._value + (len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)
    
    def get_status(self):
        return {
            "model_names": [self.model_names],
            "speed": 1,
            "queue_length": self.get_queue_length()
        }
    
    def embed_pdf(self, params, fnames):
        # fnames = params.fnames
        window_size = params.window_size
        step_size = params.step_size
        try:
            for fname in fnames:
                pdf = PyPDF2.PdfReader(fname)
                count = len(pdf.pages)
                text = []
                for i in range(count):
                    page = pdf.pages[i]
                    text.append(page.extract_text())
                text = ','.join(text)
                text = " ".join(text.split())
                # text_tokens = text.split()
                # text = extract_text(fname)
                # text = " ".join(text.split())
                # text_tokens = text.split()

                # sentences = []
                # print(len(text_tokens))
                # for i in range(0, len(text_tokens), step_size):
                #     window = text_tokens[i : i + window_size]
                #     if len(window) < window_size:
                #         break
                #     sentences.append(window)
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
                current_chunk = ""
                chunks = []
                k_sentence_chunk = []
                chunk_size = 1000
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence + " "
                        k_sentence_chunk.insert(0, sentence)
                        if len(k_sentence_chunk) > 2:
                            k_sentence_chunk.pop()
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = " ".join(k_sentence_chunk) + " " + sentence + " "
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # paragraphs = [" ".join(s) for s in sentences]
                paragraphs = chunks
                embeddings = self.model.encode(
                    paragraphs,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device="cuda"
                )

                if self.embeddings_doc is None:
                    self.embeddings_doc = embeddings
                    self.paragraphs_doc = paragraphs
                else:
                    self.embeddings_doc = torch.cat([self.embeddings_doc, embeddings], axis=0)
                    self.paragraphs_doc += paragraphs

            ret = {"text": "", "error_code": 0}
        except Exception as e:
            ret = {"text": e, "error_code": 1}
        return ret

    def generate_stream_func(self, params):
        query = params.query
        top_k = params.top_k
        # try:
        query_embeddings = self.model.encode(query, convert_to_tensor=True, device=self.device)
        query_embeddings = query_embeddings.cuda()
        hits = util.semantic_search(
            query_embeddings,
            self.embeddings_doc,
            top_k=top_k,
        )[0]
        # print(hits)
        cross_input = [[query, self.paragraphs_doc[hit["corpus_id"]]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_input)

        for idx in range(len(cross_scores)):
            hits[idx]["cross_score"] = cross_scores[idx]

        results = []
        hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
        for hit in hits[:3]:
            results.append(self.paragraphs_doc[hit["corpus_id"]].replace("\n", " "))

        input_documents = Document(page_content=results[0], metadata={})
        answer_gpt = self.chain.run(input_documents=[input_documents], question=query)
        generated_text = "Answer:"
        for result in str(answer_gpt).split():
            generated_text += " " + str(result)
            ret = {"text": generated_text, "error_code": 0}
            yield json.dumps(ret).encode() + b"\0"
        # generated_text = "Answer:"
        # for result in str(results[0]).split():
        #     generated_text += " " + str(result)
        #     ret = {"text": generated_text, "error_code": 0}
        #     yield json.dumps(ret).encode() + b"\0"

    def embed_csv(self, fnames): 
        # fname = params.fname
        # window_size = params.window_size
        # step_size = params.step_size
        try:
            for fname in fnames:
                df = pd.read_csv(fname, index_col=0)
                topics = df["topic"].tolist()
                paragraphs = df["paragraph"].tolist()

                embeddings = self.model.encode(
                    paragraphs,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device=self.device
                )
                if self.embeddings_topic is None:
                    self.embeddings_topic = embeddings
                    self.paragraphs_topic = paragraphs
                    self.topics = topics
                else:
                    self.embeddings_topic = torch.cat([self.embeddings_topic, embeddings], axis=0)
                    self.paragraphs_topic += paragraphs
                    self.topics += topics
            ret = {"text": "", "error_code": 0}
        except Exception as e:
            ret = {"text": e, "error_code": 1}
        return ret
    
    def generate_topic(self, params): #params = {"query", "top_k"} 
        query = params.query
        top_k = params.top_k
        try:
            query_embeddings = self.model.encode(query, convert_to_tensor=True, device=self.device)
            query_embeddings = query_embeddings.cuda()
            hits = util.semantic_search(
                query_embeddings,
                self.embeddings_topic,
                top_k=top_k,
            )[0]
            # print(hits)
            cross_input = [[query, self.paragraphs_topic[hit["corpus_id"]]] for hit in hits]
            cross_scores = self.cross_encoder.predict(cross_input)

            for idx in range(len(cross_scores)):
                hits[idx]["cross_score"] = cross_scores[idx]

            results = []
            hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
            # for hit in hits[:3]:
            #     results.append(self.paragraphs_topic[hit["corpus_id"]].replace("\n", " "))
            topic = self.topics[hits[0]["corpus_id"]]
            ret = {"text": topic, "error_code": 0}
        except Exception as e:
            ret = {"text": str(e), "error_code": 1}
        return ret
    
    def generate_gate(self, params):
        try:

            ret = {"text": "", "error_code": 0}
            ret = self.generate_stream_func(params)
            for x in ret:
                yield x
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": 50002,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": 50001,
            }
            yield json.dumps(ret).encode() + b"\0"

app = FastAPI()

# @dataclasses.dataclass
class ParamsEmbed(BaseModel):
    # fname: str = Query("./data_test/test.csv", description="Path of csv file")
    files: List[UploadFile] = File(...)
    window_size: int = Query(128)
    step_size: int = Query(50)

class ParamsQuery(BaseModel):
    query: str = Query("Please use the provided box as a reference to segment the objects.", description="Write your question")
    top_k: int = Query(16)

def release_model_semaphore():
    model_semaphore.release()

def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(limit_model_concurrency)
    return model_semaphore.acquire()

# @app.post("/worker_generate", dependencies=[Depends(ParamsQuery)])
@app.post("/worker_generate_topic")
async def api_generate(params: ParamsQuery = Depends()):
    # params = await request.json()
    print(params)
    await acquire_model_semaphore()
    output = worker.generate_topic(params)
    release_model_semaphore()
    return JSONResponse(output)

@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()

@app.post("/worker_embed_topic")
async def api_embed(files: List[UploadFile] = File(...)): # params = {"fname", "window_size", "step_size"}
    # print(params)
    print(files)
    fnames = []
    for file in files:
        uploaded_file = await file.read()
        csv_on_memory = BytesIO(uploaded_file)
        fnames.append(csv_on_memory)
    print(fnames)
    return worker.embed_csv(fnames)

@app.post("/worker_generate_doc")
async def api_generate(params: ParamsQuery = Depends()):
    # params = await request.json()
    print(params)
    await acquire_model_semaphore()
    output = worker.generate_gate(params)
    release_model_semaphore()
    # background_tasks = BackgroundTasks()
    # background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(output)

@app.post("/woker_embed_doc")
async def api_embed(params: ParamsEmbed = Depends()): # params = {"fname", "window_size", "step_size"}
    # params = await request.json()
    fnames = []
    for file in params.files:
        uploaded_file = await file.read()
        pdf_on_memory = BytesIO(uploaded_file)
        fnames.append(pdf_on_memory)
    # print(PDFParser(fnames[0]))
    return worker.embed_pdf(params, fnames)

if __name__=="__main__":
    host = "0.0.0.0"
    port = 21002
    worker_address = f"http://localhost:{port}"
    controller_address = controller_url
    worker_id = str(uuid.uuid4())[:6]
    no_register = False
    model_path = ""
    model_config = ""
    model_names = "retrieval_topic"
    device = "cuda"

    model_semaphore = None
    limit_model_concurrency = 5
    global_counter = 0

    worker = ModelWorker(controller_address,
                        worker_address,
                        worker_id,
                        no_register, 
                        model_path, 
                        model_config,
                        model_names, 
                        device)
    uvicorn.run(app, host=host, port=port, log_level="info")
