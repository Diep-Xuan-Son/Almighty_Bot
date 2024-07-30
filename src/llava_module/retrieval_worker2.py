import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# from base.libs import *
from unstructured.partition.auto import partition
import numpy as np
from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import StreamingResponse
from typing import List
import uvicorn
import uuid
import socket

from base.service import *
import chromadb
from chromadb.utils import embedding_functions
# from sentence_transformers import CrossEncoder

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All, Ollama, HuggingFaceHub

logger_retrieval = logger.bind(name="logger_retrieval")
logger_retrieval.add(os.path.join(PATH_DEFAULT.LOGDIR, f"retrieval_worker.{datetime.date.today()}.log"), mode='w')

def pdf_chunk(pdf_file, chunk_size):
    elements = partition(file=pdf_file, strategy="fast")
    text = "\n".join([str(el) for el in elements])
    text = " ".join(text.split())
    #---------------------------------------
    text_tokens = text.split()
    sentences = []
    for i in range(0, len(text_tokens), 50):
        window = text_tokens[i : i + 128]
        if len(window) < 128:
            break
        sentences.append(window)
    chunks = [" ".join(s) for s in sentences]
    return chunks

class ModelWorker:
    def __init__(self, client, embedding_func, controller_addr, worker_addr, worker_id, no_register, model_names, device):
        self.client = client
        self.embedding_func = embedding_func
        self.cross_encoder = CrossEncoder("./weights/ms-marco-MiniLM-L-6-v2")
        self.collection_list = {}
        self.chunk_sz = 1000

        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.device = device   
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger_retrieval.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
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

    def get_list_file_name(self, collection_name):
        # return self.collection_list[collection_name]
        collection = self.client.get_collection(name=collection_name)
        all_metadatas = collection.get(include=["metadatas"]).get('metadatas')
        list_file_name = list(map(lambda x: x.get("file_name",None), all_metadatas))
        list_file_name = list(set(list_file_name))
        return list_file_name
    def add_list_file(self, collection_name, list_file_name, list_file):
        if len(list_file_name) == 0:
            return {"success": False, "error": "Don't have any files"}
        if collection_name in self.collection_list:
            for i, file_name in reversed(list(enumerate(list_file_name))):
                if file_name in self.collection_list[collection_name]:
                    list_file_name.pop(i)
                    list_file.pop(i)
            self.collection_list[collection_name] += list_file_name
        else:
            self.collection_list[collection_name] = list_file_name
        #--------------------------------------------------------
        collection_list = [cl.name for cl in self.client.list_collections()]
        if collection_name not in collection_list:
            return {"success": False, "error": f"Collection {collection_name} have not been registered yet"}
        collection = self.client.get_collection(name=collection_name, embedding_function=self.embedding_func)
        for i, file in enumerate(list_file):
            file_name = list_file_name[i]
            dt_file = collection.get(where={"file_name": {"$eq": file_name}})
            print(len(dt_file["ids"]))
            if len(dt_file["ids"]) != 0:
                continue
            documents = pdf_chunk(file, self.chunk_sz)
            print(len(documents))
            collection.add(
                documents=documents,
                ids=[f"{file_name}_{j}" for j in range(len(documents))],
                metadatas=[{"file_name": file_name} for j in range(len(documents))]
            )
        return {"success": True}
    def delete_list_file(self, collection_name, list_file_name):
        if len(list_file_name) == 0:
            return {"success": False, "error": "Don't have any files"}
        if collection_name in self.collection_list:
            for i, file_name in reversed(list(enumerate(self.collection_list[collection_name]))):
                if file_name in list_file_name:
                    self.collection_list[collection_name].pop(i)
        #---------------------------------------------------------
        collection_list = [cl.name for cl in self.client.list_collections()]
        if collection_name not in collection_list:
            return {"success": False, "error": f"Collection {collection_name} have not been registered yet"}
        collection = self.client.get_collection(name=collection_name)
        collection.delete(where={"file_name": {"$in": list_file_name}})
        return {"success": True}
    def get_list_collection(self):
        collection_list = [cl.name for cl in self.client.list_collections()]
        return collection_list
    def add_collection(self, collection_name):
        self.collection_list[collection_name] = []
        #--------------------------------------------
        collection_list = [cl.name for cl in self.client.list_collections()]
        if collection_name in collection_list:
            return {"success": False, "error": f"Collection {collection_name} already exists"}
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"},
        )
        return {"success": True}
    def delete_collection(self, collection_name):
        if collection_name in self.collection_list:
            self.collection_list.pop(collection_name)
        #----------------------------------------------
        collection_list = [cl.name for cl in self.client.list_collections()]
        if collection_name not in collection_list:
            return {"success": False, "error": f"Collection {collection_name} have not been registered yet"}
        self.client.delete_collection(name=collection_name)
        return {"success": True}

    def test_query(self, collection_names, text_querys, n_result):
        results = []
        results_dis = []
        for collection_name in collection_names:
            collection = chroma_client.get_collection(name=collection_name, embedding_function=self.embedding_func)
            query_results = collection.query(
                                query_texts=text_querys,
                                n_results=n_result,
                            )
            # print(query_results)
            if len(results) == 0:
                results = np.array(query_results["documents"])
                results_dis = np.array(query_results["distances"])
            else:
                results = np.concatenate((results, query_results["documents"]), axis=1)
                results_dis = np.concatenate((results_dis, query_results["distances"]), axis=1)

        sorted_results = [results[i][np.where(ldis<0.6)[0]].tolist() for i, ldis in enumerate(results_dis)]
        # sorted_results = results
        # sorted_results = []
        # for i, query_results in enumerate(results):
        #     cross_input = [[text_querys[i], query_result] for query_result in query_results]
        #     cross_scores = self.cross_encoder.predict(cross_input)
        #     inds = np.argsort(cross_scores)[::-1]
        #     query_results = np.array(query_results)[inds]
        #     sorted_results.append(query_results[:n_result])
        print(sorted_results)
        use_ctx = True
        if len(sorted_results[0])==0:
            use_ctx = False
        return sorted_results, use_ctx

    def generate_stream_func(self, params):
        collection_names = params.collection_names
        text_query = params.text_query
        n_result = params.n_result
        results, use_ctx = self.test_query(collection_names, [text_query], int(n_result))
        # print(results[0])
        # print(use_ctx)
        if not use_ctx:
            ret = {"text": "", "error_code": 1}
            yield json.dumps(ret).encode() + b"\0"
            return
        result_ctx = "\n\n".join(results[0])
        message = ""
        # print("----results: ", results)
        # print("----text_query: ", text_query)
        #-----------llm generate text--------------------
        chat = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token="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", model_kwargs={"max_new_tokens":512})
        system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "Use the following pieces of context to answer the user's question.\n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
            "If you answer by english, the user will not understand, please answer by vietnamese.\n"
            "'''\n"
            "This is the context: {context}"
            "'''\n"
            "This is the user's question: {question}\n"
            "Output:\n\n"
        )
        # chat = HuggingFaceHub(repo_id="vilm/vinallama-2.7b-chat", huggingfacehub_api_token="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", model_kwargs={"max_new_tokens":240})
        # system_message_prompt = SystemMessagePromptTemplate.from_template("Bạn là một trợ lí AI hữu ích.")
        # human_message_prompt = HumanMessagePromptTemplate.from_template(
        #     "Sử dụng những đoạn văn bản dưới đây để trả lời câu hỏi của người dùng.\n"
        #     "Nếu bạn không biết câu trả lời, hãy chỉ trả lời tôi không biết, đừng cố gắng để tạo ra một câu trả lời.\n"
        #     "'''\n"
        #     "Đây là đoạn văn bản: {context}"
        #     "'''\n"
        #     "Đây là câu hỏi của người dùng: {question}\n"
        #     "Câu trả lời:\n\n"
        # )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        result = chain.run(question=text_query, context=result_ctx)
        # print(result)
        clear_answer = result.replace("```", "").strip().split('\n\n')[-1]
        #////////////////////////////////////////////////
        results_split = clear_answer.split(" ")
        for re in results_split:
            message += re + " "
            ret = {"text": message, "error_code": 0}
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        try:

            ret = {"text": "", "error_code": 0}
            ret = self.generate_stream_func(params)
            # print(ret)
            for x in ret:
                yield x
        # except torch.cuda.OutOfMemoryError as e:
        #     ret = {
        #         "text": f"{Configuration.server_error_msg}\n\n({e})",
        #         "error_code": 50002,
        #     }
        #     yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{Configuration.server_error_msg}\n\n({e})",
                "error_code": 50001,
            }
            yield json.dumps(ret).encode() + b"\0"

app = FastAPI(docs_url="/docs")

class ParamsQuery(BaseModel):
    collection_names: List[str] = ["Collection_2"]
    text_query: str = ""
    n_result: int = 1

    # @model_validator(mode='before')
    # @classmethod
    # def validate_to_json(cls, value):
    #     if isinstance(value, str):
    #         return cls(**json.loads(value))
    #     return value

    @classmethod
    def as_form(
        cls,
        collection_names: List[str] = Form(["Collection_2"]),
        text_query: str = Form(""),
        n_result: int = Form(1)
    ):
        if len(collection_names)==1 and ("," in collection_names):
            collection_names = collection_names[0].split(",")
        return cls(collection_names=collection_names, text_query=text_query, n_result=n_result)

def release_model_semaphore():
    model_semaphore.release()

def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(limit_model_concurrency)
    return model_semaphore.acquire()

@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()
    

@app.get("/worker_get_collection")
async def api_get_collection(request: Request):
    return {"list_collection": worker.get_list_collection()}

@app.post("/worker_generate_doc")
async def api_generate(params: ParamsQuery = Depends(ParamsQuery.as_form)):
# async def api_generate(params: ParamsQuery = Body(...)):
    # params = await request.json()
    print(params)
    await acquire_model_semaphore()
    output = worker.generate_gate(params)
    release_model_semaphore()
    return StreamingResponse(output)

if __name__ == "__main__":
    host = str(urlparse(AddressWorker.RETRIEVAL_WORKER_URL).hostname)
    port = int(urlparse(AddressWorker.RETRIEVAL_WORKER_URL).port)
    worker_address = AddressWorker.RETRIEVAL_WORKER_URL
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_real = s.getsockname()[0]
        worker_address = f"http://{ip_real}:{port}"
    except:
        pass
    controller_address = AddressWorker.CONTROLLER_URL
    worker_id = str(uuid.uuid4())[:6]
    no_register = False
    model_names = ModelName.RETRIEVAL_WORKER
    device = "cpu"

    model_semaphore = None
    limit_model_concurrency = 5
    global_counter = 0

    EMBED_MODEL = "./weights/paraphrase-multilingual-mpnet-base-v2" #"./weights/paraphrase-multilingual-MiniLM-L12-v2",   "ms-marco-MiniLM-L-6-v2"
    #----------chromadb------------
    host_db = str(urlparse(Configuration.chroma_url).hostname)
    port_db = str(urlparse(Configuration.chroma_url).port)
    chroma_client = chromadb.HttpClient(host=host_db, port=port_db)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device=device, trust_remote_code=True)
    #//////////////////////////////

    worker = ModelWorker(chroma_client, 
                        embedding_func,
                        controller_address,
                        worker_address,
                        worker_id,
                        no_register, 
                        model_names, 
                        device)

    uvicorn.run(app, host=host, port=port, log_level="info")