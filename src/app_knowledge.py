import os
import numpy as np
import re
import time

from unstructured.partition.auto import partition

import gradio as gr 
import logging
#///////////////////////////////////////
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All, Ollama, HuggingFaceHub
# from base.libs import *
# from base.constants import *
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
CHROMA_DATA_PATH = "./database/chroma_data/"
EMBED_MODEL = "./weights/paraphrase-multilingual-mpnet-base-v2" #"./weights/paraphrase-multilingual-MiniLM-L12-v2",   "ms-marco-MiniLM-L-6-v2"
# COLLECTION_NAME = "demo_docs1"

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
    #////////////////////////////////////////
    # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    # current_chunk = ""
    # chunks = []
    # k_sentence_chunk = []
    # for sentence in sentences:
    #     if len(current_chunk) + len(sentence) <= chunk_size:
    #         current_chunk += sentence + " "
    #         k_sentence_chunk.insert(0, sentence)
    #         if len(k_sentence_chunk) > 2:
    #             k_sentence_chunk.pop()
    #     else:
    #         chunks.append(current_chunk.strip())
    #         current_chunk = " ".join(k_sentence_chunk) + " " + sentence + " "
    # if current_chunk:
    #     chunks.append(current_chunk.strip())
    return chunks

class ModelWorker:
    def __init__(self,client=None, embedding_func=None):
        self.client = client
        self.embedding_func = embedding_func
        self.cross_encoder = CrossEncoder("./weights/ms-marco-MiniLM-L-6-v2")
        self.collection_list = {}
        self.chunk_sz = 1000

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
        for collection_name in collection_names:
            collection = chroma_client.get_collection(name=collection_name, embedding_function=self.embedding_func)
            query_results = collection.query(
                                query_texts=text_querys,
                                n_results=16,
                            )
            if len(results) == 0:
                results = query_results["documents"]
            else:
                results = np.concatenate((results, query_results["documents"]), axis=1)
        sorted_results = []
        for i, query_results in enumerate(results):
            cross_input = [[text_querys[i], query_result] for query_result in query_results]
            cross_scores = self.cross_encoder.predict(cross_input)
            inds = np.argsort(cross_scores)[::-1]
            query_results = np.array(query_results)[inds]
            sorted_results.append(query_results[:n_result])
        # print(results)
        return sorted_results


def get_previous_status():
    collection_list = worker.get_list_collection()
    global collection_count
    collection_count = len(collection_list)
    list_checkbox = []
    for collection_name in collection_list:
        file_names = worker.get_list_file_name(collection_name)
        list_checkbox.append(gr.CheckboxGroup(choices=file_names, value=[]))
    return (gr.Row(visible=True),)*collection_count + \
            (gr.Row(visible=False),)*(5-collection_count) + \
            tuple(list_checkbox) + (gr.CheckboxGroup(choices=[], value=[]),)*(5-len(list_checkbox))

def submit_file(files, collection_name):
    pattern = r'<h1>(.*)</h1>'
    collection_name = re.findall(pattern, collection_name, re.DOTALL)[0]
    collection_name = collection_name.replace(" ", "_")
    print(collection_name)
    if files is not None:
        file_names = [os.path.basename(file.name) for file in files]
        list_file = [open(path_file.name,mode='rb') for path_file in files]
        res = worker.add_list_file(collection_name, file_names, list_file)
        print(res)
    file_names = worker.get_list_file_name(collection_name)
    return gr.CheckboxGroup(choices=file_names, value=[])

def delete_file(file_checkbox, collection_name):
    pattern = r'<h1>(.*)</h1>'
    collection_name = re.findall(pattern, collection_name, re.DOTALL)[0]
    collection_name = collection_name.replace(" ", "_")
    if file_checkbox is not None:
        res = worker.delete_list_file(collection_name, file_checkbox)
        print(res)
    file_paths = worker.get_list_file_name(collection_name)
    return gr.CheckboxGroup(choices=file_paths, value=[])

def create_collection():
    collection_names = ["Collection_1", "Collection_2", "Collection_3", "Collection_4", "Collection_5"]
    global collection_count
    if collection_count < 5:
        res = worker.add_collection(collection_names[collection_count])
        print(res)
        collection_count += 1
    return (gr.Row(visible=True),)*collection_count + (gr.Row(visible=False),)*(5-collection_count)

def delete_collection():
    collection_names = ["Collection_1", "Collection_2", "Collection_3", "Collection_4", "Collection_5"]
    global collection_count
    if collection_count > 0:
        collection_count -= 1
    res = worker.delete_collection(collection_names[collection_count])
    print(res)
    return (gr.Row(visible=True),)*collection_count + (gr.Row(visible=False),)*(5-collection_count)

def get_registered_collection():
    collection_list = worker.get_list_collection()
    return gr.Dropdown(choices=collection_list, value=[])

def bot_query(collection_names, text_query, n_result, chatbox):
    results = worker.test_query(collection_names, [text_query], int(n_result))[0]
    results = "\n\n".join(results)
    # print(results)
    message = ""
    chatbox.append([text_query, message])
    #-----------llm generate text--------------------
    chat = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token="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", model_kwargs={"max_new_tokens":512})
    template = "You are a helpful assistant."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "Use the following pieces of context to answer the user's question.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
        "'''\n"
        "This is the context: {context}"
        "'''\n"
        "This is the user's question: {question}\n"
        "Output:\n\n"
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(question=text_query, context=results)
    # print(result)
    results = result.replace("```", "").strip().split('\n\n')[-1]
    #////////////////////////////////////////////////
    results_split = results.split(" ")
    for re in results_split:
        message += re + " "
        message_show = message + "▌"
        chatbox[-1] = [text_query, message_show]
        yield chatbox
    # chatbox[-1] = [text_query, message]
    return
    # chatbox.append([text_query, results])
    # return chatbox

def build_demo():
    create_collection_button = gr.Button(value="Create a new collection", visible=True)
    delete_collection_button = gr.Button(value="Delete a collection", visible=True)
    # file_checkbox = gr.CheckboxGroup(choices= ["doc1.pdf","doc2.pdf","doc3.pdf"], label="List file", type="value", visible=True)
    # file_output = gr.File(file_types=['.pdf'], file_count="multiple")
    # submit_button = gr.UploadButton("Click to Upload a File", file_types=[".pdf"], file_count="multiple")
    # submit_button = gr.Button(value="Submit", visible=True)

    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # Knowledge Storage
        Create your own collection and add files into that.
        """)
        create_collection_button.render()
        delete_collection_button.render()

        with gr.Row(elem_id="collection_1", variant="panel", visible=False) as demo_collection1:
            with gr.Row():
                mkd1 = gr.Markdown("""# Collection 1""")
            with gr.Column(scale=6):
                file_checkbox1 = gr.CheckboxGroup(choices= [], label="List file", type="value", visible=True)
            with gr.Column(scale=3):
                file_output1 = gr.File(file_types=['.pdf'], file_count="multiple")
                submit_button1 = gr.Button(value="Submit", visible=True)
                submit_button1.click(submit_file, [file_output1, mkd1], [file_checkbox1])
                delete_button1 = gr.Button(value="Delete selected file", visible=True)
                delete_button1.click(delete_file, [file_checkbox1, mkd1], [file_checkbox1])

        with gr.Row(elem_id="collection_2", variant="panel", visible=False) as demo_collection2:
            with gr.Row():
                mkd2 = gr.Markdown(f"""# Collection 2""")
            with gr.Column(scale=6):
                file_checkbox2 = gr.CheckboxGroup(choices= [], label="List file", type="value", visible=True)
            with gr.Column(scale=3):
                file_output2 = gr.File(file_types=['.pdf'], file_count="multiple")
                submit_button2 = gr.Button(value="Submit", visible=True)
                submit_button2.click(submit_file, [file_output2, mkd2], [file_checkbox2])
                delete_button2 = gr.Button(value="Delete selected file", visible=True)
                delete_button2.click(delete_file, [file_checkbox2, mkd2], [file_checkbox2])

        with gr.Row(elem_id="collection_3", variant="panel", visible=False) as demo_collection3:
            with gr.Row():
                mkd3 = gr.Markdown("""# Collection 3""")
            with gr.Column(scale=6):
                file_checkbox3 = gr.CheckboxGroup(choices= [], label="List file", type="value", visible=True)
            with gr.Column(scale=3):
                file_output3 = gr.File(file_types=['.pdf'], file_count="multiple")
                submit_button3 = gr.Button(value="Submit", visible=True)
                submit_button3.click(submit_file, [file_output3, mkd3], [file_checkbox3])
                delete_button3 = gr.Button(value="Delete selected file", visible=True)
                delete_button3.click(delete_file, [file_checkbox3, mkd3], [file_checkbox3])

        with gr.Row(elem_id="collection_4", variant="panel", visible=False) as demo_collection4:
            with gr.Row():
                mkd4 = gr.Markdown("""# Collection 4""")
            with gr.Column(scale=6):
                file_checkbox4 = gr.CheckboxGroup(choices= [], label="List file", type="value", visible=True)
            with gr.Column(scale=3):
                file_output4 = gr.File(file_types=['.pdf'], file_count="multiple")
                submit_button4 = gr.Button(value="Submit", visible=True)
                submit_button4.click(submit_file, [file_output4, mkd4], [file_checkbox4])
                delete_button4 = gr.Button(value="Delete selected file", visible=True)
                delete_button4.click(delete_file, [file_checkbox4, mkd4], [file_checkbox4])

        with gr.Row(elem_id="collection_5", variant="panel", visible=False) as demo_collection5:
            with gr.Row():
                mkd5 = gr.Markdown("""# Collection 5""")
            with gr.Column(scale=6):
                file_checkbox5 = gr.CheckboxGroup(choices= [], label="List file", type="value", visible=True)
            with gr.Column(scale=3):
                file_output5 = gr.File(file_types=['.pdf'], file_count="multiple")
                submit_button5 = gr.Button(value="Submit", visible=True)
                submit_button5.click(submit_file, [file_output5, mkd5], [file_checkbox5])
                delete_button5 = gr.Button(value="Delete selected file", visible=True)
                delete_button5.click(delete_file, [file_checkbox5, mkd5], [file_checkbox5])

        submit_collection = gr.Button(value="Save all collection", visible=True)
        with gr.Row(elem_id="chat_bot", variant="panel", visible=True) as chat_bot:
            with gr.Column(scale=3):
                n_result = gr.Number(value=1)
                collection_select = gr.Dropdown(choices=[], value=[], multiselect=True, visible=True)
            with gr.Column(scale=6):
                chatbox = gr.Chatbot()
                textbox = gr.Textbox()

        create_collection_button.click(create_collection, [], \
            [demo_collection1, demo_collection2, demo_collection3, demo_collection4, demo_collection5])
        delete_collection_button.click(delete_collection, [], \
            [demo_collection1, demo_collection2, demo_collection3, demo_collection4, demo_collection5])
        submit_collection.click(get_registered_collection, [], [collection_select])
        textbox.submit(bot_query, [collection_select, textbox, n_result, chatbox], [chatbox])

        demo.load(get_previous_status, [], \
            [demo_collection1, demo_collection2, demo_collection3, demo_collection4, demo_collection5, \
            file_checkbox1, file_checkbox2, file_checkbox3, file_checkbox4, file_checkbox5, ])
    return demo

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8887
    share = False
    #config queue gradio
    api_open = False
    max_size = 100

    global collection_count
    collection_count = 0

    #----------chromadb------------
    host_db = '0.0.0.0'
    port_db = 8008
    chroma_client = chromadb.HttpClient(host=host_db, port=port_db)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cuda", trust_remote_code=True)
    worker = ModelWorker(client=chroma_client, embedding_func=embedding_func)
    #//////////////////////////////
    demo = build_demo()
    demo.queue(
        api_open = api_open,
        max_size = max_size
    ).launch(
        server_name = host,
        server_port = port,
        share = share
    )
    # demo.launch(server_name = host,server_port = port,share = share)