import asyncio
from multiprocessing import Process
from time import sleep, time
from threading import Thread
from queue import Queue
import requests 
import pickle 

q = Queue()

def post_rq(id_thread):
    print("----id_thread: ", id_thread)
    header = {}
    payloads = {}
    params = {"id_thread": id_thread}
    files = [("image", open("./data_test/Mun_coc.jpg", "rb"))]
    ret = requests.request("POST", url="http://192.168.6.161:21004/worker_generate", headers=header, data=payloads, params=params, files=files)
    print(f"-------Thread {id_thread}: {ret.json()['id_thread']==id_thread}")
    data = {f"{id_thread}": "abc"}
    q.put(data)

def run_multi_thread(id_proc):
    print("----id_proc: ", id_proc)
    num_thread = 10
    my_threads = []
    for i in range(num_thread):
        my_thread = Thread(target=post_rq, args=(id_proc*num_thread + i,))
        my_thread.start()
        my_threads.append(my_thread)

    for my_thread in my_threads:
        my_thread.join()
        
    while not q.empty():
        print(q.get())
        
if __name__=="__main__":
    start = time()
    procs = []
    num_proc = 5
    for i in range(num_proc):
        proc = Process(target=run_multi_thread, args=(i,))
        proc.start()
        print(f"Process ID: {proc.pid}")
        procs.append(proc)

    for proc in procs:
        proc.join() # Wait for the process to finish

    print(f"Total time: {time() - start}s")
    # while not q.empty():