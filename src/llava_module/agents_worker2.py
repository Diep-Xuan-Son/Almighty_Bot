import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

# from base.libs import *
# import torch
from threading import Thread
from queue import Queue
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import base64

from base.constants import *
from llava_module.constants import *
from llava_module.conversation2 import Conversation
from prompts.prompts import (
	prompt_task_decompose,
	prompt_task_topology,
	prompt_choose_tool,
	prompt_choose_parameter,
	prompt_choose_parameter_depend,
	prompt_answer_generation,
	prompt_answer_summarize_2,
	prompt_answer_check,
	prompt_answer_inference,
	prompt_choose_tool_parameter
)
from base.agents import Agent
from langchain_groq import ChatGroq
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import tritonclient.grpc as grpcclient

logger_agent = logger.bind(name="logger_agent")
logger_agent.add(os.path.join(PATH_DEFAULT.LOGDIR, f"agent.{datetime.date.today()}.log"), mode='w')

# class Translate():
# 	def __init__(self,):
# 		self.device = "cuda"
# 		self.tokenizer_vi2en = AutoTokenizer.from_pretrained("./weights/vinai-translate-vi2en-v2", src_lang="vi_VN")
# 		self.model_vi2en =  AutoModelForSeq2SeqLM.from_pretrained("./weights/vinai-translate-vi2en-v2")
# 		self.model_vi2en.to(self.device)

# 		self.tokenizer_en2vi = AutoTokenizer.from_pretrained("./weights/vinai-translate-en2vi-v2", src_lang="en_XX")
# 		self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("./weights/vinai-translate-en2vi-v2")
# 		self.model_en2vi.to(self.device)

# 	def translate_vi2en(self, vi_texts: str) -> str:
# 		input_ids = self.tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(self.device)
# 		output_ids = self.model_vi2en.generate(
# 			**input_ids,
# 			decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
# 			num_return_sequences=1,
# 			num_beams=5,
# 			early_stopping=True
# 		)
# 		en_texts = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
# 		return en_texts[0]

# 	def translate_en2vi(self, en_texts: str) -> str:
# 		input_ids = self.tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(self.device)
# 		output_ids = self.model_en2vi.generate(
# 			**input_ids,
# 			decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
# 			num_return_sequences=1,
# 			num_beams=5,
# 			early_stopping=True
# 		)
# 		vi_texts = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
# 		return vi_texts[0]

class ServiceTrion():
	def __init__(self):
		self.client = grpcclient.InferenceServerClient(url=Configuration.tritonserver_url)
		if not self.client.is_server_ready():
			logger_agent.error("Failed to connect to triton server")
		if not self.client.is_model_ready("vinai_translate_vi2en"):
			logger_agent.error("Model translate vi2en is not ready")
		if not self.client.is_model_ready("vinai_translate_en2vi"):
			logger_agent.error("Model translate en2vi is not ready")

	def translate_vi2en(self, vi_texts: str) -> str:
		text = np.array([vi_texts])
		text = np.expand_dims(text, axis=0)
		text = np.char.encode(text, encoding = 'utf-8')
		input_tensors = [grpcclient.InferInput("texts", text.shape, "BYTES")]
		input_tensors[0].set_data_from_numpy(text)
		results = self.client.infer(model_name="vinai_translate_vi2en", inputs=input_tensors)
		en_texts = results.as_numpy("en_texts").astype(str)[0][0]
		return en_texts

	def translate_en2vi(self, en_texts: str) -> str:
		text = np.array([en_texts])
		text = np.expand_dims(text, axis=0)
		text = np.char.encode(text, encoding = 'utf-8')
		input_tensors = [grpcclient.InferInput("texts", text.shape, "BYTES")]
		input_tensors[0].set_data_from_numpy(text)
		results = self.client.infer(model_name="vinai_translate_en2vi", inputs=input_tensors)
		en_texts = results.as_numpy("vi_texts")[0][0].decode("utf-8")
		return en_texts

	def voice2text(self, sample, sampling_rate):
		samples = np.expand_dims(sample, axis=0).astype(np.float32)
		sampling_rate = np.array([[sampling_rate]], dtype=np.int16)
		input_tensors = [grpcclient.InferInput("samples", samples.shape, "FP32"), grpcclient.InferInput("sampling_rate", sampling_rate.shape, "INT16")]
		input_tensors[0].set_data_from_numpy(samples)
		input_tensors[1].set_data_from_numpy(sampling_rate)
		results = self.client.infer(model_name="phowhisper_voice2text", inputs=input_tensors)
		output_data = results.as_numpy("texts")[0]
		return output_data.decode("utf-8")
SERVICETRION = ServiceTrion()

class AgentGraph():
	def __init__(self,model_path="mixtral-8x7b-32768"):
		self.model_path = model_path
		self.llm = ChatGroq(temperature=0, groq_api_key="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", model_name=model_path)
		self.result_image_path = []
		self.q = Queue()

	# @estimate_execute_time("Run call function", logger_agent)
	def Call_function(self, api_skill, args, state):
		try:
			payload = {}
			headers = {}
			files = {}
			if 'file' in args:
				for file in args['file']:
					if file in ["image", "photo"]:
						if state.images[-1][-1] is None:
							return ("Tool cannot return answer, the reason is missing images")
						all_images = state.get_images()
						files = [("image", image) for image in all_images]

			if 'payload' in args:
				if len(args['payload'])!=0:
					payload = args['payload']

			#------------get list function----------
			# spec = importlib.util.spec_from_file_location('tools', Configuration.path_tool_function)
			# app_module = importlib.util.module_from_spec(spec)
			# spec.loader.exec_module(app_module)
			app_module = importlib.import_module("services.tools")
			list_func = np.array(inspect.getmembers(app_module, inspect.isfunction))[:,0].tolist()
			all_module_reload = [x[1] for x in sys.modules.items() if "services" in x[0]]
			mr = [importlib.reload(module) for module in all_module_reload]
			if api_skill in list_func:
				func = getattr(app_module, api_skill)
				if files:
					for file in files:
						payload[file[0]] = file[1]
				result = func(**payload)
				return result
			#////////////////////////////////////////
		
			if not api_skill.startswith(("http", "https")):
				api_skill = get_worker_addr(controller_url, api_skill)
				if api_skill == -1:
					return (f"Tool cannot return the answer because this tool has not been registered")
				api_skill += "/worker_generate"
			logger_agent.info("----api_skill: {}", api_skill)

			if isinstance(payload, (str)):
				payload = json.loads(payload)
			logger_agent.info("-----payload: {}", payload)
			try:
				res = requests.request("POST", url=api_skill, headers=headers, data=payload, files=files)
			except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
				return(f"Tool cannot return the answer because Unable to establish connection with tool {api_skill}.")

			if res.status_code == 422:
				return (f"Tool cannot return the answer because of missing prameter for tool")

			res = res.json()
			logger_agent.info("----res: {}", res)

			# print(res["Information"])
			# exit()
			if not res["success"]:
				error_res = res["error"]
				return (f"Tool cannot return the right answer, the reason is {error_res}")
			if "image" in res:
				pil_img = Image.open(BytesIO(base64.b64decode(res["image"])))
				path_img = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_IMAGE, f"{state._id}/result_{len(state.images)}.jpg"))
				image_np = np.array(pil_img)
				cv2.imwrite(path_img, image_np)
				self.result_image_path.append(path_img)
			if "information" in res:
				result = res["information"]
			return result
		except Exception as e:
			logger_agent.exception(e)
			return -1
	
	@estimate_execute_time("Run choose tool", logger_agent)
	def choose_tool(self, question, tool_descriptions, tool_usages):
		system_prompt = "You are an AI assistant that selects appropriate tools based on user queries and available data."
		agent_tool_param = Agent(system_prompt=system_prompt, prompt=prompt_choose_tool_parameter, llm=self.llm)
		
		ind = 0
		while True:
			try:
				response = agent_tool_param.run(question=question, tool_descriptions=tool_descriptions, tool_usage=tool_usages)
				logger_agent.info("----response: {}", response)
				result = agent_tool_param.clean_response_1(response=response)
				break 
			except Exception as e:
				logger_agent.exception(e)
				logger_agent.info("----response: {}", response)
				if ind > 1:
					return []
				ind += 1
				continue
		return result

	@estimate_execute_time("Run answer generation", logger_agent)
	def answer_generation(self, question, call_result):
		system_prompt = "You are a helpful AI assistant that generates answers from tool's result."
		agent_answer_generation = Agent(system_prompt=system_prompt, prompt=prompt_answer_generation, llm=self.llm)
		clean_answer = agent_answer_generation.run(question=question, call_result=call_result)
		clean_answer = clean_answer.replace("```", "").strip().split('\n\n')[-1]
		logger_agent.info("----answer: {}", clean_answer)
		return clean_answer


	def answer_summarize(self, question, answer_task):
		system_prompt = "You are a helpful AI assistant that summerize generated answer."
		agent_answer_summarize = Agent(system_prompt=system_prompt, prompt=prompt_answer_summarize_2, llm=self.llm)
		clean_answer = agent_answer_summarize.run(question=question, answer_task=answer_task)
		clean_answer = clean_answer.replace("```", "").strip().split('\n\n')[-1]
		return clean_answer

	@estimate_execute_time("Run answer check", logger_agent)
	def answer_check(self, question, answer):
		system_prompt = "You are a helpful AI assistant."
		agent_answer_check = Agent(system_prompt=system_prompt, prompt=prompt_answer_check, llm=self.llm)
		clean_answer = agent_answer_check.run(question=question, answer=answer)
		clean_answer = agent_answer_check.clean_response_2(response=clean_answer)
		logger_agent.info("----clean_answer: {}", clean_answer)
		if 'yes'.lower() in clean_answer["Choice"].lower():
			return 1
		else:
			return -1

	@estimate_execute_time("Run answer inference", logger_agent)
	def answer_inference(self, question):
		system_prompt = "You are a helpful AI assistant."
		agent_answer_inference = Agent(system_prompt=system_prompt, prompt=prompt_answer_inference, llm=self.llm)
		clean_answer = agent_answer_inference.run(question=question)
		clean_answer = agent_answer_inference.clean_response_2(response=clean_answer)
		return clean_answer

	def do_call_function(self, tool_name, api_tool, parameters, state):
		call_result = self.Call_function(api_tool, parameters, state)
		if call_result != -1:
			self.q.put({tool_name: str(call_result)})
		logger_agent.info(f"----{api_tool}: {call_result}")
	
	@estimate_execute_time("Run execute", logger_agent)
	def execute(self, state):
		question = state.messages[-1]["User"]
		tool_names = []
		tool_descriptions = ""
		tool_usages = ""
		for id, data in state.functions_data.items():
			tool_descriptions += f"{data['ID']}:  {data['API_description']}\n"
			tool_usages += f"For {data['ID']} tool:\n{str(data['Usage'])[1:-1]}\n" 
			tool_names.append(data['ID'])
		tool_list = self.choose_tool(question, tool_descriptions, tool_usages)
		logger_agent.info("----tool_list: {}", tool_list)

		if not tool_list:
			final_answer = self.answer_inference(question)
			return final_answer

		call_results = {}
		my_threads = []
		for tool in tool_list:
			if tool["Name"] not in tool_names:
				continue
			idx_tool = tool_names.index(tool["Name"])
			api_tool = list(state.functions_data.values())[idx_tool]["API_name"]
			# logger_agent.info("----api_tool: {}", api_tool)
			parameters = tool["Usage"]["Parameters"]
			my_thread = Thread(target=self.do_call_function, args=(tool['Name'], api_tool, parameters, state,))
			my_thread.start()
			my_threads.append(my_thread)
			# call_result = self.Call_function(api_tool, parameters, state)
			# if call_result == -1:
			# 	continue
			# logger_agent.info("----call_result: {}", call_result)
			# call_results[f"{tool['Name']}"] = str(call_result)
		for my_thread in my_threads:
			my_thread.join()
		while not self.q.empty():
			call_reuslt = self.q.get()
			call_results[list(call_reuslt.keys())[0]] = list(call_reuslt.values())[0]
		logger_agent.info(f"----call_results: {call_results}")
		answer = self.answer_generation(question, call_results)
		# check_answer = self.answer_check(question, answer)
		return answer
  
def bot_execute(state, model_selector, conversation_id):
	if state.use_knowledge:
		yield (state, state.chat) + (enable_btn,)*6
		return
	message = ""
	state.chat.append([None, "..."])
	yield (state, state.chat) + (disable_btn,)*6
	agent = AgentGraph(model_selector)
	answer = agent.execute(state)
	logger_agent.info("----answer_en: {}", answer)
	#--------------translate en2vi------------------
	answer_vi = SERVICETRION.translate_en2vi(answer)
	logger_agent.info("----answer_vi: {}", answer_vi)
	#///////////////////////////////////////////////
	state.messages[-1][state.roles[1]] = answer
	results_split = answer_vi.split(" ")
	for re in results_split:
		message += re + " "
		message_show = message + "▌"
		state.chat[-1] = [None, message_show]
		yield (state, state.chat) + (disable_btn,)*6
	state.chat[-1][1] = state.chat[-1][1][:-1]
	yield (state, state.chat) + (disable_btn,)*6

	for image_path in agent.result_image_path:
		state.chat.append((None, (image_path,)))
		yield (state, state.chat) + (enable_btn,)*6

	if not conversation_id:
		conversation_id = "conver_default"
	state.save_conversation(os.path.abspath(os.path.join(PATH_DEFAULT.PATH_CONVER, f"{conversation_id}.json")))
	return (state, state.chat) + (enable_btn,)*6

def bot_load_init(conversation_id):
	path_conver = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_CONVER, f"{conversation_id}.json"))
	if not os.path.exists(path_conver) or conversation_id=="":
		if conversation_id=="":
			conversation_id = "conver_default"
			path_image_default = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_IMAGE, f"{conversation_id}"))
			delete_folder_exist(path_image_default=path_image_default)

		path_conver = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_CONVER, f"{conversation_id}.json"))
		conversation = Conversation(_id = conversation_id, \
									roles = ["User", "Assistant"], \
									chat = [], \
									messages = [], \
									images = [], \
									voices = [], \
									image_process_mode = [], \
									tool_dic = [], \
									functions_data = {}, 
									use_knowledge = False)
		conversation.save_conversation(path_conver) # delete_after
		dataset = read_json(Configuration.path_tool_data)
		conversation.functions_data = dataset
	else:
		kwargs_conversation = json.load(open(path_conver))
		# print("-------kwargs_conversation: ", kwargs_conversation)   
		conversation = Conversation(**kwargs_conversation)
		
	path_image_conver = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_IMAGE, f"{conversation_id}"))
	check_folder_exist(path_image_conver=path_image_conver)
	return conversation

def bot_delete_conver(conversation_id):
	path_conver = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_CONVER, f"{conversation_id}.json"))
	conversation = Conversation(_id = "conver_default", \
								roles = ["User", "Assistant"], \
								chat = [], \
								messages = [], \
								images = [], \
								voices = [], \
								image_process_mode = [], \
								tool_dic = [], \
								functions_data = {})
	conversation.save_conversation(os.path.abspath(os.path.join(PATH_DEFAULT.PATH_CONVER, "conver_default.json"))) # delete_after 
	dataset = read_json(Configuration.path_tool_data)
	conversation.functions_data = dataset
	path_image_conver = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_IMAGE, f"{conversation_id}"))
	delete_folder_exist(path_conver=path_conver, path_image_conver=path_image_conver)
	return conversation

def add_text(state, textbox, image_dict, image_process_mode, knowledge_selector, n_result, conversation_id, with_debug_parameter_from_state=False):
	text = textbox['text']
	state.chat.append((text, None))
	if len(knowledge_selector)!=0:
		state.use_knowledge = True
		state.chat[-1] = [text, "..."]
		yield (state, state.chat, "", None) + (disable_btn,) * 6
		payload = {
			"collection_names": knowledge_selector,
			"text_query": text,
			"n_result": n_result
		}
		api_generate_doc = get_worker_addr(controller_url, "retrieval_docs") + "/worker_generate_doc"
		# print(api_generate_doc)
		headers = {}
		doc_response = requests.request("POST", url=api_generate_doc, headers=headers, data=payload)
		# print(doc_response)
		for chunk in doc_response.iter_lines(decode_unicode=False, delimiter=b"\0"):
			# print("-----chunk: ", chunk)
			if chunk:
				data = json.loads(chunk.decode())
				# print(data)
				if data["error_code"] == 0:
					output = data["text"].strip()
					message_show = output + "▌"
					state.chat[-1] = [text, message_show]
					yield (state, state.chat, "", None) + (disable_btn,) * 6
				else:
					state.use_knowledge = False
					output = data["text"] + \
						f" (error_code: {data['error_code']})"
					message_error = output
					# message_show = "Don't find necessary information for this question, change to using inference"
					message_show = "Không tìm thấy thông tin cần thiết cho câu hỏi này, chuyển sang sử dụng suy luận."
					state.chat[-1] = [text, message_show]
					yield (state, state.chat, "", None) + (disable_btn,) * 6
					# return
					# break
				time.sleep(0.01)
			else:
				state.chat[-1][1] = state.chat[-1][1][:-1]
				yield (state, state.chat, "", None) + (disable_btn,) * 6
		# print(state.use_knowledge)
		if state.use_knowledge:
			if not conversation_id:
				conversation_id = "conver_default"
			state.save_conversation(os.path.abspath(os.path.join(PATH_DEFAULT.PATH_CONVER, f"{conversation_id}.json")))
			yield (state, state.chat, "", None) + (enable_btn,) * 6
			return
	state.use_knowledge = False
	logger_agent.info("----use_knowledge: {}", state.use_knowledge)
	#----------------translate vi2en-------------------------
	vi_text = text
	text = SERVICETRION.translate_vi2en(text)
	logger_agent.info("------en_text: {}", text)
	#////////////////////////////////////////////////////////
	# print(image_dict)
	# print(image_process_mode)
	if image_dict is not None:
		pil_img = Image.open(BytesIO(image_dict['image']))
		path_img = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_IMAGE, f"{state._id}/{len(state.images)}.jpg"))
		pil_img.save(path_img)
		state.images.append([path_img])
	elif textbox['files']:
		path_imgs = []
		for i, img_path in enumerate(textbox['files']):
			path_img = os.path.abspath(os.path.join(PATH_DEFAULT.PATH_IMAGE, f"{state._id}/{len(state.images)}_{i}.jpg"))
			if img_path.endswith(("jpg", "png", "jpeg")):
				shutil.copy2(img_path, path_img)
				path_imgs.append(path_img)
		if not path_imgs:
			state.images.append([None])
		else:
			state.images.append(path_imgs)
	else:
		state.images.append([None])
	state.image_process_mode.append(image_process_mode)
	message = {state.roles[0]: str(text), state.roles[1]: ""}
	state.messages.append(message)
	# state.chat.append((state.messages[-1][state.roles[0]], None))
	for img_path in state.images[-1]:
		if img_path is not None:
			state.chat.append(((img_path,), None))
	# state.save_conversation(os.path.abspath(os.path.join(PATH_CONVER, "conver_default.json")))
	return (state, state.chat, "", None) + (disable_btn,) * 6

# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# PROCESSOR_VOICE = WhisperProcessor.from_pretrained("weights/PhoWhisper-small")
# MODEL_VOICE = WhisperForConditionalGeneration.from_pretrained("weights/PhoWhisper-small")
def add_voice(state, record_dict, image_dict, image_process_mode, knowledge_selector, n_result, conversation_id):
	yield (state, state.chat) + (disable_btn, )*7
	sampling_rate = record_dict["sampling_rate"]
	sample = np.frombuffer(record_dict["sample"], dtype=np.float32)
	# print(sample)
	#----------------------------------
 
	# inputs = PROCESSOR_VOICE(sample/sampling_rate, sampling_rate=sampling_rate, return_tensors="pt").input_features
	# predicted_ids = MODEL_VOICE.generate(inputs, language="vi")
	# transcription = PROCESSOR_VOICE.batch_decode(predicted_ids, skip_special_tokens=True)
	# print(transcription[0])
	# vi_text = transcription[0]
	vi_text = SERVICETRION.voice2text(sample, sampling_rate)
	#-----------------------------retrieval----------------------------
	if len(knowledge_selector)!=0:
		state.use_knowledge = True
		state.chat.append([vi_text, "..."])
		yield (state, state.chat, "", None) + (disable_btn,) * 6
		payload = {
			"collection_names": knowledge_selector,
			"text_query": vi_text,
			"n_result": n_result
		}
		api_generate_doc = get_worker_addr(controller_url, "retrieval_docs") + "/worker_generate_doc"
		# print(api_generate_doc)
		headers = {}
		doc_response = requests.request("POST", url=api_generate_doc, headers=headers, data=payload)
		# print(doc_response)
		for chunk in doc_response.iter_lines(decode_unicode=False, delimiter=b"\0"):
			# print("-----chunk: ", chunk)
			if chunk:
				data = json.loads(chunk.decode())
				# print(data)
				if data["error_code"] == 0:
					output = data["text"].strip()
					message_show = output + "▌"
					state.chat[-1] = [vi_text, message_show]
					yield (state, state.chat) + (disable_btn,)*7
				else:
					state.use_knowledge = False
					output = data["text"] + \
						f" (error_code: {data['error_code']})"
					message_error = output
					message_show = "Don't find necessary information for this question, change to using inference"
					state.chat[-1] = (vi_text, message_show)
					yield (state, state.chat) + (disable_btn,)*7
					return
				time.sleep(0.01)
			else:
				state.chat[-1][1] = state.chat[-1][1][:-1]
				yield (state, state.chat) + (disable_btn,)*7
		# print(state.use_knowledge)
		if state.use_knowledge:
			if not conversation_id:
				conversation_id = "conver_default"
			state.save_conversation(os.path.abspath(os.path.join(PATH_DEFAULT.PATH_CONVER, f"{conversation_id}.json")))
			yield (state, state.chat) + (enable_btn,)*7
			return
	#//////////////////////////////////////////////////////////////////

	text = SERVICETRION.translate_vi2en(vi_text)
	print(text)

	yield (state, state.chat) + (enable_btn,)*7
	return


if __name__=="__main__":
	model_path = "mixtral-8x7b-32768"
	param_conver = {"_id": "conver1", \
					"roles": ["User", "Assistant"], \
					"chat": [], \
					"messages": [], \
					"images": [], \
					"voices": [], \
					"image_process_mode": [], \
					"tool_dic": [], \
					"functions_data": {},\
					"use_knowledge": False}
	default_conversation = Conversation(**param_conver)
	dataset = read_json(Configuration.path_tool_data)
	default_conversation.functions_data = dataset
	default_conversation.messages.append({default_conversation.roles[0]: "What is the weather like in Da Nang Vietnam?", default_conversation.roles[1]: ""})
	default_conversation.images.append(["/home/mq/disk2T/son/code/GitHub/MQ_GPT/src/hieu.jpg"])
	default_conversation.image_process_mode.append("Crop")

	agent = AgentGraph(model_path)
	agent.execute(default_conversation)
 
 
 
# res.status_code = 422 : param missing
# res.status_code = 500 : error tool
# try:
#     response = requests.options(url)
#     if response.ok:   # alternatively you can use response.status_code == 200
#         print("Success - API is accessible.")
#     else:
#         print(f"Failure - API is accessible but sth is not right. Response codde : {response.status_code}")
# except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
#     print(f"Failure - Unable to establish connection: {e}.")
# except Exception as e:
#     print(f"Failure - Unknown error occurred: {e}.")


# a = {"1":1, "2":2, "3":3, "4":4}
# del a["2"]
# k_new = list(range(len(a.keys())))
# k_new = list(map(str, k_new))
# a_new = dict(zip(k_new, list(a.values())))
