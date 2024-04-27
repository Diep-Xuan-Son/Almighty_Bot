import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

from base.libs import *
from base.constants import *
from llava_module.constants import *
from llava_module.conversation2 import Conversation
from langchain.prompts import (
	ChatPromptTemplate,
	MessagesPlaceholder,
	SystemMessagePromptTemplate,
	HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All, Ollama, HuggingFaceHub
import importlib.util

PATH_CONVER = os.path.abspath(f"{ROOT}/history")
PATH_IMAGE = os.path.abspath(os.path.join(PATH_CONVER, "images"))
PATH_VOICES = os.path.abspath(os.path.join(PATH_CONVER, "voices"))
check_folder_exist(path_conver=PATH_CONVER, path_image=PATH_IMAGE, path_voices=PATH_VOICES)

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
def change_name(name):
	change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
	if name in change_list:
		name = "is_" + name.lower()
	return name

class Agents():
	def __init__(self,model_path="mistralai/Mixtral-8x7B-Instruct-v0.1"):
		self.model_path = model_path
		# self.previous_log = {}
		# self.tool_used = []
		# self.answer_bad = []
		# self.answer_good = []
		self.result_image_path = []

	def Call_function(self, api_skill, args, state):
		payload = {}
		headers = {}
		files = {}
		if 'file' in args:
			for file in args['file']:
				if file in ["image", "photo"]:
					if state.images[-1][-1] is None:
						return ("Tool cannot return anwser, the reason is missing images")
					all_images = state.get_images()
					files = [("image", image) for image in all_images]
		if 'payload' in args:
			for param, value in args['payload'].items():
				payload[param] = value
		if not api_skill.startswith(("http", "https")):
			api_skill = get_worker_addr(controller_url, api_skill) + "/worker_generate"
		print(api_skill)
		print("-----payload: ", payload)
		res = requests.request("POST", url=api_skill, headers=headers, data=payload, files=files).json()
		# print(res)
		# print(res["Information"])
		# exit()
		if not res["success"]:
			error_res = res["error"]
			return (f"Tool cannot return the right anwser, the reason is {error_res}")
		if "image" in res:
			pil_img = Image.open(BytesIO(base64.b64decode(res["image"])))
			path_img = os.path.abspath(os.path.join(PATH_IMAGE, f"{state._id}/result_{len(state.images)}.jpg"))
			image_np = np.array(pil_img)
			cv2.imwrite(path_img, image_np)
			self.result_image_path.append(path_img)
		return res["Information"]

	def task_decompose(self, question, Tool_dic):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")      # "google/gemma-7b", "mistralai/Mixtral-8x7B-Instruct-v0.1"
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.\n"
			"This is the user's question: {question}\n"
			"This is tool list:\n"
			"{Tool_list}\n"
			"Please note that: \n"
			"1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using one single tool in the tool list.\n"
			"2. If one subtask need the results from other subtask, you can should write clearly. For example:"
			"{{\"Tasks\": [\"Convert 23 km/h to X km/min by 'divide_'\", \"Multiply X km/min by 45 min to get Y by 'multiply_'\"]}}\n"
			"3. You must ONLY output in a parsible JSON format. An example output looks like:\n"
			# "'''\n"
			"{{\"Tasks\": [string 1, string 2, ...]}}\n"
			# "'''\n"
			"Output:\n\n"
		)
		template_human = human_message_prompt.prompt.template
		input_variables = human_message_prompt.prompt.input_variables
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		# print(chat_prompt.__dict__)
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		Tool_list = []
		for ele in Tool_dic:
			Tool_list.append(ele)

		ind = 0
		while True:
			try:
				result = chain.run(question=question, Tool_list=Tool_list)
				# print(result)
				# exit()
				pattern = r'{(.*)}'
				result = re.findall(pattern, result.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
				result = eval(f'{{{result[0]}}}')
				a = result["Tasks"]
				break
			except Exception as e:
				print(f"task decompose fails: {e}")
				if ind > 10:
					return -1
				ind += 1
				continue
		return a

	def task_topology(self, question, task_ls):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"Given a complex user's question, I have decompose this question into some simple subtasks"
			"I think there exists a logical connections and order amontg the tasks. "
			"Thus you need to help me output this logical connections and order.\n"
			"You must ONLY output in a parsible JSON format with the following format:\n"
			"'''\n"
			"[{{\"task\": task, \"id\", task_id, \"dep\": [dependency_task_id1, dependency_task_id2, ...]}}]\n"
			"'''\n"
			"The \"dep\" field denotes the id of the previous task which generates a new resource upon which the current task depends. If there are no dependencies, set \"dep\" to -1.\n\n"
			"This is user's question: {question}\n"
			"These are subtasks of this question:\n"
			"{task_ls}\n"
			"Output:\n\n"
		)
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		ind = 0
		while True:
			try:
				result = chain.run(question=question, task_ls=task_ls)
				# print(result)
				# exit()
				result = result.replace("```", "").strip().split('\n\n')[-1]
				result = eval(result)
				for i in range(len(result)):
					if isinstance(result[i]['dep'], str):
						temp = []
						for ele in result[i]['dep'].split(','):
							temp.append(int(ele))
						result[i]['dep'] = temp
					elif isinstance(result[i]['dep'], int):
						result[i]['dep'] = [result[i]['dep']]
					elif isinstance(result[i]['dep'], list):
						temp = []
						for ele in result[i]['dep']:
							temp.append(int(ele))
						result[i]['dep'] = temp
					elif result[i]['dep'] == -1:
						result[i]['dep'] = [-1]
				a = result[i]['dep'][0]
				return result
			except Exception as e:
				print(f"task topology fails: {e}")
				if ind > 10:
					return -1
				ind += 1
				continue
		return result

	def choose_tool(self, question, Tool_dic, tool_used):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"This is the user's question: {question}\n"
			"These are the tools you can select to solve the question:\n"
			"Tool List:\n"
			"{Too_list}\n\n"
			"Please note that: \n"
			"1. You should only chooce one tool the Tool List to solve this question.\n"
			"2. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:\n"
			"'''\n"
			"Example 1: {{\"ID\": 1}}\n"
			"Example 2: {{\"ID\": 2}}\n"
			"Example 3: {{\"ID\": 10}}\n"
			"'''\n"
			"Output:\n\n"
		)
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		ind = 0
		Tool_list = []
		for ele in Tool_dic:
			for key in ele.keys():
				if str(key) not in tool_used:
					Tool_list.append(f'''ID: {key}\n{ele[key]}''')

		while True:
			try:
				result = chain.run(question=question,
								   Too_list=Tool_dic)
				pattern = r'{(.*)}'
				clean_answer = re.findall(pattern, result.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
				clean_answer = eval(f'{{{clean_answer[0]}}}')
				# print(clean_answer)
				# exit()
				break
			except Exception as e:
				print(f"choose tool fails: {e}")
				print(result)
				if ind > 10:
					return -1
				ind += 1
				continue
		return clean_answer

	def choose_parameter(self, API_instruction, api, api_dic, question):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"This is an API tool documentation. Given a user's question, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
			"This is API tool documentation: {api_dic}\n"
			"Please note that: \n"
			"1. The Example in the API tool documentation can help you better understand the use of the API.\n"
			"2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{\"Parameters\":{{}}}}\n"
			"3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
			"4. If you need to use this API multiple times, please set \"Parameters\" to a list.\n"
			"5. You must ONLY output in a parsible JSON format. Two examples output looks like:\n"
			# "'''\n"
			"Example 1: {{\"Parameters\":{{\"input\": [1,2,3]}}}}\n"
			"Example 2: {{\"Parameters\":[{{\"input\": [1,2,3]}}, {{\"input\": [2,3,4]}}]}}\n"
			# "'''\n"
			"This is user's question: {question}\n"
			"Output:\n\n"
		)
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		ind = 0
		while True:
			try:
				result = chain.run(api_dic=api_dic,
								   question=question, )
				pattern = r'{(.*)}'
				clean_answer = re.findall(pattern, result.replace(": true", ": True").replace(":true", ": True")\
														.replace(":false", ": False").replace(": false", ": False")\
														.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
				clean_answer = eval(f'{{{clean_answer[0]}}}')
				# print(clean_answer)
				# exit()
				a = clean_answer["Parameters"]
				return a
			except Exception as e:
				print(f"Choose Parameter fails: {e}")
				if ind > 10:
					return -1
				ind += 1
				continue
		return a

	def choose_parameter_depend(self, API_instruction, api, api_dic, question, previous_log):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"Given a user's question and a API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
			"Please note that: \n"
			"1. The Example in the API tool documentation can help you better understand the use of the API.\n"
			"2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{\"Parameters\":{{}}}}\n"
			"3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
			"4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers for your reference.\n"
			"5. If you need to use this API multiple times,, please set \"Parameters\" to a list.\n"
			"6. You must ONLY output in a parsible JSON format. Two examples output looks like:\n"
			# "'''\n"
			"Example 1: {{\"Parameters\":{{\"input\": [1,2,3]}}}}\n"
			"Example 2: {{\"Parameters\":[{{\"input\": [1,2,3]}}, {{\"input\": [2,3,4]}}]}}\n"
			# "'''\n"
			"There are logs of previous questions and answers: \n {previous_log}\n"
			"This is the current user's question: {question}\n"
			"This is API tool documentation: {api_dic}\n"
			"Output:\n\n"
		)
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		ind = 0
		while True:
			try:
				result = chain.run(api_dic=api_dic,
								   question=question,
								   previous_log=previous_log)
				pattern = r'{(.*)}'
				clean_answer = re.findall(pattern, result.replace(": true", ": True").replace(":true", ": True")\
														.replace(":false", ": False").replace(": false", ": False")\
														.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
				clean_answer = eval(f'{{{clean_answer[0]}}}')
				a = clean_answer["Parameters"]

				return a
			except Exception as e:
				print(f"choose parameter depend fails: {e}")
				if ind > 10:
					return -1
				ind += 1
				continue
		return a

	def answer_generation_depend(self, question, API_instruction, call_result, previous_log):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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", model_kwargs={"max_new_tokens":512})
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"You should answer the question based on the response output by the API tool."
			"Please note that:\n"
			"1. Try to organize the response into a natural language answer.\n"
			"2. We will not show the API response to the user, "
			"thus you need to make full use of the response and give the information "
			"in the response that can satisfy the user's question in as much detail as possible.\n"
			"3. If the API tool does not provide useful information in the response, "
			"please answer with your knowledge.\n"
			"4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers.\n"
			"There are logs of previous questions and answers: \n {previous_log}\n"
			"This is the user's question: {question}\n"
			"This is the response output by the API tool: \n{call_result}\n"
			"We will not show the API response to the user, "
			"thus you need to make full use of the response and give the information "
			"in the response that can satisfy the user's question in as much detail as possible.\n"
			"Output:\n\n"
		)
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		ind = 0
		while True:
			try:
				result = chain.run(question=question,
								   API_instruction=API_instruction,
								   call_result=call_result,
								   previous_log=previous_log)
				clean_answer = result.replace("```", "").strip().split('\n\n')[-1]
				break
			except Exception as e:
				print(f"answer generation depend fails: {e}")
				if ind > 2:
					return -1
				ind += 1
				continue
		return clean_answer

	def answer_summarize(self, question, answer_task):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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", model_kwargs={"max_new_tokens":512})
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"We break down a complex user's problems into simple subtasks and provide answers to each simple subtask. "
			"You need to organize these answers to each subtask and form a self-consistent final answer to the user's question\n"
			"This is the user's question: {question}\n"
			"These are subtasks and their answers: {answer_task}\n"
			"Final answer:\n\n"
		)
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		result = chain.run(question=question, answer_task=answer_task)
		clean_answer = result.replace("```", "").strip().split('\n\n')[-1]
		return clean_answer

	def answer_check(self, question, answer):
		chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
		template = "You are a helpful assistant."
		system_message_prompt = SystemMessagePromptTemplate.from_template(template)
		human_message_prompt = HumanMessagePromptTemplate.from_template(
			"Please check whether the response can reasonably and accurately answer the question."
			"If can, please output 'YES'; If not, please output 'NO'\n"
			"You need to give reasons first and then decide whether the response can reasonably and accurately answer the question. You must only output in a parsible JSON format. Two example outputs look like:\n"
			"Example 1: {{\"Reason\": \"The reason why you think the response can reasonably and accurately answer the question\", \"Choice\": \"Yes\"}}\n"
			"Example 2: {{\"Reason\": \"The reason why you think the response cannot reasonably and accurately answer the question\", \"Choice\": \"No\"}}\n"
			"This is the user's question: {question}\n"
			"This is the response: {answer}\n"
			"Output:\n\n"
		)
		chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
		chain = LLMChain(llm=chat, prompt=chat_prompt)
		result = chain.run(question=question, answer=answer)
		pattern = r'{(.*)}'
		result = re.findall(pattern, result.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
		result = eval(f'{{{result[0]}}}')
		print(result)
		if 'yes'.lower() in result["Choice"].lower():
			return 1
		else:
			return -1

	def retrieval(self, question, Tool_dic, dataset, tool_used, state, previous_log=None):
		tool_id = self.choose_tool(question, Tool_dic, tool_used)
		print(tool_id)
		if tool_id == -1:
			return tool_id, "", "", "", ""
		tool_instruction = dataset[str(tool_id["ID"])]
		print(tool_instruction)
		API_instruction = tool_instruction["API_description"]
		API_tool = tool_instruction["standardized_name"]

		api_selection = [API_tool]
		api_result = []
		for api in api_selection:
			if previous_log is None:
				parameter = self.choose_parameter(API_instruction, api,
										 tool_instruction["Usage"], question)
			else:
				parameter = self.choose_parameter_depend(API_instruction, api,
												tool_instruction["Usage"],
												question, previous_log)
			if parameter == -1:
				continue
			api_result.append({"api_name": api, "parameters": parameter})
		if len(api_result) == 0:
			call_result = ""
			return tool_id, api_result, call_result, tool_instruction, API_instruction

		print(api_result)
		call_results = []
		for api in api_result:
			if isinstance(api["parameters"], dict):
				parameters = {}
				for key in api["parameters"]:
					value = api["parameters"][key]
					key = change_name(key)
					parameters[key] = value
				call_result = self.Call_function(API_tool, parameters, state)
				if call_result == -1:
					continue
				call_results.append(str(call_result))
			elif isinstance(api["parameters"], list):
				parameters = {}
				for para_ls in api["parameters"]:
					for key in para_ls:
						value = para_ls[key]
						key = change_name(key)
						parameters[key] = value
				call_result = self.Call_function(API_tool, parameters, state)
				if call_result == -1:
					continue
				call_results.append(str(call_result))
		call_result = '\n\n'.join(call_results)
		print(call_result)
		return tool_id, api_result, call_result, tool_instruction, API_instruction

	def task_execution(self, state):
		tool_used = []
		answer_bad = []
		answer_good = []
		previous_log = None

		question = state.messages[-1]["User"]
		print(question)
		temp = self.task_decompose(question=question, Tool_dic=state.tool_dic)
		print(temp)
		task_ls = []
		for t in range(len(temp)):
			task_ls.append({"task": temp[t], "id": t + 1})
		# print(task_ls)
		# task_ls = self.task_topology(question, task_ls)
		print(task_ls)
		# if task_ls==-1:
		# 	return ("Cannot answer the question, please try it again!")
		task_depend = {'Original Question': question}
		for task_dic in task_ls:
			task_depend[task_dic['id']] = {'task': task_dic['task'], 'answer': ''}
		print(task_depend)
		answer_task = []
		for task_dic in task_ls:
			task = task_dic['task']

			tool_id, api_result, call_result, tool_instruction, API_instruction = self.retrieval(task, state.tool_dic,
																							state.functions_data,
																							tool_used, state,
																							previous_log=previous_log)
			if len(str(call_result)) > 5000:
				call_result = str(call_result)[:5000]
			answer = self.answer_generation_depend(task, API_instruction, call_result, previous_log)

			check_index = 1
			if str(call_result).strip() == '-1' or str(call_result).strip() == '':
				check_index = -1
			if check_index == 1:
				answer_task.append({'task': task, 'answer': answer})
				# tool_instruction_ls.append(tool_instruction)
				# api_result_ls.append(api_result)
				# call_result_ls.append(call_result)
				tool_used.append(str(tool_id["ID"]))
			else:
				answer_bad.append({'task': task, 'answer': answer})

			task_depend[task_dic['id']]['answer'] = answer
			previous_log = task_depend

		final_answer = self.answer_summarize(question, answer_task)
		check_index = self.answer_check(question, final_answer)
		print(final_answer)
		return final_answer

def bot_execute(state, model_selector):
	# exit()
	agent = Agents(model_selector)
	# message = {"User": "identify the person in this image", "Assistant": ""}
	# image_process_mode = "Pad"
	# path_image = "./history/images/image.jpeg"
	# state.messages.append(message)
	# state.image_process_mode.append(image_process_mode)
	# state.images.append([path_image])
	answer = agent.task_execution(state)
	#--------------translate en2vi------------------
	from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
	def translate_en2vi(en_texts: str, tokenizer_en2vi: object) -> str:
	    input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(device_en2vi)
	    output_ids = model_en2vi.generate(
	        **input_ids,
	        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
	        num_return_sequences=1,
	        num_beams=5,
	        early_stopping=True
	    )
	    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
	    return vi_texts
	tokenizer_en2vi = AutoTokenizer.from_pretrained("./weights/vinai-translate-en2vi-v2", src_lang="en_XX")
	model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("./weights/vinai-translate-en2vi-v2")
	device_en2vi = torch.device("cuda")
	model_en2vi.to(device_en2vi)
	answer_vi = translate_en2vi(answer, tokenizer_en2vi)[0]
	print("------answer_vi: ", answer_vi)
	#///////////////////////////////////////////////
	state.messages[-1][state.roles[1]] = answer
	message = " "
	state.chat.append((None, message))
	results_split = answer_vi.split(" ")
	for re in results_split:
		message += re + " "
		message_show = message + "▌"
		state.chat[-1] = [None, message_show]
		yield (state, state.chat) + (disable_btn,)*6
	for image_path in agent.result_image_path:
		state.chat.append((None, (image_path,)))
		yield (state, state.chat) + (enable_btn,)*6
	return (state, state.chat) + (enable_btn,)*6

def bot_load_init(conversation_id):
	path_conver = os.path.abspath(os.path.join(PATH_CONVER, f"{conversation_id}.json"))
	if not os.path.exists(path_conver) or conversation_id=="":
		if conversation_id=="":
			conversation_id = "conver_default"
		path_conver = os.path.abspath(os.path.join(PATH_CONVER, f"{conversation_id}.json"))
		conversation = Conversation(_id = conversation_id, \
									roles = ["User", "Assistant"], \
									chat = [], \
									messages = [], \
									images = [], \
									voices = [], \
									image_process_mode = [], \
									tool_dic = [], \
									functions_data = {})
		conversation.save_conversation(path_conver) # delete_after
		# Tool_dic = read_jsonline('src/tool_instruction/tool_dic.jsonl')
		dataset = read_json('src/tool_instruction/functions_data.json')
		Tool_dic = []
		for k, v in dataset.items():
			Tool_dic.append({"ID": k, "description": v["API_description"]})
		conversation.tool_dic = Tool_dic
		conversation.functions_data = dataset
	else:
		kwargs_conversation = json.load(open(path_conver))
		# print("-------kwargs_conversation: ", kwargs_conversation)   
		conversation = Conversation(**kwargs_conversation)
		
	path_image_conver = os.path.abspath(os.path.join(PATH_IMAGE, f"{conversation_id}"))
	check_folder_exist(path_image_conver=path_image_conver)
	return conversation

def bot_delete_conver(conversation_id):
	path_conver = os.path.abspath(os.path.join(PATH_CONVER, f"{conversation_id}.json"))
	conversation = Conversation(_id = "conver_default", \
								roles = ["User", "Assistant"], \
								chat = [], \
								messages = [], \
								images = [], \
								voices = [], \
								image_process_mode = [], \
								tool_dic = [], \
								functions_data = {})
	conversation.save_conversation(os.path.abspath(os.path.join(PATH_CONVER, "conver_default.json"))) # delete_after 
	# Tool_dic = read_jsonline('src/tool_instruction/tool_dic.jsonl')
	dataset = read_json('src/tool_instruction/functions_data.json')
	Tool_dic = []
	for k, v in dataset.items():
		Tool_dic.append({"ID": k, "description": v["API_description"]})
	conversation.tool_dic = Tool_dic
	conversation.functions_data = dataset
	path_image_conver = os.path.abspath(os.path.join(PATH_IMAGE, f"{conversation_id}"))
	delete_folder_exist(path_conver=path_conver, path_image_conver=path_image_conver)
	return conversation

def add_text(state, text, image_dict, image_process_mode, with_debug_parameter_from_state=False):
	print(text)
	#----------------translate vi2en-------------------------
	from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
	def translate_vi2en(vi_texts: str, tokenizer_vi2en: object) -> str:
		input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
		output_ids = model_vi2en.generate(
			**input_ids,
			decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
			num_return_sequences=1,
			num_beams=5,
			early_stopping=True
		)
		en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
		return en_texts
	tokenizer_vi2en = AutoTokenizer.from_pretrained("./weights/vinai-translate-vi2en-v2", src_lang="vi_VN")
	model_vi2en =  AutoModelForSeq2SeqLM.from_pretrained("./weights/vinai-translate-vi2en-v2").to(torch.device("cuda"))
	device_vi2en = torch.device("cuda")
	model_vi2en.to(device_vi2en)
	vi_text = text
	text = translate_vi2en(text, tokenizer_vi2en)[0]
	print("------en_text: ", text)
	#////////////////////////////////////////////////////////
	# print(image_dict)
	# print(image_process_mode)
	if image_dict is not None:
		pil_img = Image.open(BytesIO(image_dict['image']))
		path_img = os.path.abspath(os.path.join(PATH_IMAGE, f"{state._id}/{len(state.images)}.jpg"))
		pil_img.save(path_img)
		state.images.append([path_img])
	else:
		state.images.append([None])
	state.image_process_mode.append(image_process_mode)
	message = {state.roles[0]: str(text), state.roles[1]: ""}
	state.messages.append(message)
	# state.chat.append((state.messages[-1][state.roles[0]], None))
	state.chat.append((vi_text, None))
	for img_path in state.images[-1]:
		if img_path is not None:
			state.chat.append(((img_path,), None))
	# state.save_conversation(os.path.abspath(os.path.join(PATH_CONVER, "conver_default.json")))
	return (state, state.chat, "", None) + (disable_btn,) * 6

def add_voice(state, record_dict, image_dict, image_process_mode):
	yield (state, ) + (disable_btn, )
	# exit()
	sampling_rate = record_dict["sampling_rate"]
	sample = np.frombuffer(record_dict["sample"], dtype=np.float32)
	print(sample)
	#----------------------------------
	from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
	def translate_vi2en(vi_texts: str, tokenizer_vi2en: object) -> str:
		input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
		output_ids = model_vi2en.generate(
			**input_ids,
			decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
			num_return_sequences=1,
			num_beams=5,
			early_stopping=True
		)
		en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
		return en_texts

	processor = WhisperProcessor.from_pretrained("weights/PhoWhisper-small")
	model = WhisperForConditionalGeneration.from_pretrained("weights/PhoWhisper-small")
	inputs = processor(sample/sampling_rate, sampling_rate=sampling_rate, return_tensors="pt").input_features
	predicted_ids = model.generate(inputs, language="vi")
	transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
	print(transcription[0])

	# tokenizer_vi2en = AutoTokenizer.from_pretrained("./weights/vinai-translate-vi2en-v2", src_lang="vi_VN")
	# model_vi2en =  AutoModelForSeq2SeqLM.from_pretrained("./weights/vinai-translate-vi2en-v2").to(torch.device("cuda"))
	# device_vi2en = torch.device("cuda")
	# model_vi2en.to(device_vi2en)
	# text = translate_vi2en(input_text, tokenizer_vi2en)
	# print(text)
	# yield add_text(state, text, image_dict, image_process_mode) + (disable_btn,)*7

	yield (state, ) + (enable_btn,)
	return

def add_topic(topic_box):
	yield disable_btn
	# print("-------------topic_box: ", topic_box)
	path_file = topic_box[0].name
	# print(path_file)
	df = pd.read_csv(path_file)
	# print(df)
	datab = BytesIO(df.to_csv(index=False).encode('utf-8'))
	# print(datab)
	files = [("files", datab)]
	api_name = "retrieval_topic"
	worker_topic_addr = get_worker_addr(controller_url, api_name)
	print("----worker_topic_addr: ", worker_topic_addr)

	topic_response = requests.post(
			url=worker_topic_addr + "/worker_embed_topic",
			files=files,
		).json()
	print(topic_response)
	print("-----Success!")
	print(topic_response['error_code']==0)
	if topic_response['error_code']==0:
		yield enable_btn
	return

def add_doc(pdf_box):
	yield disable_btn
	# print("-------------pdf_box: ", pdf_box)
	path_file = pdf_box[0].name
	# print(path_file)
	# doc = PyPDF2.PdfReader(path_file)
	# print(doc)
	datab = open(path_file,mode='rb')
	# datab = BytesIO(doc.encode('utf-8'))
	# print(datab)
	files = [("files", datab)]
	api_name = "retrieval_topic"
	worker_topic_addr = get_worker_addr(controller_url, api_name)
	print("----worker_topic_addr: ", worker_topic_addr)
	params = {"window_size": 128, "step_size": 50}
	doc_response = requests.post(
			url=worker_topic_addr + "/woker_embed_doc",
			params=params,
			files=files,
		).json()
	print(doc_response)
	print("-----Success!")
	if doc_response['error_code']==0:
		yield enable_btn
	return
if __name__=="__main__":
	model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
	param_conver = {"_id": "conver1", \
					"roles": ["User", "Assistant"], \
					"chat": [], \
					"messages": [], \
					"images": [], \
					"voices": [], \
					"image_process_mode": [], \
					"tool_dic": [], \
					"functions_data": {}}
	default_conversation = Conversation(**param_conver)
	Tool_dic = read_jsonline('src/tool_instruction/tool_dic.jsonl')
	dataset = read_json('src/tool_instruction/functions_data.json')
	default_conversation.tool_dic = Tool_dic
	default_conversation.functions_data = dataset
	bot_execute(default_conversation, model_path)
