# from base.libs import *
import gradio as gr 
from io import BytesIO
import numpy as np
from base.constants import *
from llava_module.constants import *

# from llava_module.conversation2 import (default_conversation)
# from llava_module.agents_worker import bot_execute, bot_load_init, add_topic, \
# 										add_doc, add_text, add_voice, PATH_CONVER, bot_delete_conver
from llava_module.agents_worker2 import bot_execute, bot_load_init, \
										add_text, add_voice, bot_delete_conver
from scipy.io import wavfile
from scipy import interpolate

logger_app = logger.bind(name="logger_app")
logger_app.add(os.path.join(PATH_DEFAULT.LOGDIR, f"app.{datetime.date.today()}.log"), mode='w')

def get_list_func():
	spec = importlib.util.spec_from_file_location('tools', Configuration.path_tool_function)
	app_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(app_module)
	list_func = np.array(inspect.getmembers(app_module, inspect.isfunction))[:,0].tolist()
	list_func.append(None)
	return list_func
 
class ImageMask(gr.components.Image):
	is_template = True
	def __init__(self, **kwargs):
		super().__init__(sources=["upload"], type='pil', interactive=True, **kwargs)

	def preprocess(self, x):
		x = super().preprocess(x)
		res = None
		if x is not None:
			res = {}
			buffered = BytesIO()
			x.save(buffered, format='PNG')
			res["image"] = buffered.getvalue()
		return res

class VoiceBox(gr.components.Audio):
	is_template = True
	def __init__(self, **kwargs):
		super().__init__(sources=["upload", "microphone"],
						type="filepath",
						interactive=True,
						**kwargs)
	def preprocess(self, x):
		NEW_SAMPLERATE = 16000
		x = super().preprocess(x)
		old_samplerate, old_audio = wavfile.read(x)
		x = (old_samplerate, old_audio)
		if old_samplerate != NEW_SAMPLERATE:
			duration = old_audio.shape[0] / old_samplerate
			time_old  = np.linspace(0, duration, old_audio.shape[0])
			time_new  = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))
			interpolator = interpolate.interp1d(time_old, old_audio.T)
			new_audio = interpolator(time_new).T
			x = (NEW_SAMPLERATE, new_audio)

		res = None
		if x is not None:
			res = {}
			res["sampling_rate"] = x[0]
			res["sample"] = np.array(x[1], dtype=np.float32).tobytes()
		# res = x[0]
		return res

def create_conver(conversation_id, request: gr.Request):
	logger_app.info("----create_conver: {}", conversation_id)
	if not conversation_id:
		conversation_id = ""
	state = bot_load_init(conversation_id)
	convers = os.listdir(PATH_DEFAULT.PATH_CONVER)
	convers = [os.path.splitext(f)[0] for f in convers if f.endswith('.json')]
	return (state,"",gr.Dropdown(visible=True, choices=convers))

def delete_conver(conversation_id, request: gr.Request):
	logger_app.info("----delete_conver: {}", conversation_id)
	if not conversation_id:
		conversation_id = ""
	state = bot_delete_conver(conversation_id)
	convers = os.listdir(PATH_DEFAULT.PATH_CONVER)
	convers = [os.path.splitext(f)[0] for f in convers if f.endswith('.json')]
	return (state,gr.Dropdown(visible=True, choices=convers))

def load_demo(conversation_id, request: gr.Request):
	logger_app.info("----load_conver_init: {}", conversation_id)
	if not conversation_id:
		conversation_id = ""
	state = bot_load_init(conversation_id)
	convers = os.listdir(PATH_DEFAULT.PATH_CONVER)
	convers = [os.path.splitext(f)[0] for f in convers if f.endswith('.json')]
	api_get_collection = get_worker_addr(controller_url, "retrieval_docs") 
	logger_app.info("----api_get_collection: {}", api_get_collection)
	if api_get_collection == -1:
		collection_list = []
	else:
		api_get_collection += "/worker_get_collection"
		collection_list = requests.request("GET", url=api_get_collection).json()["list_collection"]
	# collection_list.insert(0, "")
	tool_data = read_json(Configuration.path_tool_data)
	instruction_names = list(tool_data.keys())
	instruction_names.append(None)
	return (state,
			gr.Dropdown(visible=True, choices=convers),
			gr.Dropdown(visible=True),
			gr.Dropdown(choices=collection_list, value=[]),
			gr.Chatbot(state.chat, visible=True),
			gr.Textbox(visible=True),
			gr.Button(visible=True),
			gr.Row(visible=True),
			gr.Accordion(visible=True),
   			gr.Dropdown(choices=instruction_names, value=None))

def get_model_list():
	ret = requests.post(controller_url + "/refresh_all_workers")
	assert ret.status_code == 200
	ret = requests.post(controller_url + "/list_models")
	models = ret.json()["models"]
	models.sort(key=lambda x: priority.get(x, x))
	logger_app.info(f"Models: {models}")
	return models

def load_conversation(conversation_id, request: gr.Request):
	logger_app.info("----load_conversation: {}", conversation_id)
	if len(conversation_id)==0:
		conversation_id = ""
	state = bot_load_init(conversation_id)
	# print(state.chat)
	return (state, 
			gr.Chatbot(state.chat, visible=True))

def change_type_tool(tool_type):
	logger_app.info("----tool_type: {}", tool_type)
	if tool_type==type_tools[0]:
		api_name = gr.Textbox(label="API name", placeholder="Enter a API name", show_label=True, container=True, interactive=True,)
		func_script = gr.Code(visible=False)
		func_select = gr.Dropdown(visible=False,)
		func_submit_btn = gr.Button(visible=False,)
		func_delete_btn = gr.Button(visible=False,)
	elif tool_type==type_tools[1]:
		#------------get list function----------
		list_func = get_list_func()
		#////////////////////////////////////////
		api_name = gr.Textbox(label="Function name", placeholder="Enter a function name", show_label=True, container=True, interactive=True,)
		func_script = gr.Code(label="Function script", show_label=True, visible=True, language="python", container=True, interactive=True,)
		func_select = gr.Dropdown(label="Function selection", show_label=True, choices=list_func, value=None, interactive=True, visible=True,)
		func_submit_btn = gr.Button(value="Submit function", interactive=True, scale=1, visible=True,)
		func_delete_btn = gr.Button(value="Delete function", interactive=True, scale=1, visible=False,)
	return (api_name, func_script, func_select, func_submit_btn, func_delete_btn)

def change_function(func_select):
	logger_app.info("----func_select: {}", func_select)
	if not func_select:
		api_name = gr.Textbox(label="Function name", show_label=True, placeholder="Enter a API name", container=True, interactive=True, value=None)
		func_submit_btn = gr.Button(value="Submit function", interactive=True, scale=1, visible=True,)
		func_delete_btn = gr.Button(value="Delete function", interactive=True, scale=1,visible=False,)
		func_script = gr.Code(label="Function script", show_label=True, visible=True, language="python", container=True, interactive=True, value=None)
	else:
		#------------get code function----------
		spec = importlib.util.spec_from_file_location('tools', Configuration.path_tool_function)
		app_module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(app_module)
		func = getattr(app_module, func_select)
		func_code = inspect.getsource(func.__code__)
		#////////////////////////////////////////
		pattern = r'def (.*)\(.*\):'
		func_name = regex.findall(pattern, func_code, regex.DOTALL)[0]
		api_name = gr.Textbox(label="Function name", show_label=True, placeholder="Enter a API name", container=True, interactive=True, value=func_name)
		func_script = gr.Code(label="Function script", show_label=True, visible=True, language="python", container=True, interactive=True, value=func_code)
		func_submit_btn = gr.Button(value="Update function", interactive=True, scale=1, visible=True,)
		func_delete_btn = gr.Button(value="Delete function", interactive=True, scale=1, visible=True,)
	return (api_name, func_submit_btn, func_delete_btn, func_script)

def change_api_name(tool_type, api_name, func_select):
	if tool_type==type_tools[0]:
		func_script = gr.Code(visible=False)
	elif tool_type==type_tools[1]:
		func_code = f"def {api_name}(**kwargs):\n\t\n\treturn"
		func_script = gr.Code(label="Function script", show_label=True, visible=True, language="python", container=True, interactive=True, value=func_code)
		if func_select:
			func_script = gr.Code(label="Function script", show_label=True, visible=True, language="python", container=True, interactive=True)
	return (func_script)

def change_inst(inst_select):
	logger_app.info("----select instruction: {}", inst_select)
	if not inst_select:
		tool_name = gr.Textbox(label="Tool name", show_label=True, placeholder="Enter a tool name", container=True, value=None)
		api_name = gr.Textbox(label="API name", show_label=True, placeholder="Enter a API name", container=True, interactive=True, value=None)
		api_desc = gr.Textbox(label="API description", show_label=True, placeholder="Enter a brief description of API", container=True, value=None)
		tool_scenario = gr.Textbox(label="Scenario", show_label=True, placeholder="Enter example about scenario the tool will use", container=True, value=None)
		tool_payload = gr.Textbox(label="Payload", show_label=True, value=str({'key1':['value1']}), container=True)
		tool_file_type = gr.Dropdown(label="File type", show_label=True, choices=file_types, value=[], interactive=True, multiselect=True, container=True)
		inst_submit_btn = gr.Button(value="Submit instruction", interactive=True, scale=1, visible=True,)
		inst_delete_btn = gr.Button(value="Delete instruction", interactive=True, scale=1, visible=False,)
	else:
		tool_data = read_json(Configuration.path_tool_data)
		tool_infor = tool_data[inst_select]
		tool_name = gr.Textbox(label="Tool name", show_label=True, placeholder="Enter a tool name", container=True, value=tool_infor["ID"])
		api_name = gr.Textbox(label="API name", show_label=True, placeholder="Enter a API name", container=True, interactive=True, value=tool_infor["API_name"])
		api_desc = gr.Textbox(label="API description", show_label=True, placeholder="Enter a brief description of API", container=True, value=tool_infor["API_description"])
		tool_scenario = gr.Textbox(label="Scenario", show_label=True, placeholder="Enter example about scenario the tool will use", container=True, value=list(tool_infor["Usage"].values())[0]["Scenario"])
		tool_payload = gr.Textbox(label="Payload", show_label=True, value=str(list(tool_infor["Usage"].values())[0]["Parameters"]["payload"]), container=True)
		tool_file_type = gr.Dropdown(label="File type", show_label=True, choices=file_types, value=list(tool_infor["Usage"].values())[0]["Parameters"]["file"], interactive=True, multiselect=True, container=True)
		inst_submit_btn = gr.Button(value="Update instruction", interactive=True, scale=1, visible=True,)
		inst_delete_btn = gr.Button(value="Delete instruction", interactive=True, scale=1, visible=True,)
	return (tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type, inst_submit_btn, inst_delete_btn)
		
def delete_inst(inst_select):
	logger_app.info("----delete instruction: {}", inst_select)
	if inst_select:
		tool_data = read_json(Configuration.path_tool_data)
		del tool_data[inst_select]
		write_json(Configuration.path_tool_data, tool_data)
		list_tool = list(tool_data.keys())
		inst_selection = gr.Dropdown(label="Instruction selection", show_label=True, choices=list_tool, value=inst_select, interactive=True, container=True)
		tool_name = gr.Textbox(label="Tool name", show_label=True, placeholder="Enter a tool name", container=True, value=None)
		api_name = gr.Textbox(label="API name", show_label=True, placeholder="Enter a API name", container=True, interactive=True, value=None)
		api_desc = gr.Textbox(label="API description", show_label=True, placeholder="Enter a brief description of API", container=True, value=None)
		tool_scenario = gr.Textbox(label="Scenario", show_label=True, placeholder="Enter example about scenario the tool will use", container=True, value=None)
		tool_payload = gr.Textbox(label="Payload", show_label=True, value=str({'key1':['value1']}), container=True)
		tool_file_type = gr.Dropdown(label="File type", show_label=True, choices=file_types, value=[], interactive=True, multiselect=True, container=True)
	return (inst_selection, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type,)

def submit_instruction(inst_selection, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type,):
	tool_data = read_json(Configuration.path_tool_data)
	list_tool = list(tool_data.keys())
	if (tool_name in list_tool) and (not inst_selection):
		gr.Warning("Cannot submit instruction because tool name must be unique!")
		inst_selection = gr.Dropdown(label="Instruction selection", show_label=True, choices=list_tool, value=inst_selection, interactive=True, container=True)
		tool_name = gr.Textbox(label="Tool name", show_label=True, placeholder="Enter a tool name", container=True,)
		api_name = gr.Textbox(label="API name", show_label=True, placeholder="Enter a API name", container=True, interactive=True)
		api_desc = gr.Textbox(label="API description", show_label=True, placeholder="Enter a brief description of API", container=True)
		tool_scenario = gr.Textbox(label="Scenario", show_label=True, placeholder="Enter example about scenario the tool will use", container=True)
		tool_payload = gr.Textbox(label="Payload", show_label=True, container=True)
		tool_file_type = gr.Dropdown(label="File type", show_label=True, choices=file_types, value=tool_file_type, multiselect=True, interactive=True, container=True)
	else:
		if not tool_file_type:
			tool_file_type = []
		if not tool_payload:
			tool_payload = []
		else:
			tool_payload = eval(tool_payload)
		tool_infor = {
			"ID":str(tool_name),
			"API_name":str(api_name),
			"API_description":str(api_desc),
			"Usage":{
				"Example":{
					"Scenario":tool_scenario,
					"Parameters":{
						"file":tool_file_type,
						"payload": tool_payload
					}
				}
			}
		}
		tool_data[str(tool_name)] = tool_infor
		write_json(Configuration.path_tool_data, tool_data)
		list_tool = list(tool_data.keys())
		list_tool.append(None)
		inst_selection = gr.Dropdown(label="Instruction selection", show_label=True, choices=list_tool, value=None, interactive=True, container=True)
		tool_name = gr.Textbox(label="Tool name", show_label=True, placeholder="Enter a tool name", container=True, value=None)
		api_name = gr.Textbox(label="API name", show_label=True, placeholder="Enter a API name", container=True, interactive=True, value=None)
		api_desc = gr.Textbox(label="API description", show_label=True, placeholder="Enter a brief description of API", container=True, value=None)
		tool_scenario = gr.Textbox(label="Scenario", show_label=True, placeholder="Enter example about scenario the tool will use", container=True, value=None)
		tool_payload = gr.Textbox(label="Payload", show_label=True, value=str({'key1':['value1']}), container=True)
		tool_file_type = gr.Dropdown(label="File type", show_label=True, choices=file_types, value=[], interactive=True, multiselect=True, container=True)
	return ( inst_selection, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type, )

def submit_function(inst_selection, type_tool, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type, func_select, func_script):
	#------------get list function----------
	list_func = get_list_func()
	#////////////////////////////////////////
	pattern = r'def (.*)\(.*\):'
	func_name = regex.findall(pattern, func_script, regex.DOTALL)[0]
	if func_name != api_name:
		gr.Warning("Function name and API name must be the same!")
	elif (api_name in list_func) and (not func_select):
		gr.Warning("Cannot submit because this function exists!")
	elif not func_select and func_script:
		with open(f"{PATH_DEFAULT.PATH_TOOL_LIB}/{api_name}.py", "w") as f:
			f.write(func_script)
		with open(f"{PATH_DEFAULT.PATH_SERVICES}/tools.py", "a") as f:
			f.write(f"from services.tool_lib.{api_name} import {api_name}\n")
	elif not func_script:
		gr.Warning("Cannot submit because of missing function script!")
	else:
		with open(f"{PATH_DEFAULT.PATH_TOOL_LIB}/{func_select}.py", "w") as f:
			f.write(func_script)
	
	api_name = gr.Textbox(label="Function name", show_label=True, placeholder="Enter a API name", container=True, interactive=True, value=None)
	func_select = gr.Dropdown(label="Function selection", show_label=True, choices=list_func, value=None, interactive=True, visible=True,)
	func_script = gr.Code(label="Function script", show_label=True, visible=True, language="python", container=True, interactive=True, value=None)
	return (
		gr.Dropdown(label="Instruction selection", show_label=True, choices=instruction_names, value=None, interactive=True, container=True),
		gr.Textbox(label="Tool name", show_label=True, placeholder="Enter a tool name", container=True),
  		api_name,
		gr.Textbox(label="API description", show_label=True, placeholder="Enter a brief description of API", container=True), 
	 	gr.Textbox(label="Scenario", show_label=True, placeholder="Enter example about scenario the tool will use", container=True), 
	  	gr.JSON(label="Payload", show_label=True, value={'key1':'value1', 'key2':['value2']}, container=True), 
	   	gr.Dropdown(label="File type", show_label=True, choices=file_types, value=None, interactive=True, container=True), 
		func_select, 
		func_script
	)
 
def delete_function(func_select):
	if func_select:
		os.remove(f"{PATH_DEFAULT.PATH_TOOL_LIB}/{func_select}.py")
	list_func_file = os.listdir(PATH_DEFAULT.PATH_TOOL_LIB)
	with open(f"{PATH_DEFAULT.PATH_SERVICES}/tools.py", "w") as f:
		for func in list_func_file:
			if func.endswith(".py") and not func.endswith("__init__.py"):
				name, ext = os.path.splitext(func)
				f.write(f"from services.tool_lib.{name} import {name}\n")
	list_func = get_list_func()
	func_select = gr.Dropdown(label="Function selection", show_label=True, choices=list_func, value=None, interactive=True, visible=True,)
	func_script = gr.Code(label="Function script", show_label=True, visible=True, language="python", container=True, interactive=True,)
	return (
		func_select, 
		func_script
	)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
	with open(get_conv_log_filename(), "a") as fout:
		data = {
			"tstamp": round(time.time(), 4),
			"type": vote_type,
			"model": model_selector,
			"state": state.dict(),
			"ip": request.client.host,
		}
		fout.write(json.dumps(data) + "\n")

def upvote_last_response(state, model_selector, request: gr.Request):
	logger_app.info(f"upvote. ip: {request.client.host}")
	vote_last_response(state, "upvote", model_selector, request)
	return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
	logger_app.info(f"downvote. ip: {request.client.host}")
	vote_last_response(state, "downvote", model_selector, request)
	return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
	logger_app.info(f"flag. ip: {request.client.host}")
	vote_last_response(state, "flag", model_selector, request)
	return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, with_debug_parameter_from_state, request: gr.Request):
	logger_app.info(f"regenerate. ip: {request.client.host}")
	state.messages[-1][-1] = None
	prev_human_msg = state.messages[-2]
	if type(prev_human_msg[1]) in (tuple, list):
		prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
	state.skip_next = False
	return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None, None) + (disable_btn,) * 5


def clear_history(with_debug_parameter_from_state, request: gr.Request):
	logger_app.info(f"clear_history. ip: {request.client.host}")
	state = default_conversation.copy()
	return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None, None) + (disable_btn,) * 5

def change_debug_state(state, with_debug_parameter_from_state, request: gr.Request):
	# logger.info(f"change_debug_state. ip: {request.client.host}")
	print("with_debug_parameter_from_state: ", with_debug_parameter_from_state)
	with_debug_parameter_from_state = not with_debug_parameter_from_state

	# modify the text on debug_btn
	debug_btn_value = "🈚 Prog (off)" if not with_debug_parameter_from_state else "🈶 Prog (on)"

	debug_btn_update = gr.Button.update(
		value=debug_btn_value,
	)
	state_update = with_debug_parameter_from_state
	return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None) + (debug_btn_update, state_update)

def build_demo():
	textbox = gr.MultimodalTextbox(show_label=False, placeholder="Enter text and press ENTER", container=False, interactive=True, file_types=["image"])
	with gr.Blocks(title="MQ-GPT", theme=gr.themes.Default(), css=block_css) as demo:
		state = gr.State()
		gr.Markdown(title_markdown)
		with gr.Row():
			with gr.Column(scale=3):
				with gr.Row(elem_id="conver_creator_row"):
					conver_creator = gr.Textbox(
						label="Create a conversation",
						show_label=True,
						placeholder="Enter name conversation and press ENTER",
						container=True)

				with gr.Row(elem_id="conver_selector_row"):
					conver_selector = gr.Dropdown(
						label="Select conversation",
						# choices=convers,
						interactive=True,
						container=False)

				with gr.Row(elem_id="conver_delete_row"):
					conver_select_btn = gr.Button(value="Select conversation", interactive=True, scale=1)
					conver_delete_btn = gr.Button(value="Delete conversation", interactive=True, scale=1)

				with gr.Row(elem_id="model_selector_row"):
					model_selector = gr.Dropdown(
						label="Select LLM model",
						choices=models,
						value=models[0] if len(models) > 0 else "",
						interactive=True,
						container=True)
				
				with gr.Accordion("Image", open=False, visible=True) as image_row:
					gr.Markdown(
						"The image is for vision tools.")
					imagebox = ImageMask()

				with gr.Accordion("Audio", open=False, visible=True) as audio_row:
					audiobox = VoiceBox()
					audio_upload_btn = gr.Button(value="Upload audio", interactive=True)

				with gr.Accordion("Knowledge", open=False, visible=True):
					knowledge_selector = gr.Dropdown(
						label="Select knowledge",
						interactive=True,
						container=True,
						multiselect=True, 
						visible=True)
					n_result = gr.Number(value=1)

				with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
					image_process_mode = gr.Radio(
						["Crop", "Resize", "Pad"],
						value="Crop",
						label="Preprocess for non-square image")
					temperature = gr.Slider(
						minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
					top_p = gr.Slider(
						minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
					max_output_tokens = gr.Slider(
						minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)
				
				# # DONE: add accordion Topic_file and PDF_file, each has Submit button
				# with gr.Accordion("Topic file", open=False, visible=True) as topic_row:
				# 	topic_box = gr.Files(file_types=['.csv'])
				# 	with gr.Row(visible=True) as button_row:
				# 		submit_topic_btn = gr.Button(
				# 			value="Submit", visible=True)

				# with gr.Accordion("PDF file", open=False, visible=True) as pdf_row:
				# 	pdf_box = gr.Files(file_types=['.pdf'])
				# 	with gr.Row(visible=True) as button_row:
				# 		submit_pdf_btn = gr.Button(
				# 			value="Submit", visible=True)
					
			with gr.Column(scale=6):
				chatbot = gr.Chatbot(
					elem_id="chatbot", label="MQ-Chatbot", height=550)
				with gr.Row():
					with gr.Column(scale=8):
						textbox.render()
					with gr.Column(scale=1, min_width=60):
						submit_btn = gr.Button(value="Submit", visible=True)
				with gr.Row(visible=True) as button_row:
					upvote_btn = gr.Button(
						value="👍  Upvote", interactive=False)
					downvote_btn = gr.Button(
						value="👎  Downvote", interactive=False)
					flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
					# stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
					regenerate_btn = gr.Button(
						value="🔄  Regenerate", interactive=False)
					clear_btn = gr.Button(
						value="🗑️  Clear history", interactive=False)
					debug_btn = gr.Button(
						value="🈚  Prog (off)", interactive=True)
					# import ipdb; ipdb.set_trace()
				if with_debug_parameter:
					debug_btn.value = "🈶 Prog (on)"
				with_debug_parameter_state = gr.State(
					value=with_debug_parameter,
				)

		with gr.Blocks():
			gr.Markdown("# Instructions")
			with gr.Row(elem_id="type_tool"):
				type_tool = gr.Dropdown(
					label="Select type tool",
					show_label=True,
					choices=type_tools,
					value=type_tools[0],
					interactive=True,
					container=True)
			with gr.Row(elem_id="inst_list"):
				inst_selection = gr.Dropdown(label="Instruction selection", show_label=True, choices=[], value=None, interactive=True, container=True)
			with gr.Row(elem_id="tool_name"):
				tool_name = gr.Textbox(label="Tool name", show_label=True, placeholder="Enter a tool name", container=True)
			with gr.Row(elem_id="api_name"):
				api_name = gr.Textbox(label="API name", show_label=True, placeholder="Enter a API name", container=True, interactive=True,)
			with gr.Row(elem_id="api_description"):
				api_desc = gr.Textbox(label="API description", show_label=True, placeholder="Enter a brief description of API", container=True)
			with gr.Blocks():
				gr.Markdown("Example for tool")
				with gr.Row():
					with gr.Column(scale=3):
						tool_scenario = gr.Textbox(label="Scenario", show_label=True, placeholder="Enter example about scenario the tool will use", container=True)
					with gr.Column(scale=3):
						tool_payload = gr.Textbox(label="Payload", show_label=True, value=str({'key1':['value1']}), container=True)
					with gr.Column(scale=3):
						tool_file_type = gr.Dropdown(label="File type", show_label=True, choices=file_types, value=None, interactive=True, multiselect=True, container=True)
			with gr.Row():
				inst_submit_btn = gr.Button(value="Submit instruction", interactive=True, scale=1, visible=True,)
				inst_delete_btn = gr.Button(value="Delete instruction", interactive=True, scale=1, visible=False,)
			with gr.Blocks():
				gr.Markdown("Function")
				with gr.Row(elem_id="func_script"):
					with gr.Column(scale=3):
						func_select = gr.Dropdown(label="Function selection", show_label=True, choices=[""], value="", interactive=True, visible=False,)
						with gr.Row():
							func_submit_btn = gr.Button(value="Submit function", interactive=True, scale=1, visible=False,)
							func_delete_btn = gr.Button(value="Delete function", interactive=True, scale=1, visible=False,)
					with gr.Column(scale=7):
						func_script = gr.Code(label="Function script", show_label=True, visible=False, language="python", container=True, interactive=True,)

		url_params = gr.JSON(visible=False)

		# # Register listeners
		btn_list = [upvote_btn, downvote_btn,
					flag_btn, regenerate_btn, clear_btn]
		# upvote_btn.click(upvote_last_response,
		#                  [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
		# downvote_btn.click(downvote_last_response,
		#                    [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
		# flag_btn.click(flag_last_response,
		#                [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
		# regenerate_btn.click(regenerate, [state, image_process_mode, with_debug_parameter_state],
		#                      [state, chatbot, textbox, imagebox] + btn_list).then(
		#     http_bot, [state, model_selector, temperature, top_p,
		#                max_output_tokens, with_debug_parameter_state],
		#     [state, chatbot] + btn_list + [debug_btn])
		# clear_btn.click(clear_history, [], [state, chatbot, textbox, imagebox, audiobox] + btn_list)
		inst_selection.change(change_inst, [inst_selection], [tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type, inst_submit_btn, inst_delete_btn])
		inst_submit_btn.click(submit_instruction, [inst_selection, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type,], \
      												[inst_selection, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type,])
		inst_delete_btn.click(delete_inst, [inst_selection], [inst_selection, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type,])
		api_name.change(change_api_name, [type_tool, api_name, func_select], [func_script])
		type_tool.change(change_type_tool, [type_tool], [api_name, func_script, func_select, func_submit_btn, func_delete_btn])
		func_select.change(change_function, [func_select], [api_name, func_submit_btn, func_delete_btn, func_script])
		func_submit_btn.click(submit_function, [inst_selection, type_tool, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type, func_select, func_script],\
	  												[inst_selection, tool_name, api_name, api_desc, tool_scenario, tool_payload, tool_file_type, func_select, func_script])
		func_delete_btn.click(delete_function, [func_select], [func_select, func_script])
  

		textbox.submit(add_text, [state, textbox, imagebox, image_process_mode, knowledge_selector, n_result, conver_selector, with_debug_parameter_state], [state, chatbot, textbox, imagebox] + btn_list + [debug_btn]
						).then(bot_execute, [state, model_selector, conver_selector], [state, chatbot] + btn_list + [debug_btn])

		submit_btn.click(add_text, [state, textbox, imagebox, image_process_mode, knowledge_selector, n_result, conver_selector, with_debug_parameter_state], [state, chatbot, textbox, imagebox] + btn_list + [debug_btn]
						).then(bot_execute, [state, model_selector, conver_selector], [state, chatbot] + btn_list + [debug_btn])
		# debug_btn.click(change_debug_state, [state, with_debug_parameter_state], [
		#                 state, chatbot, textbox, imagebox] + [debug_btn, with_debug_parameter_state])
		# #config submit_PDF_btn.click
		# #config submit_topic_btn.click
		# submit_topic_btn.click(add_topic, [topic_box], [submit_topic_btn])
		# submit_pdf_btn.click(add_doc, [pdf_box], [submit_pdf_btn])

		conver_creator.submit(create_conver, [conver_creator], [state, conver_creator, conver_selector])
		# conver_selector.change(load_conversation, [conver_selector], [state, chatbot])
		conver_select_btn.click(load_conversation, [conver_selector], [state, chatbot])
		conver_delete_btn.click(delete_conver, [conver_selector], [state, conver_selector])

		# audiobox.stop_recording(add_voice, [state, audiobox, imagebox, image_process_mode], [])
		audio_upload_btn.click(add_voice, [state, audiobox, imagebox, image_process_mode, knowledge_selector, n_result, conver_selector], [state, chatbot, audio_upload_btn] + btn_list + [debug_btn])
		
		model_list_mode = "once"
		if model_list_mode == "once":
			demo.load(load_demo, [conver_selector], [state, conver_selector, model_selector, knowledge_selector,
												chatbot, textbox, submit_btn, button_row, parameter_row, inst_selection])
		elif model_list_mode == "reload":
			demo.load(load_demo_refresh_model_list, None, [state, conver_selector, model_selector,
														   chatbot, textbox, submit_btn, button_row, parameter_row])
		else:
			raise ValueError(
				f"Unknown model list mode: {model_list_mode}")
	return demo

if __name__=="__main__":
	#config launch gradio
	host = "0.0.0.0"
	port = 8888
	share = False
	#config queue gradio
	api_open = False
	max_size = 100
	with_debug_parameter = True
	models = ["mixtral-8x7b-32768"]
	type_tools = ["API", "Function"]
	file_types = ["image", "audio", "doc", "video"]
	# tool_data = read_json(Configuration.path_tool_data)
	# instruction_names = list(tool_data.keys())
	# instruction_names.append(None)
	# path_history = os.path.abspath(f"{ROOT}/history")
	# check_folder_exist(path_history=path_history)
	# convers = os.listdir(path_history)
	# convers = [os.path.splitext(f)[0] for f in convers if f.endswith('.json')]
	# print(convers)
	# exit()

	demo = build_demo()
	demo.queue(
		api_open = api_open,
		max_size = max_size
	).launch(
		server_name = host,
		server_port = port,
		share = share,
		favicon_path="./icons/bot.png",
		ssl_keyfile="key.pem",
		ssl_certfile="cert.pem",
		ssl_verify=False
	)


# import importlib.util
# import inspect
# # app_path = "tools.py"
# # spec = importlib.util.spec_from_file_location('tools', app_path)
# # app_module = importlib.util.module_from_spec(spec)
# # spec.loader.exec_module(app_module)
# # func1 = getattr(app_module, "get_time")
# # inspect.getsource(func1.__code__)
# # inspect.getmembers(app_module, inspect.isfunction)