from base.libs import *
from base.constants import *
from llava_module.constants import *

# from llava_module.conversation2 import (default_conversation)
from llava_module.agents_worker import bot_execute, bot_load_init, add_topic, add_doc, add_text

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """
    is_template = True
    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch",
                         type='pil', interactive=True, **kwargs)
        # super().__init__(source="upload", tool="boxes", type='pil', interactive=True, **kwargs)

    def preprocess(self, x):
        # a hack to get the mask
        if isinstance(x, str):
            im = processing_utils.decode_base64_to_image(x)
            w, h = im.size
            # a mask, array, uint8
            mask_np = np.zeros((h, w, 4), dtype=np.uint8)
            # to pil
            mask_pil = Image.fromarray(mask_np, mode='RGBA')
            # to base64
            mask_b64 = processing_utils.encode_pil_to_base64(mask_pil)

            buffered = BytesIO()
            x.save(buffered, format="PNG")
            x = {
                'image': buffered.getvalue(),
                'mask': mask_b64
            }

        res = super().preprocess(x)
        return res

def load_demo(conversation_id, request: gr.Request):
	print(conversation_id)
	dropdown_update = gr.Dropdown.update(visible=True)

	state = bot_load_init(conversation_id)
	return (state,
	        dropdown_update,
	        gr.Chatbot.update(visible=True),
	        gr.Textbox.update(visible=True),
	        gr.Button.update(visible=True),
	        gr.Row.update(visible=True),
	        gr.Accordion.update(visible=True))

def get_model_list():
	ret = requests.post(controller_url + "/refresh_all_workers")
	assert ret.status_code == 200
	ret = requests.post(controller_url + "/list_models")
	models = ret.json()["models"]
	models.sort(key=lambda x: priority.get(x, x))
	logger_app.info(f"Models: {models}")
	return models

def load_conversation(conversation_id, request: gr.Request):
	print(conversation_id)
	state = bot_load_init(conversation_id)
	return (state)

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
	textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
	with gr.Blocks(title="MQ-GPT", theme=gr.themes.Default(), css=block_css) as demo:
		state = gr.State()
		gr.Markdown(title_markdown)
		with gr.Row():
			with gr.Column(scale=3):
				with gr.Row(elem_id="conver_selector_row"):
					conver_selector = gr.Dropdown(
						label="Select conversation",
						choices=convers,
						interactive=True,
						show_label=True,
						container=False)

				with gr.Row(elem_id="model_selector_row"):
					model_selector = gr.Dropdown(
						label="Select LLM model",
						choices=models,
						value=models[0] if len(models) > 0 else "",
						interactive=True,
						show_label=True,
						container=False)
				
				with gr.Accordion("Image", open=False, visible=True) as image_row:
					gr.Markdown(
						"The image is for vision tools.")
					imagebox = ImageMask()

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
				
				# DONE: add accordion Topic_file and PDF_file, each has Submit button
				with gr.Accordion("Topic file", open=False, visible=True) as topic_row:
					topic_box = gr.Files(file_types=['.csv'])
					with gr.Row(visible=True) as button_row:
						submit_topic_btn = gr.Button(
							value="Submit", visible=True)

				with gr.Accordion("PDF file", open=False, visible=True) as pdf_row:
					pdf_box = gr.Files(file_types=['.pdf'])
					with gr.Row(visible=True) as button_row:
						submit_pdf_btn = gr.Button(
							value="Submit", visible=True)
					
			with gr.Column(scale=6):
				chatbot = gr.Chatbot(
					elem_id="chatbot", label="LLaVA-Plus Chatbot", height=550)
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
		# clear_btn.click(clear_history, [with_debug_parameter_state], [
		#                 state, chatbot, textbox, imagebox] + btn_list)

		textbox.submit(add_text, [state, textbox, imagebox, image_process_mode, with_debug_parameter_state], [state, chatbot, textbox, imagebox] + btn_list + [debug_btn]
		               ).then(bot_execute, [state, model_selector, image_process_mode], [state, chatbot] + btn_list + [debug_btn])
		# submit_btn.click(add_text, [state, textbox, imagebox, image_process_mode, with_debug_parameter_state], [state, chatbot, textbox, imagebox] + btn_list + [debug_btn]
		#                  ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens, with_debug_parameter_state],
		#                         [state, chatbot] + btn_list + [debug_btn])
		# debug_btn.click(change_debug_state, [state, with_debug_parameter_state], [
		#                 state, chatbot, textbox, imagebox] + [debug_btn, with_debug_parameter_state])
		# #config submit_PDF_btn.click
		# #config submit_topic_btn.click
		# submit_topic_btn.click(add_topic, [topic_box], [submit_topic_btn])
		# submit_pdf_btn.click(add_doc, [pdf_box], [submit_pdf_btn])

		conver_selector.select(load_conversation, [conver_selector], [state])
		
		model_list_mode = "once"
		if model_list_mode == "once":
			demo.load(load_demo, [conver_selector], [state, model_selector,
												chatbot, textbox, submit_btn, button_row, parameter_row],
					  _js=get_window_url_params)
		elif model_list_mode == "reload":
			demo.load(load_demo_refresh_model_list, None, [state, model_selector,
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
	models = ["mistralai/Mixtral-8x7B-Instruct-v0.1"]
	convers = os.listdir(f"{ROOT}/history/")
	convers = [os.path.splitext(f)[0] for f in convers if f.endswith('.json')]
	# print(convers)
	# exit()

	demo = build_demo()
	demo.queue(
		api_open = api_open,
		max_size = max_size
	).launch(
		server_name = host,
		server_port = port,
		share = share
	)