import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *
from base.constants import *
from llava.constants import *
from llava.serve.utils import annotate_xyxy, show_mask
from llava.conversation import default_conversation, conv_templates, SeparatorStyle
from llava.utils import violates_moderation, moderation_msg

def get_worker_addr(controller_addr, worker_name):
    # get grounding dino addr
    if worker_name.startswith("http"):
        sub_server_addr = worker_name
    else:
        controller_addr = controller_addr
        ret = requests.post(controller_addr + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        # print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": worker_name}
        )
        sub_server_addr = ret.json()["address"]
    # print(f"worker_name: {worker_name}")
    return sub_server_addr

def b64_encode(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str

def get_mask_bbox(mask_img: Image):
    # convert to np array
    mask = np.array(mask_img)[..., 0]

    # check if has masks
    if mask.sum() == 0:
        return None

    # get coords
    coords = np.argwhere(mask > 0)

    # calculate bbox
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # get h and w
    h, w = mask.shape[:2]

    # norm to [0, 1]
    x0, y0, x1, y1 = R(x0 / w), R(y0 / h), R(x1 / w), R(y1 / h)
    return [x0, y0, x1, y1]

def plot_boxes(image: Image, res: dict) -> Image:
    boxes = torch.Tensor(res["boxes"])
    logits = torch.Tensor(res["logits"]) if 'logits' in res else None
    phrases = res["phrases"] if 'phrases' in res else None
    image_source = np.array(image)
    annotated_frame = annotate_xyxy(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return Image.fromarray(annotated_frame)


def plot_masks(image: Image, res: dict) -> Image:
    masks_rle = res["masks_rle"]
    for mask_rle in masks_rle:
        mask = mask_util.decode(mask_rle)
        mask = torch.Tensor(mask)
        image = show_mask(mask, image)
    return image

def plot_points(image: Image, res: dict) -> Image:
    points = torch.Tensor(res["points"])
    point_labels = torch.Tensor(res["point_labels"])

    points = np.array(points)
    point_labels = np.array(point_labels)
    annotated_frame = np.array(image)
    h, w = annotated_frame.shape[:2]
    for i in range(points.shape[1]):
        color = (0, 255, 0) if point_labels[0, i] == 1 else (0, 0, 255)
        annotated_frame = cv2.circle(annotated_frame, (int(
            points[0, i, 0] * w), int(points[0, i, 1] * h)), 5, color, -1)
    return Image.fromarray(annotated_frame)

def http_bot(state, model_selector, temperature, top_p, max_new_tokens, with_debug_parameter_from_state, request: gr.Request):
    logger_app.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (no_change_btn,) * 6
        return

    if len(state.messages) == state.offset + 2:
        # # First round of conversation

        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower() and 'tools' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        print("template_name: ", template_name)

        # # hack:
        # # template_name = "multimodal_tools"
        # # import ipdb; ipdb.set_trace()
        # # image_name = [hashlib.md5(image.tobytes()).hexdigest() for image in state.get_images(return_pil=True)][0]

        new_state = conv_templates[template_name].copy()

        # if len(new_state.roles) == 2:
        #     new_state.roles = tuple(list(new_state.roles) + ["system"])
        # new_state.append_message(new_state.roles[2], f"receive an image with name `{image_name}.jpg`")

        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        
        # for reference image
        new_state.reference_image = getattr(state, 'reference_image', None)
        new_state.reference_mask = getattr(state, 'reference_mask', None)
        
        # update
        state = new_state
        
        print("Messages：", state.messages)
    #----------------------------- Get Topic --------------------------
    topic_type = {
        "image_processing":["sam", "instruct-pix2pix", "openseed", "clip", "stable-diffusion", "blip2+grounding_dino", "QnA"\
                            "QnA_vision", "grounding_dino", "grounding_dino+sam", "ram+grounding_dino", "semantic-sam", "seem", ], 
        "retrieval":["QnA_doc"], 
        "futureAI":["malicious_path", "url_detection", "event_chain"]
    }
    use_image = False
    raw_mess = state.messages[-2][1]
    if not isinstance(raw_mess, str):
        raw_mess = re.findall("(.*)\n<image>",raw_mess[0])[0]
        use_image = True
    print("--------raw_mess: ", raw_mess)
    api_name = "retrieval_topic"
    worker_topic_addr = get_worker_addr(controller_url, api_name)
    print("----worker_topic_addr: ", worker_topic_addr)
    api_params = {
        "query": raw_mess,
        "top_k": 20
    }
    # topic_response = requests.post(
    #         worker_topic_addr + "/worker_generate_topic",
    #         headers=headers,
    #         params=api_params,
    #     ).json()
    # topic = topic_response["text"]
    topic = "QnA"
    print("--------topic: ", topic)
    if topic in topic_type["retrieval"]:
        doc_response = requests.post(
            worker_topic_addr + "/worker_generate_doc",
            headers=headers,
            params={"query":raw_mess, "top_k": 20}
        )
        for chunk in doc_response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,)*4 + (enable_btn,)*2
                else:
                    output = data["text"] + \
                        f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,)*3 + (enable_btn,)*3
                    return
                time.sleep(0.03)
            else:
                state.messages[-1][-1] = state.messages[-1][-1][:-1]
                yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,)*4 + (enable_btn,)*2
        return
    #///////////////////////////////////////////////////////////////
    #-------------------------Gemini--------------------------------
    # api_name = "gemini"
    # worker_gemini_addr = get_worker_addr(controller_url, api_name)
    # print("-------worker_gemini_addr: ", worker_gemini_addr)
    # pload = {
    #     "prompt": raw_mess
    # }
    # all_images = state.get_images()
    
    # files = [("files", image) for image in all_images]
    # if len(files)==0:
    #     files = None
    # # try:
    # gemini_response = requests.post(worker_gemini_addr + "/worker_generate_stream",
    #                             headers=headers, params=pload, files=files)
    # for chunk in gemini_response.iter_lines(decode_unicode=False, delimiter=b"\0"):
    #     # print("----------chunk: ", chunk)
    #     if chunk:
    #         data = json.loads(chunk.decode())
    #         if data["error_code"] == 0:
    #             output = data["text"].strip()
    #             state.messages[-1][-1] = output + "▌"
    #             yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,)*4 + (enable_btn,)*2
    #         else:
    #             output = data["text"] + \
    #                 f" (error_code: {data['error_code']})"
    #             state.messages[-1][-1] = output
    #             yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,)*3 + (enable_btn,)*3
    #             return
    #         time.sleep(0.03)
    #     else:
    #         state.messages[-1][-1] = state.messages[-1][-1][:-1]
    #         yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,)*4 + (enable_btn,)*2
    # return
#///////////////////////////////////////////////////////////////

    # Query worker address
    # controller_url = "http://localhost:21001"    
    # controller_url = controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    print("----ret: ", ret)
    worker_addr = ret.json()["address"]
    print("-----worker_addr: ", worker_addr)
    logger_app.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()
    # import ipdb; ipdb.set_trace()

    # Save images
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest()
                      for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)
    # import ipdb; ipdb.set_trace()

    print("--------prompt: ", prompt)
    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger_app.info(f"==== request ====\n{pload}\n==== request ====")

    pload['images'] = state.get_images()
    # print("---pload: ", pload)

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=10)
        # import ipdb; ipdb.set_trace()
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6
                else:
                    output = data["text"] + \
                        f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        print("error: ", e)
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
        return

    # remove the cursor
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (enable_btn,) * 6

    # check if we need tools
    model_output_text = state.messages[-1][1]
    # import ipdb; ipdb.set_trace()
    print("model_output_text: ", model_output_text,
          "Now we are going to parse the output.")
    # parse the output

    # import ipdb; ipdb.set_trace()

    try:
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        matches = re.findall(pattern, model_output_text, re.DOTALL)
        print("------matches: ", matches)
        # import ipdb; ipdb.set_trace()
        if len(matches) > 0:
            # tool_cfg = json.loads(matches[0][1].strip())
            try:
                tool_cfg = json.loads(matches[0][1].strip())
            except Exception as e:
                tool_cfg = json.loads(
                    matches[0][1].strip().replace("\'", "\""))
            print("tool_cfg:", tool_cfg)
        else:
            tool_cfg = None
    except Exception as e:
        logger_app.info(f"Failed to parse tool config: {e}")
        tool_cfg = None

    # run tool augmentation
    print("trigger tool augmentation with tool_cfg: ", tool_cfg)
    if tool_cfg is not None and len(tool_cfg) > 0:
        assert len(
            tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)
        api_name = tool_cfg[0]['API_name']
        tool_cfg[0]['API_params'].pop('image', None)
        images = state.get_raw_images()
        if len(images) > 0:
            image = images[0]
        else:
            image = None
        api_paras = {
            'image': image,
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            **tool_cfg[0]['API_params']
        }
        if api_name in ['inpainting']:
            api_paras['mask'] = getattr(state, 'mask_rle', None)
        if api_name in ['openseed', 'controlnet']:
            if api_name == 'controlnet':
                api_paras['mask'] = getattr(state, 'image_seg', None)
            api_paras['mode'] = api_name
            api_name = 'controlnet'
        if api_name == 'seem':
            reference_image = getattr(state, 'reference_image', None)
            reference_mask = getattr(state, 'reference_mask', None)
            api_paras['refimg'] = reference_image
            api_paras['refmask'] = reference_mask
            # extract ref image and mask
            

        # import ipdb; ipdb.set_trace()
        tool_worker_addr = get_worker_addr(controller_url, api_name)
        print("tool_worker_addr: ", tool_worker_addr)
        tool_response = requests.post(
            tool_worker_addr + "/worker_generate",
            headers=headers,
            json=api_paras,
        ).json()
        tool_response_clone = copy.deepcopy(tool_response)
        print("tool_response: ", tool_response)

        # clean up the response
        masks_rle = None
        edited_image = None
        image_seg = None  # for openseed
        iou_sort_masks = None
        if 'boxes' in tool_response:
            tool_response['boxes'] = [[R(_b) for _b in bb]
                                      for bb in tool_response['boxes']]
        if 'logits' in tool_response:
            tool_response['logits'] = [R(_l) for _l in tool_response['logits']]
        if 'scores' in tool_response:
            tool_response['scores'] = [R(_s) for _s in tool_response['scores']]
        if "masks_rle" in tool_response:
            masks_rle = tool_response.pop("masks_rle")
        if "edited_image" in tool_response:
            edited_image = tool_response.pop("edited_image")
        if "size" in tool_response:
            _ = tool_response.pop("size")
        if api_name == "easyocr":
            _ = tool_response.pop("boxes")
            _ = tool_response.pop("scores")
        if "retrieval_results" in tool_response:
            tool_response['retrieval_results'] = [
                {'caption': i['caption'], 'similarity': R(i['similarity'])}
                for i in tool_response['retrieval_results']
            ]
        if "image_seg" in tool_response:
            image_seg = tool_response.pop("image_seg")
        if "iou_sort_masks" in tool_response:
            iou_sort_masks = tool_response.pop("iou_sort_masks")
        if len(tool_response) == 0:
            tool_response['message'] = f"The {api_name} has processed the image."
        # hack
        if masks_rle is not None:
            state.mask_rle = masks_rle[0]
        if image_seg is not None:
            state.image_seg = image_seg

        # if edited_image is not None:
        #     edited_image

        # build new response
        new_response = f"{api_name} model outputs: {tool_response}\n\n"
        print("----------new_response: ", new_response)
        first_question = state.messages[-2][-1]
        if isinstance(first_question, tuple):
            first_question = first_question[0].replace("<image>", "")
        first_question = first_question.strip()

        # add new response to the state
        state.append_message(state.roles[0],
                             new_response +
                             "Please summarize the model outputs and answer my first question: {}".format(
                                 first_question)
                             )
        state.append_message(state.roles[1], None)

        # Construct prompt
        prompt2 = state.get_prompt()
        print("--------prompt2: ", prompt2)

        # Make new requests
        pload = {
            "model": model_name,
            "prompt": prompt2,
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_new_tokens), 1536),
            "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
            "images": f'List of {len(state.get_images())} images: {all_image_hash}',
        }
        logger_app.info(f"==== request ====\n{pload}")
        pload['images'] = state.get_images()

        state.messages[-1][-1] = "▌"
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6

        try:
            # Stream output
            response = requests.post(worker_addr + "/worker_generate_stream",
                                     headers=headers, json=pload, stream=True, timeout=10)
            # import ipdb; ipdb.set_trace()
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][len(prompt2):].strip()
                        state.messages[-1][-1] = output + "▌"
                        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6
                    else:
                        output = data["text"] + \
                            f" (error_code: {data['error_code']})"
                        state.messages[-1][-1] = output
                        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
                        return
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            state.messages[-1][-1] = server_error_msg
            yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
            return

        # remove the cursor
        state.messages[-1][-1] = state.messages[-1][-1][:-1]

        # add image(s)
        if edited_image is not None:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(edited_image))).convert("RGB")
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if image_seg is not None:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(image_seg))).convert("RGB")
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if iou_sort_masks is not None:
            assert isinstance(
                iou_sort_masks, list), "iou_sort_masks should be a list, but got: {}".format(iou_sort_masks)
            edited_image_pil_list = [Image.open(
                BytesIO(base64.b64decode(i))).convert("RGB") for i in iou_sort_masks]
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil_list, "Crop")
        if api_name in ['grounding_dino', 'ram+grounding_dino', 'blip2+grounding_dino']:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
            edited_image_pil = plot_boxes(edited_image_pil, tool_response)
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if api_name in ['grounding_dino+sam', 'grounded_sam']:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
            edited_image_pil = plot_boxes(edited_image_pil, tool_response)
            edited_image_pil = plot_masks(
                edited_image_pil, tool_response_clone)
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if api_name in ['sam']:
            if 'points' in tool_cfg[0]['API_params']:
                edited_image_pil = Image.open(
                    BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
                edited_image_pil = plot_masks(
                    edited_image_pil, tool_response_clone)
                tool_response_clone['points'] = tool_cfg[0]['API_params']['points']
                tool_response_clone['point_labels'] = tool_cfg[0]['API_params']['point_labels']
                edited_image_pil = plot_points(
                    edited_image_pil, tool_response_clone)

                state.messages[-1][-1] = (state.messages[-1]
                                          [-1], edited_image_pil, "Crop")
            else:
                assert 'boxes' in tool_cfg[0]['API_params'], "not find 'boxes' in {}".format(
                    tool_cfg[0]['API_params'].keys())
                edited_image_pil = Image.open(
                    BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
                edited_image_pil = plot_boxes(edited_image_pil, tool_response)
                tool_response_clone['boxes'] = tool_cfg[0]['API_params']['boxes']
                edited_image_pil = plot_masks(
                    edited_image_pil, tool_response_clone)
                state.messages[-1][-1] = (state.messages[-1]
                                          [-1], edited_image_pil, "Crop")

        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (enable_btn,) * 6

    finish_tstamp = time.time()
    logger_app.info(f"{output}")

    # models = get_model_list()

    # FIXME: disabled temporarily for image generation.
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(force_str=True),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

def add_text(state, text, image_dict, ref_image_dict, image_process_mode, with_debug_parameter_from_state, request: gr.Request):
    # dict_keys(['image', 'mask'])
    if image_dict is not None:
        image = image_dict['image']
    else:
        image = None
    logger_app.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None) + (no_change_btn,) * 5
    moderate = False
    if moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        state = default_conversation.copy()

        # a hack, for mask
        sketch_mask = image_dict['mask']
        if sketch_mask is not None:
            text = (text[0], text[1], text[2], sketch_mask)
            # check if visual prompt is used
            bounding_box = get_mask_bbox(sketch_mask)
            if bounding_box is not None:
                text_input_new = text[0] + f"\nInput box: {bounding_box}"
                text = (text_input_new, text[1], text[2], text[3])
                
        if ref_image_dict is not None:
            # text = (text[0], text[1], text[2], text[3], {
            #     'ref_image': ref_image_dict['image'],
            #     'ref_mask': ref_image_dict['mask']
            # })
            state.reference_image = b64_encode(ref_image_dict['image'])
            state.reference_mask = b64_encode(ref_image_dict['mask'])

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    print(state)
    return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None, None) + (disable_btn,) * 6

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