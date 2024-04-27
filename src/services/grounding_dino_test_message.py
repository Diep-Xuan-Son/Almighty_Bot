"""Send a test message."""
import argparse
import json
import time
from io import BytesIO
import cv2
from groundingdino.util.inference import annotate
import numpy as np
import torchvision

import requests
from PIL import Image
import base64

import torch
import torchvision.transforms.functional as F

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # import ipdb; ipdb.set_trace()
    # resize if needed
    w, h = img.size
    if max(h, w) > 800:
        if h > w:
            new_h = 800
            new_w = int(w * 800 / h)
        else:
            new_w = 800
            new_h = int(h * 800 / w)
        # import ipdb; ipdb.set_trace()
        img = F.resize(img, (new_h, new_w))
    return img

def encode(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    # img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return buffered.getvalue()


def main():
    model_name = args.model_name

    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return

    headers = {"User-Agent": "FastChat Client"}
    if args.send_image:
        img = load_image(args.image_path)
        img_arg = encode(img)
    else:
        img_arg = args.image_path
    files = [("image", img_arg)]
    datas = {
        "caption": [args.caption],
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }
    print(datas)
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        data=datas,
        files=files
    )
    toc = time.time()
    print(f"Time: {toc - tic:.3f}s")

    print("detection result:")
    # print(response.json())
    # response is 'Response' with :
    # ['_content', '_content_consumed', '_next', 'status_code', 'headers', 'raw', 'url', 'encoding', 'history', 'reason', 'cookies', 'elapsed', 'request', 'connection', '__module__', '__doc__', '__attrs__', '__init__', '__enter__', '__exit__', '__getstate__', '__setstate__', '__repr__', '__bool__', '__nonzero__', '__iter__', 'ok', 'is_redirect', 'is_permanent_redirect', 'next', 'apparent_encoding', 'iter_content', 'iter_lines', 'content', 'text', 'json', 'links', 'raise_for_status', 'close', '__dict__', '__weakref__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']

    # visualize
    img = base64.b64decode(response.json()["image"])
    img = np.array(Image.open(BytesIO(img)))
    cv2.imwrite("annotated_image1.jpg", img)

    res = response.json()["result"]
    boxes = torch.Tensor(res["boxes"])
    logits =  torch.Tensor(res["logits"])
    phrases = res["phrases"]
    image_source = np.array(Image.open(args.image_path))
    print(image_source.shape)
    h, w, _ = image_source.shape
    xyxy = (boxes * torch.Tensor([w, h, w, h])).cpu().detach().numpy().astype(int)
    print(xyxy)
    for xy in xyxy:
        cv2.rectangle(image_source, xy[:2], xy[2:], (255, 0, 0) , 2) 
    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_image.jpg", image_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='grounding_dino')

    # model parameters
    parser.add_argument(
        "--caption", type=str, default="everything"
    )
    parser.add_argument(
        "--image_path", type=str, default="./data_test/2024-03-26/e75f292184a71c98df096cba7e880afa.jpg"
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.4,
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25,
    )
    parser.add_argument(
        "--send_image", default=True, action="store_true",
    )
    args = parser.parse_args()

    main()
