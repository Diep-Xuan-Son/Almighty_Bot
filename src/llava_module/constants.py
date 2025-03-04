import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# from base.libs import *
import gradio as gr 
from functools import partial

CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_HEART_BEAT_INTERVAL = 60

LOGDIR = "./logs"

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)
R = partial(round, ndigits=2)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

headers = {"User-Agent": "LLaVA-Plus Client"}

# title_markdown = ("""
# # 🌋 MQ-GPT
# ## **L**arge **L**anguage **a**nd **V**ision **A**ssistants that **P**lug and **L**earn to **U**se **S**kills
# """)
title_markdown = ("""
# 🤖 MQ-GPT ⚡⚡⚡
""")  
block_css = """
#buttons button {
    min-width: min(120px,100%);
}
footer {
    visibility: hidden
}
"""
get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""
