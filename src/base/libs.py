import dataclasses
import datetime
import json
import copy
import base64
from functools import partial
import os
import io
import torch
from pathlib import Path
import cv2
import numpy as np
import re
import time
from io import BytesIO
from PIL import Image
import requests
import hashlib
import pycocotools.mask as mask_util
import pandas as pd
import PyPDF2
import shutil
# from pdfminer.high_level import extract_text
from unstructured.partition.auto import partition

import gradio as gr 
from gradio import processing_utils
from gradio_client.utils import decode_base64_to_file

import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import logging
from typing import List, Union, Tuple
import threading
import uvicorn
from pydantic import BaseModel

from fastapi import FastAPI, Request, BackgroundTasks, Depends, Query, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse

import uuid
from threading import Thread
