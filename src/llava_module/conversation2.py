import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    _id: str
    roles: List[str]
    messages: List[dict]
    images: List[List[str]]
    voices: List[List[str]]
    image_process_mode: List[str]
    tool_dic: List[dict]
    functions_data: dict

    def append_message(self, role, message):
        self.messages.append([{role, message}])

    def get_images(self, return_pil=False):
        images = []
        if len(images[-1])==0:
            return images
        for img_path in images[-1]:
            image = Image.open(img_path)
            if image_process_mode[-1] == "Pad":
                def expand2square(pil_img, background_color=(122, 116, 104)):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color)
                        result.paste(
                            pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color)
                        result.paste(
                            pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image)
            elif image_process_mode in ["Default", "Crop"]:
                pass
            elif image_process_mode == "Resize":
                image = image.resize((336, 336))
            elif image_process_mode == "None":
                pass
            else:
                raise ValueError(
                    f"Invalid image_process_mode: {image_process_mode}")

            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 800, 400
            shortest_edge = int(
                min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if longest_edge != max(image.size):
                if H > W:
                    H, W = longest_edge, shortest_edge
                else:
                    H, W = shortest_edge, longest_edge
                image = image.resize((W, H))
            if return_pil:
                images.append(image)
            else:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_b64_str = base64.b64encode(
                    buffered.getvalue()).decode()
                images.append(img_b64_str)
        return images

    # def get_voices(self):
    #     voices = []
    #     if len(voices[-1])==0:
    #         return voices
    #     for voice in voices[-1]:
