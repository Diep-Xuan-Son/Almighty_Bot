from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# article_en = "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'"
# model = MBartForConditionalGeneration.from_pretrained("./weights/Llama2-13b-Language-translate")
# tokenizer = MBart50TokenizerFast.from_pretrained("./weights/Llama2-13b-Language-translate", src_lang="en_XX")

# model_inputs = tokenizer(article_en, return_tensors="pt")

# # translate from English to Hindi
# generated_tokens = model.generate(
#     **model_inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"]
# )
# decode = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(decode)

#----------------vi2en--------------------------
# article_en = "Mẫu iPhone mới nhất có các tính năng ấn tượng và camera mạnh mẽ."
# model = MBartForConditionalGeneration.from_pretrained("./weights/Llama2-13b-Language-translate")
# tokenizer = MBart50TokenizerFast.from_pretrained("./weights/Llama2-13b-Language-translate", src_lang="vi_VN")

# model_inputs = tokenizer(article_en, return_tensors="pt")

# generated_tokens = model.generate(
#     **model_inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
# )
# decode = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(decode)


#-------------test_translate------------------------
import json
def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark

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

# prompt_tem = ("You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.\n"
# 			"This is the user's question: {question}\n"
# 			"This is tool list:\n"
# 			"{Tool_list}\n"
# 			"Please note that: \n"
# 			"1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using one single tool in the tool list.\n"
# 			"2. If one subtask need the results from other subtask, you can should write clearly. For example:"
# 			"{{\"Tasks\": [\"Convert 23 km/h to X km/min by 'divide_'\", \"Multiply X km/min by 45 min to get Y by 'multiply_'\"]}}\n"
# 			"3. You must ONLY output in a parsible JSON format. An example output looks like:\n"
# 			"'''\n"
# 			"{{\"Tasks\": [\"Task 1\", \"Task 2\", ...]}}\n"
# 			"'''\n"
# 			"Output:")
# question = "Calculate the sum of 123 and 456"
# Tool_dic = read_jsonline('test/tool_instruction/tool_dic.jsonl')
# Tool_list = []
# for ele in Tool_dic:
#     Tool_list.append(str(ele))
# dict_input_variables={"question":question, "Tool_list":Tool_list}
# print(prompt_tem.format(**dict_input_variables))
# exit()
# prompt_tem = prompt_tem.format(**dict_input_variables)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
import torch
from datasets import load_dataset
import soundfile as sf
import re
import numpy as np
#-------------------------------vinai_translate-----------------------
# tokenizer_vi2en = AutoTokenizer.from_pretrained("./weights/vinai-translate-vi2en-v2", src_lang="vi_VN")
# # model = AutoModelForCausalLM.from_pretrained("./weights/vinai-translate-vi2en-v2", device_map="auto", torch_dtype=torch.bfloat16)
# model_vi2en =  AutoModelForSeq2SeqLM.from_pretrained("./weights/vinai-translate-vi2en-v2").to(torch.device("cuda"))
# device_vi2en = torch.device("cuda")
# model_vi2en.to(device_vi2en)
# input_text = "Hãy Tìm đối tượng trong bức hình"
# print(translate_vi2en(input_text, tokenizer_vi2en))

# tokenizer_en2vi = AutoTokenizer.from_pretrained("./weights/vinai-translate-en2vi-v2", src_lang="en_XX")
# model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("./weights/vinai-translate-en2vi-v2")
# device_en2vi = torch.device("cuda")
# model_en2vi.to(device_en2vi)
# input_text = "The response is correct because the quotient of 123 and 456 is indeed approximately 0.27. This is because 123 divided by 456 equals 0.26973684210526316, and rounding this to two decimal places gives 0.27"
# print(translate_en2vi(input_text, tokenizer_en2vi))
#//////////////////////////////////////////////////////////////////////

# chat = [
# 	{ "role": "system", "content": "You are a helpful assistant" },
# 	{ "role": "user", "content": prompt_tem }
# ]
# prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# print(prompt)

# inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
# outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

# print(tokenizer.decode(outputs[0]))

#------------------------------voice-------------------------------
#........text2voice----------
# def remove_special_characters(sentence):
#     # Use regular expression to keep only letters, periods, and commas
#     sentence_after_removal =  re.sub(r'[^a-zA-Z\s,.\u00C0-\u1EF9]', ' ,', sentence)
#     return sentence_after_removal
# dataset = load_dataset("./weights/vi-xvector-speechbrain")
# dataset = dataset["train"].to_list()
# dataset_dict = {}
# for rc in dataset:
#     dataset_dict[rc["speaker_id"]] = rc["embedding"]
# # print(dataset_dict.keys())
# # exit()
# speaker_embeddings = torch.tensor(dataset_dict["voice_quynh_anh_spiderum"]) # voice_quynh_anh_spiderum, lisa, tts_kechuyenbenghe, VIVOSSPK46, VIVOSSPK24, steve_job

# processor = SpeechT5Processor.from_pretrained("./weights/speecht5-vietnamese-voiceclone-lsvsc")
# model = SpeechT5ForTextToSpeech.from_pretrained("./weights/speecht5-vietnamese-voiceclone-lsvsc")
# vocoder = SpeechT5HifiGan.from_pretrained("./weights/speecht5_hifigan")
# model.eval()

# separators = r";|\.|!|\?|\n"

# # exit()
# input_text = "Hệ thống truyền dẫn bao gồm hệ thống truyền dẫn quốc gia, hệ thống truyền\
# dẫn kết nối quốc tế, hệ thống vệ tinh, hệ thống truyền dẫn của doanh nghiệp cung\
# cấp dịch vụ trên mạng viễn thông, mạng Internet, các dịch vụ gia tăng trên không\
# gian mạng"
# input_text = remove_special_characters(input_text)
# input_text = input_text.replace(" ", "▁")
# split_texts = re.split(separators, input_text)
# print(split_texts)
# # exit()
# full_speech = []
# for split_text in split_texts:
# 	if split_text != "▁":
# 		split_text = split_text.lower() + "▁"
# 		print(split_texts)
# 		# exit()
# 		inputs = processor.tokenizer(text=split_text, return_tensors="pt")

# 		# generate speech
# 		speech = model.generate(inputs["input_ids"], threshold=0.5, speaker_embeddings=speaker_embeddings, vocoder=vocoder)
# 		print(speech.numpy().shape)
# 		full_speech.append(speech.numpy())
# full_speech = np.concatenate(full_speech)
# sf.write("speech1.wav", full_speech, samplerate=16000)
#///////////////////////
#-------voice2text-------
from scipy.io import wavfile
wavdt = wavfile.read('./data_test/speech1.wav')
print(wavdt[0])
print(wavdt[1].shape)

# chars_to_ignore_regex = '[\\,\\?\\.\\!\\-\\;\\:\\"\\“\\%\\‘\\”\\�\\)\\(\\*)]'
# model = Wav2Vec2ForCTC.from_pretrained("./weights/Fine-Tune-XLSR-Wav2Vec2-Speech2Text-Vietnamese").to(torch.device("cpu"))
# processor = Wav2Vec2Processor.from_pretrained("./weights/Fine-Tune-XLSR-Wav2Vec2-Speech2Text-Vietnamese")
# exit()

# inputs = processor(np.array(wavdt[1], dtype=float), sampling_rate=16_000, return_tensors="pt", padding=True)
# with torch.no_grad():
#     logits = model(inputs.input_values.to("cpu"), attention_mask=inputs.attention_mask.to("cpu")).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# predicted_sentences = processor.batch_decode(predicted_ids)
# print(predicted_sentences)
#////////////////////////
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
inputs = processor(wavdt, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])
#/////////////////////////////////////////////////////////////////