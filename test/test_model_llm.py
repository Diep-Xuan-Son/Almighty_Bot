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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
import torch
from datasets import load_dataset
import soundfile as sf
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
processor = SpeechT5Processor.from_pretrained("./weights/speecht5-vietnamese-voiceclone-lsvsc")
model = SpeechT5ForTextToSpeech.from_pretrained("./weights/speecht5-vietnamese-voiceclone-lsvsc")
vocoder = SpeechT5HifiGan.from_pretrained("./weights/speecht5-vietnamese-voiceclone-lsvsc")

# exit()
input_text = "Hãy Tìm đối tượng trong bức hình"
inputs = processor.tokenizer(text=input_text, return_tensors="pt")
embeddings_dataset = load_dataset("./weights/multilingual-xvector", split="train")
print(embeddings_dataset)
speaker_embeddings = torch.tensor(embeddings_dataset[569]["xvector"]).unsqueeze(0)

# generate speech
speech = model.generate(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
print(speech.shape)

#/////////////////////////////////////////////////////////////////