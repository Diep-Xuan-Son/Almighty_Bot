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


#-------------test_gemma2b------------------------
import json
def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark

prompt_tem = ("You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.\n"
			"This is the user's question: {question}\n"
			"This is tool list:\n"
			"{Tool_list}\n"
			"Please note that: \n"
			"1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using one single tool in the tool list.\n"
			"2. If one subtask need the results from other subtask, you can should write clearly. For example:"
			"{{\"Tasks\": [\"Convert 23 km/h to X km/min by 'divide_'\", \"Multiply X km/min by 45 min to get Y by 'multiply_'\"]}}\n"
			"3. You must ONLY output in a parsible JSON format. An example output looks like:\n"
			"'''\n"
			"{{\"Tasks\": [\"Task 1\", \"Task 2\", ...]}}\n"
			"'''\n"
			"Output:")
question = "Calculate the sum of 123 and 456"
Tool_dic = read_jsonline('test/tool_instruction/tool_dic.jsonl')
Tool_list = []
for ele in Tool_dic:
    Tool_list.append(str(ele))
dict_input_variables={"question":question, "Tool_list":Tool_list}
# print(prompt_tem.format(**dict_input_variables))
# exit()
prompt_tem = prompt_tem.format(**dict_input_variables)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./weights/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("./weights/gemma-2b", device_map="auto", torch_dtype=torch.bfloat16)

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))
chat = [
	{ "role": "system", "content": "You are a helpful assistant" },
	{ "role": "user", "content": prompt_tem }
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

print(tokenizer.decode(outputs[0]))