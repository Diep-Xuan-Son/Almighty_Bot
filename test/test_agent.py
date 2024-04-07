import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import json
from langchain_community.llms import GPT4All, Ollama

openai.api_key = ""

def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark

def task_decompose(**kwargs):
    question = kwargs['question']
    Tool_dic = kwargs['Tool_dic']
    model_name = kwargs['model_name']
    # chat = ChatOpenAI(openai_api_key="", model_name=model_name)
    # chat = Ollama(model="gemma:2b", n_threads=8)
    chat = GPT4All(model="./models/gpt4all-model.bin", n_threads=8)
    template = "You are a helpful assistant."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.\n"
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
        "Output:"
    )
    # print(system_message_prompt.prompt.template)
    # print(human_message_prompt.prompt.__dict__)
    template_human = human_message_prompt.prompt.template
    input_variables = human_message_prompt.prompt.input_variables
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # print(chat_prompt.__dict__)
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    Tool_list = []
    for ele in Tool_dic:
        Tool_list.append(str(ele))
    kwargs['Tool_list'] = Tool_list
    # print(Tool_list)
    # print(input_variables)
    dict_input_variables = {}
    for input_variable in input_variables:
        if input_variable in kwargs:
            dict_input_variables[input_variable] = kwargs[input_variable]

    print(template_human.format(**dict_input_variables))
    # exit()
    ind = 0
    while True:
        try:
            result = chain.run(question=question, Tool_list=Tool_list)
            print(result)
            exit()
            result = eval(result.split('\n\n')[0])
            a = result["Tasks"]
            break
        except Exception as e:
            print(f"task decompose fails: {e}")
            if ind > 10:
                return -1
            ind += 1
            continue
    return result

if __name__=="__main__":
    question = "Calculate the sum of 123 and 456"
    Tool_dic = read_jsonline('test/tool_instruction/tool_dic.jsonl')
    model_name = 'gpt-3.5-turbo'
    task_decompose(question=question, Tool_dic=Tool_dic, model_name=model_name)




HumanMessagePromptTemplate.from_template(
    "Answer the questions using the following context:\n"
    "{context}")











# def choose_tool(question, Tool_dic, tool_used, model_name):
#     chat = ChatOpenAI(model_name=model_name)
#     template = "You are a helpful assistant."
#     system_message_prompt = SystemMessagePromptTemplate.from_template(template)
#     human_message_prompt = HumanMessagePromptTemplate.from_template(
#         "This is the user's question: {question}\n"
#         "These are the tools you can select to solve the question:\n"
#         "Tool List:\n"
#         "{Too_list}\n\n"
#         "Please note that: \n"
#         "1. You should only chooce one tool the Tool List to solve this question.\n"
#         "2. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:\n"
#         "'''\n"
#         "Example 1: {{\"ID\": 1}}\n"
#         "Example 2: {{\"ID\": 2}}\n"
#         "'''\n"
#         "Output:"
#     )
#     print(system_message_prompt)
#     exit()
#     chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
#     chain = LLMChain(llm=chat, prompt=chat_prompt)
#     ind = 0
#     Tool_list = []
#     for ele in Tool_dic:
#         for key in ele.keys():
#             if str(key) not in tool_used:
#                 Tool_list.append(f'''ID: {key}\n{ele[key]}''')
#     while True:
#         try:
#             result = chain.run(question=question,
#                                Too_list=Tool_dic)
#             clean_answer = eval(result.split("(")[0].strip())
#             # clean_answer = lowercase_parameter_keys(clean_answer)
#             # print(clean_answer)
#             break
#         except Exception as e:
#             print(f"choose tool fails: {e}")
#             print(result)
#             if ind > 10:
#                 return -1
#             ind += 1
#             continue
#     return clean_answer