import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

from base.libs import *
from base.constants import *
from conversation2 import Conversation
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All, Ollama, HuggingFaceHub
import importlib.util

def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark
def read_json(address):
    with open(address, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data
def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name

class Agents():
    def __init__(self,model_path="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_path = model_path
        # self.previous_log = {}
        # self.tool_used = []
        # self.answer_bad = []
        # self.answer_good = []

    def Call_function(self, B, arg):
        app_path = 'test/math_func.py'
        spec = importlib.util.spec_from_file_location('math', app_path)
        app_module = importlib.util.module_from_spec(spec)
        # print(dir(app_module))
        spec.loader.exec_module(app_module)
        # print(dir(app_module))
        # print(hasattr(app_module, B))
        # exit()
        if hasattr(app_module, B):
            function_B = getattr(app_module, B)
            try:
                call_result = function_B(arg['input'])
                return call_result
            except Exception as e:
                try:
                    arg = {change_name(k.lower()): v for k, v in arg.items()}
                    call_result = function_B(arg['input'])
                    return call_result
                except Exception as e:
                    try:
                        arg = {change_name(k.lower()): v for k, v in arg.items()}
                        arg = {change_name(k.replace("-", "_")): v for k, v in arg.items()}
                        call_result = function_B(arg['input'])
                        return call_result
                    except Exception as e:
                        print(f"fails: {e}")
                        with open('wrong_log.json', 'a+', encoding='utf-8') as f:
                            line = json.dumps({
                                "id": id,
                                "parameters": arg,
                                "wrong": str(e)
                            }, ensure_ascii=False)
                            f.write(line + '\n')
                        return -1
        else:
            with open('wrong_log.json', 'a+', encoding='utf-8') as f:
                line = json.dumps({
                    "id": id,
                    "parameters": arg,
                    "wrong": f"No function named {B} in {app_path}"
                }, ensure_ascii=False)
                f.write(line + '\n')
            return (f"No function named {B} in {app_path}")

    def task_decompose(self, question, Tool_dic):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")      # "google/gemma-7b", "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
            "{{\"Tasks\": [string 1, string 2, ...]}}\n"
            "'''\n"
            "Output:\n\n"
        )
        template_human = human_message_prompt.prompt.template
        input_variables = human_message_prompt.prompt.input_variables
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        # print(chat_prompt.__dict__)
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        Tool_list = []
        for ele in Tool_dic:
            Tool_list.append(ele)

        ind = 0
        while True:
            try:
                result = chain.run(question=question, Tool_list=Tool_list)
                pattern = r'{(.*)}'
                result = re.findall(pattern, result.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
                result = eval(f'{{{result[0]}}}')
                a = result["Tasks"]
                break
            except Exception as e:
                print(f"task decompose fails: {e}")
                if ind > 10:
                    return -1
                ind += 1
                continue
        return a

    def task_topology(self, question, task_ls):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
        template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "Given a complex user's question, I have decompose this question into some simple subtasks"
            "I think there exists a logical connections and order amontg the tasks. "
            "Thus you need to help me output this logical connections and order.\n"
            "You must ONLY output in a parsible JSON format with the following format:\n"
            "'''\n"
            "[{{\"task\": task, \"id\", task_id, \"dep\": [dependency_task_id1, dependency_task_id2, ...]}}]\n"
            "'''\n"
            "The \"dep\" field denotes the id of the previous task which generates a new resource upon which the current task depends. If there are no dependencies, set \"dep\" to -1.\n\n"
            "This is user's question: {question}\n"
            "These are subtasks of this question:\n"
            "{task_ls}\n"
            "Output:\n\n"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        ind = 0
        while True:
            try:
                result = chain.run(question=question, task_ls=task_ls)
                # print(result)
                # exit()
                result = result.replace("```", "").strip().split('\n\n')[-1]
                result = eval(result)
                for i in range(len(result)):
                    if isinstance(result[i]['dep'], str):
                        temp = []
                        for ele in result[i]['dep'].split(','):
                            temp.append(int(ele))
                        result[i]['dep'] = temp
                    elif isinstance(result[i]['dep'], int):
                        result[i]['dep'] = [result[i]['dep']]
                    elif isinstance(result[i]['dep'], list):
                        temp = []
                        for ele in result[i]['dep']:
                            temp.append(int(ele))
                        result[i]['dep'] = temp
                    elif result[i]['dep'] == -1:
                        result[i]['dep'] = [-1]
                a = result[i]['dep'][0]
                return result
            except Exception as e:
                print(f"task topology fails: {e}")
                if ind > 10:
                    return -1
                ind += 1
                continue
        return result

    def choose_tool(self, question, Tool_dic, tool_used):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
        template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "This is the user's question: {question}\n"
            "These are the tools you can select to solve the question:\n"
            "Tool List:\n"
            "{Too_list}\n\n"
            "Please note that: \n"
            "1. You should only chooce one tool the Tool List to solve this question.\n"
            "2. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:\n"
            # "'''\n"
            "Example 1: {{\"ID\": 1}}\n"
            "Example 2: {{\"ID\": 2}}\n"
            # "'''\n"
            "Output:\n\n"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        ind = 0
        Tool_list = []
        for ele in Tool_dic:
            for key in ele.keys():
                if str(key) not in tool_used:
                    Tool_list.append(f'''ID: {key}\n{ele[key]}''')

        while True:
            try:
                result = chain.run(question=question,
                                   Too_list=Tool_dic)
                pattern = r'{(.*)}'
                clean_answer = re.findall(pattern, result.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
                clean_answer = eval(f'{{{clean_answer[0]}}}')
                # print(clean_answer)
                # exit()
                break
            except Exception as e:
                print(f"choose tool fails: {e}")
                print(result)
                if ind > 10:
                    return -1
                ind += 1
                continue
        return clean_answer

    def choose_parameter(self, API_instruction, api, api_dic, question):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
        template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "This is an API tool documentation. Given a user's question, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
            "This is API tool documentation: {api_dic}\n"
            "Please note that: \n"
            "1. The Example in the API tool documentation can help you better understand the use of the API.\n"
            "2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{\"Parameters\":{{}}}}\n"
            "3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
            "4. If you need to use this API multiple times, please set \"Parameters\" to a list.\n"
            "5. You must ONLY output in a parsible JSON format. Two examples output looks like:\n"
            # "'''\n"
            "Example 1: {{\"Parameters\":{{\"input\": [1,2,3]}}}}\n"
            "Example 2: {{\"Parameters\":[{{\"input\": [1,2,3]}}, {{\"input\": [2,3,4]}}]}}\n"
            # "'''\n"
            "This is user's question: {question}\n"
            "Output:\n\n"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        ind = 0
        while True:
            try:
                result = chain.run(api_dic=api_dic,
                                   question=question, )
                pattern = r'{(.*)}'
                clean_answer = re.findall(pattern, result.replace(": true", ": True").replace(":true", ": True")\
                                                        .replace(":false", ": False").replace(": false", ": False")\
                                                        .replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
                clean_answer = eval(f'{{{clean_answer[0]}}}')
                # print(clean_answer)
                # exit()
                a = clean_answer["Parameters"]
                return a
            except Exception as e:
                print(f"Choose Parameter fails: {e}")
                if ind > 10:
                    return -1
                ind += 1
                continue
        return a

    def choose_parameter_depend(self, API_instruction, api, api_dic, question, previous_log):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
        template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "Given a user's question and a API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
            "Please note that: \n"
            "1. The Example in the API tool documentation can help you better understand the use of the API.\n"
            "2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {{\"Parameters\":{{}}}}\n"
            "3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
            "4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers for your reference.\n"
            "5. If you need to use this API multiple times,, please set \"Parameters\" to a list.\n"
            "6. You must ONLY output in a parsible JSON format. Two examples output looks like:\n"
            # "'''\n"
            "Example 1: {{\"Parameters\":{{\"input\": [1,2,3]}}}}\n"
            "Example 2: {{\"Parameters\":[{{\"input\": [1,2,3]}}, {{\"input\": [2,3,4]}}]}}\n"
            # "'''\n"
            "There are logs of previous questions and answers: \n {previous_log}\n"
            "This is the current user's question: {question}\n"
            "This is API tool documentation: {api_dic}\n"
            "Output:\n\n"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        ind = 0
        while True:
            try:
                result = chain.run(api_dic=api_dic,
                                   question=question,
                                   previous_log=previous_log)
                pattern = r'{(.*)}'
                clean_answer = re.findall(pattern, result.replace(": true", ": True").replace(":true", ": True")\
                                                        .replace(":false", ": False").replace(": false", ": False")\
                                                        .replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
                clean_answer = eval(f'{{{clean_answer[0]}}}')
                a = clean_answer["Parameters"]

                return a
            except Exception as e:
                print(f"choose parameter depend fails: {e}")
                if ind > 10:
                    return -1
                ind += 1
                continue
        return a

    def answer_generation_depend(self, question, API_instruction, call_result, previous_log):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
        template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "You should answer the question based on the response output by the API tool."
            "Please note that:\n"
            "1. Try to organize the response into a natural language answer.\n"
            "2. We will not show the API response to the user, "
            "thus you need to make full use of the response and give the information "
            "in the response that can satisfy the user's question in as much detail as possible.\n"
            "3. If the API tool does not provide useful information in the response, "
            "please answer with your knowledge.\n"
            "4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers.\n"
            "There are logs of previous questions and answers: \n {previous_log}\n"
            "This is the user's question: {question}\n"
            "This is the response output by the API tool: \n{call_result}\n"
            "We will not show the API response to the user, "
            "thus you need to make full use of the response and give the information "
            "in the response that can satisfy the user's question in as much detail as possible.\n"
            "Output:\n\n"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        ind = 0
        while True:
            try:
                result = chain.run(question=question,
                                   API_instruction=API_instruction,
                                   call_result=call_result,
                                   previous_log=previous_log)
                clean_answer = result.replace("```", "").strip().split('\n\n')[-1]
                break
            except Exception as e:
                print(f"answer generation depend fails: {e}")
                if ind > 2:
                    return -1
                ind += 1
                continue
        return clean_answer

    def answer_summarize(self, question, answer_task):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
        template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "We break down a complex user's problems into simple subtasks and provide answers to each simple subtask. "
            "You need to organize these answers to each subtask and form a self-consistent final answer to the user's question\n"
            "This is the user's question: {question}\n"
            "These are subtasks and their answers: {answer_task}\n"
            "Final answer:\n\n"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        result = chain.run(question=question, answer_task=answer_task)
        clean_answer = result.replace("```", "").strip().split('\n\n')[-1]
        return clean_answer

    def answer_check(self, question, answer):
        chat = HuggingFaceHub(repo_id=self.model_path, huggingfacehub_api_token="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")
        template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "Please check whether the response can reasonably and accurately answer the question."
            "If can, please output 'YES'; If not, please output 'NO'\n"
            "You need to give reasons first and then decide whether the response can reasonably and accurately answer the question. You must only output in a parsible JSON format. Two example outputs look like:\n"
            "Example 1: {{\"Reason\": \"The reason why you think the response can reasonably and accurately answer the question\", \"Choice\": \"Yes\"}}\n"
            "Example 2: {{\"Reason\": \"The reason why you think the response cannot reasonably and accurately answer the question\", \"Choice\": \"No\"}}\n"
            "This is the user's question: {question}\n"
            "This is the response: {answer}\n"
            "Output:\n\n"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        result = chain.run(question=question, answer=answer)
        pattern = r'{(.*)}'
        result = re.findall(pattern, result.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
        result = eval(f'{{{result[0]}}}')
        print(result)
        if 'yes'.lower() in result["Choice"].lower():
            return 1
        else:
            return -1

    def retrieval(self, question, Tool_dic, dataset, tool_used, previous_log=None):
        tool_id = self.choose_tool(question, Tool_dic, tool_used)
        if tool_id == -1:
            return tool_id, "", "", "", ""
        tool_instruction = dataset[str(tool_id["ID"])]
        print(tool_instruction)
        API_instruction = tool_instruction["API_description"]
        API_tool = tool_instruction["standardized_name"]

        api_selection = [API_tool]
        api_result = []
        for api in api_selection:
            if previous_log is None:
                parameter = self.choose_parameter(API_instruction, api,
                                         tool_instruction["Usage"], question)
            else:
                parameter = self.choose_parameter_depend(API_instruction, api,
                                                tool_instruction["Usage"],
                                                question, previous_log)
            if parameter == -1:
                continue
            api_result.append({"api_name": api, "parameters": parameter})
        if len(api_result) == 0:
            call_result = ""
            return tool_id, api_result, call_result, tool_instruction, API_instruction

        call_results = []
        for api in api_result:
            if isinstance(api["parameters"], dict):
                parameters = {}
                for key in api["parameters"]:
                    value = api["parameters"][key]
                    key = change_name(key)
                    parameters[key] = value
                call_result = self.Call_function(API_tool, parameters)
                if call_result == -1:
                    continue
                call_results.append(str(call_result))
            elif isinstance(api["parameters"], list):
                for para_ls in api["parameters"]:
                    parameters = {}
                    for key in para_ls:
                        value = para_ls[key]
                        key = change_name(key)
                        parameters[key] = value
                    call_result = self.Call_function(API_tool, parameters)
                    if call_result == -1:
                        continue
                    call_results.append(str(call_result))
        call_result = '\n\n'.join(call_results)
        print(call_result)
        return tool_id, api_result, call_result, tool_instruction, API_instruction

def task_execution(state, agent):
    tool_used = []
    answer_bad = []
    answer_good = []
    previous_log = None

    question = state.messages[-1]["User"]
    temp = agent.task_decompose(question=question, Tool_dic=state.tool_dic)
    print(temp)
    task_ls = []
    for t in range(len(temp)):
        task_ls.append({"task": temp[t], "id": t + 1})
    # print(task_ls)
    task_ls = agent.task_topology(question, task_ls)
    print(task_ls)
    task_depend = {'Original Question': question}
    for task_dic in task_ls:
        task_depend[task_dic['id']] = {'task': task_dic['task'], 'answer': ''}
    print(task_depend)
    answer_task = []
    for task_dic in task_ls:
        task = task_dic['task']

        tool_id, api_result, call_result, tool_instruction, API_instruction = agent.retrieval(task, Tool_dic,
                                                                                        dataset,
                                                                                        tool_used,
                                                                                        previous_log=previous_log)
        if len(str(call_result)) > 5000:
            call_result = str(call_result)[:5000]
        answer = agent.answer_generation_depend(task, API_instruction, call_result, previous_log)

        check_index = 1
        if str(call_result).strip() == '-1' or str(call_result).strip() == '':
            check_index = -1
        if check_index == 1:
            answer_task.append({'task': task, 'answer': answer})
            # tool_instruction_ls.append(tool_instruction)
            # api_result_ls.append(api_result)
            # call_result_ls.append(call_result)
            tool_used.append(str(tool_id["ID"]))
        else:
            answer_bad.append({'task': task, 'answer': answer})

        task_depend[task_dic['id']]['answer'] = answer
        previous_log = task_depend

    final_answer = agent.answer_summarize(question, answer_task)
    check_index = agent.answer_check(question, final_answer)
    print(final_answer)

if __name__=="__main__":
    model_path="mistralai/Mixtral-8x7B-Instruct-v0.1"
    agent = Agents(model_path)
    param_conver = {"_id": "conver1", \
                    "roles": ["User", "Assistant"], \
                    "messages": [{"User": "Calculate sum of 1 and 2", "Assistant": ""},], \
                    "images": [[],], \
                    "voices": [[],], \
                    "image_process_mode": "Pad", \
                    "tool_dic": [], \
                    "functions_data": {}}
    default_conversation = Conversation(**param_conver)
    Tool_dic = read_jsonline('test/tool_instruction/tool_dic.jsonl')
    dataset = read_json('test/tool_instruction/functions_data.json')
    default_conversation.tool_dic = Tool_dic
    default_conversation.functions_data = dataset
    task_execution(default_conversation, agent)
