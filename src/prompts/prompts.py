prompt_task_decompose = (
"You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.\n"
"This is the user's question: {question}\n"
"This is tool list:\n"
"{Tool_list}\n"
"Please note that: \n"
"1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using one single tool in the tool list.\n"
"2. If one subtask need the results from other subtask, you can should write clearly. For example:"
"{{\"Tasks\": [\"Convert 23 km/h to X km/min by 'divide_'\", \"Multiply X km/min by 45 min to get Y by 'multiply_'\"]}}\n"
"3. If subtask doesn't focus on the object in the question, please don't use that subtask\n"
"4. Subtask must be written in text form, don't write in code form\n"
"5. You must ONLY ouput in a parsible JSON format. An example output looks like:\n"
# "'''\n"
"{{\"Tasks\": [string 1, string 2, ...]}}\n"
# "'''\n"
"Output:\n\n"
)

prompt_task_topology = (
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

prompt_choose_tool = (
"This is the user's question: {question}\n"
"These are the tools you can select to solve the question:\n"
"Tool List:\n"
"{Too_list}\n\n"
"Please note that: \n"
"1. You should only chooce one tool the Tool List to solve this question.\n"
"2. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two example outputs look like:\n"
"'''\n"
"Example 1: {{\"ID\": 1}}\n"
"Example 2: {{\"ID\": 2}}\n"
"Example 3: {{\"ID\": 10}}\n"
"'''\n"
"Output:\n\n"
)

prompt_choose_parameter = (
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

prompt_choose_parameter_depend = (
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
"Example 3: {{\"Parameters\":[{{\"file\": [\"image\",\"voice\"]}}, {{\"payload\": {{\"input\": [2,3,4]}} }}]}}\n"
# "'''\n"
"There are logs of previous questions and answers: \n {previous_log}\n"
"This is the current user's question: {question}\n"
"This is API tool documentation: {api_dic}\n"
"Output:\n\n"
)

prompt_answer_generation = (
"You should answer the question based on the response output by the API tool."
"Please note that:\n"
"1. Try to organize the response into a natural language answer.\n"
"2. We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.\n"
"3. If the API tool does not provide useful information in the response, please answer with your knowledge.\n"
# "4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers.\n"
# "There are logs of previous questions and answers: \n {previous_log}\n"
"This is the user's question: {question}\n"
"This is the response output by the API tool: \n{call_result}\n"
"We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.\n"
"Output:\n\n"
)

prompt_answer_summarize = (
"We break down a complex user's problems into simple subtasks and provide answers to each simple subtask. "
"You need to organize these answers to each subtask and form a self-consistent final answer to the user's question\n"
"This is the user's question: {question}\n"
"These are subtasks and their answers: {answer_task}\n"
"Final answer:\n\n"
)

prompt_answer_summarize_2 = (
"We break down a complex user's problems by using simple tools and provide answers to each simple tool. "
"You need to organize these answers to each tool and form a self-consistent final answer to the user's question\n"
"This is the user's question: {question}\n"
"These are tools and their answers: {answer_task}\n"
"Final answer:\n\n"
)

prompt_answer_check = (
"Please check whether the response can reasonably and accurately answer the question."
"If can, please output 'YES'; If not, please output 'NO'\n"
"You need to give reasons first and then decide whether the response can reasonably and accurately answer the question. You must only output in a parsible JSON format. Two example outputs look like:\n"
"Example 1: {{\"Reason\": \"The reason why you think the response can reasonably and accurately answer the question\", \"Choice\": \"Yes\"}}\n"
"Example 2: {{\"Reason\": \"The reason why you think the response cannot reasonably and accurately answer the question\", \"Choice\": \"No\"}}\n"
"This is the user's question: {question}\n"
"This is the response: {answer}\n"
"Output:\n\n"
)

prompt_answer_inference = (
"Use your knowledge to answer the user's question.\n"
"If you know something, please share with user.\n"
"If you don't sure about the answer, just say that you don't know, don't try to make up an answer.\n"
"This is the user's question: {question}\n"
"Output:\n\n"
)

prompt_choose_tool_parameter = """
You are an advanced AI agent responsible for selecting the most appropriate tools based on user queries and available data. Your task is to analyze the input, reason about the best tools to use, and output a JSON-formatted response.

Input:
- User question: {question}
- Available tools: 
{tool_descriptions}
Instrutions:
1. Carefully analyze the user's question and any additional context provided.
2. Review the available tools, their descriptions and usage examples.
3. Choose the most suitable tool(s) based on your analysis.
4. For each selected tool, determine the appropriate parameters based on the tool's usage and examples provided.
5. Generate result only in JSON format with following structure:
{{
    "Tools": [
		{{
			"Name": "<tool_name>",
			"API_description": "<brief_description>",
			"Usage": {{
				"Scenario": "<scenario_description>",
				"Parameters": {{
					"file": ["<file_type>"],
					"payload": {{"<payload_key>": ["<payload_value1>", "<payload_value2>"]}}
				}}
			}}
		}},
	]
}}
6. Ensure that your output is right JSON format and includes all necessary information for each selected tool, including relevant examples.
7. If the input mentions image, audio or video file, please set file_type is "image/audio/video".
8. If tool doesn't have payload, please return []
9. Return all tools can handle the question's problem

Remember:
- Choose only the most relevant tools for the given question.
- Use the exact parameter names and types as specified in the tool descriptions.
- Ensure all required parameters for each tool are included.
- Include relevant usage examples that match the user's query as closely as possible.
- Double-check that the output is in valid JSON format.

Examples of tool usage:
{tool_usage}
"""

if __name__=="__main__":
	from langchain.prompts import (
		ChatPromptTemplate,
		MessagesPlaceholder,
		SystemMessagePromptTemplate,
		HumanMessagePromptTemplate
	)
	from langchain.chains import LLMChain
	from langchain_community.llms import GPT4All, Ollama, HuggingFaceHub
	import re

	# print(prompt_task_decompose)
	chat = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token="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")
	template = "You are a helpful assistant."
	system_message_prompt = SystemMessagePromptTemplate.from_template(template)
	human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_task_decompose)

	template_human = human_message_prompt.prompt.template
	input_variables = human_message_prompt.prompt.input_variables
	chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
	# print(chat_prompt.__dict__)
	chain = LLMChain(llm=chat, prompt=chat_prompt)

	question = "find out person in this image then determine his shirt's color"
	Tool_list = ["'FaceRecognition' returns the information of a person, this tool just be used for identifying person.", "'grounding_dino' returns the position or location of everything in a image, something can be detected is person, car, cat, table, etc", "'skinlesion' returns a predicted disease, specializing in skin lesion disease.", "'ATTT' analyze and identify file, link or connection is benign or malicious", "'pdclothe' returns color of clothes, segment shape of clothes and determine color"]
	result = chain.run(question=question, Tool_list=Tool_list)
	print(result)
	# pattern = r'{(.*)}'
	# result = re.findall(pattern, result.replace("```", "").strip().split('\n\n')[-1], re.DOTALL)
	# result = eval(f'{{{result[0]}}}')
	# print(result)
 