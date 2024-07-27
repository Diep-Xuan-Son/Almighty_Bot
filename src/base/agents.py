
from langchain.schema import HumanMessage, SystemMessage
import re as regex
from string import Formatter

class Agent():
    def __init__(self, system_prompt, prompt, llm):
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.llm = llm
        
    def run(self, **kwargs):
        input_variables = {
            v for _, v, _, _ in Formatter().parse(self.prompt) if v is not None
        }
        for i in input_variables:
            assert i in list(kwargs.keys()), f"Don't have argument {i}"
        
        self.prompt = self.prompt.format(**kwargs)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.prompt)
        ]
        response = self.llm(messages)
        return response.content
    
    def clean_response_1(self, response=""):
        pattern = r'{.*}'
        clean_answer = regex.findall(pattern, response.replace("```", "").strip(), regex.DOTALL)
        if isinstance(clean_answer, list):
            clean_answer = clean_answer[0]
            
        clean_answer = clean_answer.split("\n\n")
        results = {}
        for re in clean_answer:  
            try:
                results = eval(re)
            except:
                re = "[" + re + "]"
                re = eval(re)
                for r in re:
                    value = r["Tools"]
                    if not results:
                        results["Tools"] = []
                    if isinstance(value, list):   
                        results["Tools"] += value
                    elif isinstance(value, dict):
                        results["Tools"].append(value)
        return results["Tools"]
    
    def clean_response_2(self, response=""):
        pattern = r'{(.*)}'
        result = regex.findall(pattern, response.replace("```", "").strip().split('\n\n')[-1], regex.DOTALL)
        # result = eval(f'{{{result[0]}}}')
        result = eval(result)
        return result
    
    def clean_response_3(self, response=""):
        pattern = r'{.*}'
        clean_answer = regex.findall(pattern, response.replace("```", "").strip(), regex.DOTALL)
        if isinstance(clean_answer, list):
            clean_answer = clean_answer[0]
            
        clean_answer = clean_answer.split("\n\n")
        results = []
        for re in clean_answer:  
            try:
                results.append(eval(re))
            except:
                re = "[" + re + "]"
                re = eval(re)
                for r in re:
                    results.append(r)
        return results