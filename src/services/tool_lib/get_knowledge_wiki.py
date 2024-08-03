def get_knowledge_wiki(query):
	import wikipedia
	try:
		result = wikipedia.summary(query)
	except:
		result = "Wkipedia do not have information about the question."
	return result
	
