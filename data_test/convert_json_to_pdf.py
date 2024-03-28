from fpdf import FPDF
import json 

pdf = FPDF()
pdf.set_font("Arial", size = 14)

f = open("./data_test/llava-plus-v1-117k-tool-merge.json")
data = json.load(f)

# print(data[0])
dt = data[0]
# for i, dt in enumerate(data[2966:2967]):
for i, dt in enumerate(data[:100000:100]):
    print(f"----Processing: {i+1}/{len(data)}")
    if len(dt['conversations'][1]['actions']) == 0:
        topic = 'QnA'
        answer = dt['conversations'][1]['value']
    else:
    # print(dt['conversations'])
        topic = dt['conversations'][1]['actions'][0]['API_name']
        answer = dt['conversations'][3]['value']
    question = dt['conversations'][0]['value']

    p = question + "\n" + answer
    topic_txt = f"<TOPIC>{topic}</TOPIC>"
    p_txt = f"<P>{p}</P>"
    # exit()
    # Add a page
    pdf.add_page()
    # create a cell
    pdf.multi_cell(200, 10, txt = "<SCRIPT>", 
            border = 0, align = 'L')
    pdf.multi_cell(200, 10, txt = topic_txt,
            border = 0, align = 'L')
    pdf.multi_cell(200, 10, txt = p_txt.encode().decode('latin-1'),
            border = 0, align = 'L')
    pdf.multi_cell(200, 10, txt = "</SCRIPT>", 
            border = 0, align = 'L')
# save the pdf with name .pdf
pdf.output("test1.pdf") 
