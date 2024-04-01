from base.libs import *
from base.constants import *

class ModelWorker:
    def __init__(self,):
        self.list_file = []
        self.list_collection = []


def submit_file(files, file_checkbox):
    print(file_checkbox)
    file_paths = [os.path.basename(file.name) for file in files]
    print(file_paths)
    worker.list_file += file_paths
    file_paths = worker.list_file
    return gr.CheckboxGroup.update(choices=file_paths, value=file_paths)

def create_collection():
    with gr.Row(elem_id="collection_1", variant="panel") as demo_collection:
        with gr.Column(scale=6):
            gr.CheckboxGroup(choices= ["doc1.pdf","doc2.pdf","doc3.pdf"], label="List file", type="value", visible=True)
        with gr.Column(scale=3):
            gr.File(file_types=['.pdf'], file_count="multiple")
            gr.Button(value="Submit", visible=True, elem_id="btn_submit")
    return demo_collection.update()

# def get_checkbox_value(values):
#     print(values)

def build_demo():
    create_collection_button = gr.Button(value="Create a new collection", visible=True)
    file_checkbox = gr.CheckboxGroup(choices= ["doc1.pdf","doc2.pdf","doc3.pdf"], label="List file", type="value", visible=True)
    file_output = gr.File(file_types=['.pdf'], file_count="multiple")
    # submit_button = gr.UploadButton("Click to Upload a File", file_types=[".pdf"], file_count="multiple")
    submit_button = gr.Button(value="Submit", visible=True)
    # with gr.Row(elem_id="collection_1", variant="panel") as demo_collection:
    #     with gr.Column(scale=6):
    #         gr.CheckboxGroup(choices= ["doc1.pdf","doc2.pdf","doc3.pdf"], label="List file", type="value", visible=True)
    #     with gr.Column(scale=3):
    #         gr.File(file_types=['.pdf'], file_count="multiple")
    #         gr.Button(value="Submit", visible=True, elem_id="btn_submit")
            # submit_button.click(submit_file, [file_output, file_checkbox], [file_checkbox])
    result_row = gr.Row(elem_id="collection_1", variant="panel")
    with result_row:
        with gr.Column(scale=6):
            gr.CheckboxGroup(choices= ["doc1.pdf","doc2.pdf","doc3.pdf"], label="List file", type="value", visible=True)
        with gr.Column(scale=3):
            gr.File(file_types=['.pdf'], file_count="multiple")
            gr.Button(value="Submit", visible=True, elem_id="btn_submit")

    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # Knowledge Storage
        Create your own collection and add files into that.
        """)
        result_row.render()
        create_collection_button.render()
        create_collection_button.click(create_collection, [], [result_row])
        # demo_collection.render()
        # print(demo_collection)
        # with gr.Row(elem_id="collection_1", variant="panel") as demo_collection:
        #     with gr.Column(scale=6):
        #         file_checkbox.render()
        #     with gr.Column(scale=3):
        #         file_output.render()
        #         submit_button.render()
        #         submit_button.click(submit_file, [file_output, file_checkbox], [file_checkbox])

                # file_checkbox.update(choices=list_file, list_file, [file_checkbox])
                # file_checkbox.change(get_checkbox_value, file_checkbox)
    return demo

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8887
    share = False
    #config queue gradio
    api_open = False
    max_size = 100

    worker = ModelWorker()
    demo = build_demo()
    demo.queue(
        api_open = api_open,
        max_size = max_size
    ).launch(
        server_name = host,
        server_port = port,
        share = share
    )
    # demo.launch(server_name = host,server_port = port,share = share)