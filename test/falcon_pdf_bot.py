import argparse

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from text_generation import Client
from langchain.vectorstores import Pinecone
import pinecone
import numpy as np
import pandas as pd

PREPROMPT = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.\n"
PROMPT = """"Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms which are not available in the context.
{context}"""

END_7B = "\n<|prompter|>{query}<|endoftext|><|assistant|>"
END_40B = "\nUser: {query}\nFalcon:"

PARAMETERS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 100,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["<|endoftext|>", "</s>"],
}
CLIENT_7B = Client("http://192.168.6.131:3000")  # Fill this part
CLIENT_40B = Client("https://192.168.6.131:3001")  # Fill this part


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=50)
    return parser.parse_args()


def embed(fname, window_size, step_size):
    # df = pd.read_csv(fname, index_col=0)
    # print(df)
    # print(df["topic"].tolist())
    # exit()
    text = extract_text(fname)
    text = " ".join(text.split())
    # text_tokens = text.split()

    # sentences = []
    # print(len(text_tokens))
    # for i in range(0, len(text_tokens), step_size):
    #     window = text_tokens[i : i + window_size]
    #     if len(window) < window_size:
    #         break
    #     sentences.append(window)

    # paragraphs = [" ".join(s) for s in sentences]
    topics = []
    paragraphs = []
    scripts = [x.split("</SCRIPT>")[0] for x in text.split("<SCRIPT>")[1:]]
    # print(scripts)
    for sc in scripts:
        topic = sc.split("<TOPIC>")[1].split("</TOPIC>")[0]
        p = sc.split("<P>")[1].split("</P>")[0]
        topics.append(topic)
        paragraphs.append(p)
    # df = pd.DataFrame({"topic": topics, "paragraph": paragraphs})
    # df.to_csv("./data_test/test.csv")
    # print(topics)
    # print(paragraphs)
    # exit()
    # paragraphs = text.split("<END>")[:-1]
    print(len(paragraphs))
    model = SentenceTransformer("./weights/all-mpnet-base-v2")
    model.max_seq_length = 512
    cross_encoder = CrossEncoder("./weights/ms-marco-MiniLM-L-6-v2")

    embeddings = model.encode(
        paragraphs,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda"
    )
    # ids_embed = map(str, np.arange(embeddings.shape[0]).tolist())
    # indexx.upsert(vectors=zip(ids_embed,embeddings[:500].cpu().numpy().tolist()), namespace='data')
    # print(embeddings.shape)
    # exit()
    return model, cross_encoder, embeddings, paragraphs, topics


def search(query, model, cross_encoder, embeddings, paragraphs, topics, top_k):
    query_embeddings = model.encode(query, convert_to_tensor=True, device="cuda")
    query_embeddings = query_embeddings.cuda()
    hits = util.semantic_search(
        query_embeddings,
        embeddings,
        top_k=top_k,
    )[0]
    # print(hits)
    cross_input = [[query, paragraphs[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_input)

    for idx in range(len(cross_scores)):
        hits[idx]["cross_score"] = cross_scores[idx]

    results = []
    hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
    print(hits)
    print("-----topic: ", topics[hits[0]["corpus_id"]])
    for hit in hits[:5]:
        results.append(paragraphs[hit["corpus_id"]].replace("\n", " "))
    return results


if __name__ == "__main__":
    # pinecone.init(api_key='0c0d779b-0078-44ff-b04e-4bda0baf1817', environment='gcp-starter')
    # indexx = pinecone.Index("langchain301")
    # pinecone.list_indexes()
    # pico = Pinecone(indexx)
    # print(pico)
    # exit()

    args = parse_args()
    model, cross_encoder, embeddings, paragraphs, topics = embed(
        args.fname,
        args.window_size,
        args.step_size,
    )
    print(embeddings.shape)
    while True:
        print("\n")
        query = input("Enter query: ")
        results = search(
            query,
            model,
            cross_encoder,
            embeddings,
            paragraphs,
            topics,
            top_k=args.top_k,
        )
        
        print(results[0])
        print(len(results))
        exit()
        query_7b = PREPROMPT + PROMPT.format(context="\n".join(results))
        query_7b += END_7B.format(query=query)

        query_40b = PREPROMPT + PROMPT.format(context="\n".join(results))
        query_40b += END_40B.format(query=query)

        text = ""
        for response in CLIENT_7B.generate_stream(query_7b, **PARAMETERS):
            if not response.token.special:
                text += response.token.text

        print("\n***7b response***")
        print(text)

        text = ""
        for response in CLIENT_40B.generate_stream(query_40b, **PARAMETERS):
            if not response.token.special:
                text += response.token.text

        print("\n***40b response***")
        print(text)
