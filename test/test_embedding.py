# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("./weights/all-MiniLM-L6-v2")
# texts = [
# 		"The canine barked loudly.",
# 		"The dog made a noisy bark.",
# 		"He ate a lot of pizza.",
# 		"He devoured a large quantity of pizza pie.",
# ]

# text_embeddings = model.encode(texts)

# print(type(text_embeddings))
# print(text_embeddings.shape)


# # from cosine_similarity import compute_cosine_similarity
# from scipy import spatial
# text_embeddings_dict = dict(zip(texts, list(text_embeddings)))
# dog_text_1 = "The canine barked loudly."
# dog_text_2 = "The dog made a noisy bark."
# # print(compute_cosine_similarity(text_embeddings_dict[dog_text_1], text_embeddings_dict[dog_text_2]))
# result = 1 - spatial.distance.cosine(text_embeddings_dict[dog_text_1], text_embeddings_dict[dog_text_2])
# print(result)

documents = [
	"The latest iPhone model comes with impressive features and a powerful camera.",
	"Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
	"Einstein's theory of relativity revolutionized our understanding of space and time.",
	"Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
	"The American Revolution had a profound impact on the birth of the United States as a nation.",
	"Regular exercise and a balanced diet are essential for maintaining good physical health.",
	"Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
	"Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
	"Startup companies often face challenges in securing funding and scaling their operations.",
	"Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
]
genres = [
	"technology",
	"travel",
	"science",
	"food",
	"history",
	"fitness",
	"art",
	"climate change",
	"business",
	"music",
]
documents1 = ["Mẫu iPhone mới nhất có các tính năng ấn tượng và camera mạnh mẽ.",
"Khám phá những bãi biển đẹp và nền văn hóa sôi động của Bali là ước mơ của nhiều du khách.",
"Thuyết tương đối của Einstein đã cách mạng hóa sự hiểu biết của chúng ta về không gian và thời gian.",
"Bánh pizza truyền thống của Ý nổi tiếng với lớp vỏ mỏng, nguyên liệu tươi và lò nướng đốt củi.",
"Cách mạng Hoa Kỳ có tác động sâu sắc đến sự ra đời của Hoa Kỳ với tư cách là một quốc gia.",
"Tập thể dục thường xuyên và chế độ ăn uống cân bằng là điều cần thiết để duy trì sức khỏe thể chất tốt.",
"Bức tranh Mona Lisa của Leonardo da Vinci được coi là một trong những bức tranh mang tính biểu tượng nhất trong lịch sử nghệ thuật.",
"Biến đổi khí hậu đặt ra mối đe dọa đáng kể đối với hệ sinh thái và đa dạng sinh học của hành tinh.",
"Các công ty khởi nghiệp thường phải đối mặt với những thách thức trong việc đảm bảo nguồn vốn và mở rộng quy mô hoạt động của mình.",
"Bản giao hưởng số 9 của Beethoven được tôn vinh nhờ phần kết hợp hợp xướng mạnh mẽ, 'Ode to Joy.'",
]
genres1 = [
"công nghệ",
"du lịch",
"khoa học",
"đồ ăn",
"lịch sử",
"sự thích hợp",
"nghệ thuật",
"khí hậu thay đổi",
"việc kinh doanh",
"âm nhạc",
]

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "./database/MQ_data/"
EMBED_MODEL = "./weights/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "demo_docs1"
#-----------------------------chromaDB------------------------------------
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH, tenant="default_tenant", database="database1")

# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# collection = client.create_collection(
# 	name=COLLECTION_NAME,
# 	embedding_function=embedding_func,
# 	metadata={"hnsw:space": "cosine"},
# )

# collection.add(
#     documents=documents,
#     ids=[f"id{i}" for i in range(len(documents))],
#     metadatas=[{"genre": g} for g in genres]
# )

# collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# query_results = collection.query(
#     query_texts=["Find me some delicious food!"],
#     n_results=2,
# )

# print(query_results.keys())
# print(query_results)

#//////////////////////////////////////////////////////////////////////
#-------------------------chromadb http-----------------------------------------------
# chroma_client = chromadb.HttpClient(host='localhost', port=8008)
# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
# print(chroma_client.list_collections()[1])
# chroma_client.delete_collection(name=COLLECTION_NAME)
# collection = chroma_client.create_collection(
# 	name=COLLECTION_NAME,
# 	embedding_function=embedding_func,
# 	metadata={"hnsw:space": "cosine"},
# )

# collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# collection.add(
#     documents=documents1,
#     ids=[f"id{i}_0" for i in range(len(documents1))],
#     metadatas=[{"genre": g} for g in genres1]
# )

# print(collection.get())
# print(collection.count())
# query_results = collection.query(
#     query_texts=["Tìm cho tôi một số đồ ăn ngon!"],
#     n_results=9,
# )
# print(query_results)
# genre = [data['genre'] for data in query_results['metadatas'][0]]
# print(genre)

# print(collection.get(where={"genre": {"$eq": "music"}}))
# collection.delete(where={"genre": {"$eq": "music"}})
# print(collection.get())
#////////////////////////////////////////////////////////////////////////////////////

# def build_chroma_collection(
# 	chroma_path: pathlib.Path,
# 	collection_name: str,
# 	embedding_func_name: str,
# 	ids: list[str],
# 	documents: list[str],
# 	metadatas: list[dict],
# 	distance_func_name: str = "cosine",
# ):
# 	"""Create a ChromaDB collection"""

# 	chroma_client = chromadb.PersistentClient(chroma_path)

# 	embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
# 		model_name=embedding_func_name
# 	)

# 	collection = chroma_client.create_collection(
# 		name=collection_name,
# 		embedding_function=embedding_func,
# 		metadata={"hnsw:space": distance_func_name},
# 	)

# 	document_indices = list(range(len(documents)))

# 	for batch in batched(document_indices, 166):
# 		start_idx = batch[0]
# 		end_idx = batch[-1]

# 		collection.add(
# 			ids=ids[start_idx:end_idx],
# 			documents=documents[start_idx:end_idx],
# 			metadatas=metadatas[start_idx:end_idx],
# 		)


#----------------------------test nomic--------------------
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# client.delete_collection(name=COLLECTION_NAME)
# exit()

# model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
# kwargs = {"model_name":EMBED_MODEL, "device":"cuda", "normalize_embeddings":False, "trust_remote_code":True}
model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cuda", trust_remote_code=True)
# embeddings = model.encode(documents1)
# # embeddings = model(documents1)
# print(len(embeddings[0]))

collection = client.create_collection(
	name=COLLECTION_NAME,
	embedding_function=model,
	metadata={"hnsw:space": "cosine"},
)

collection = client.get_collection(name=COLLECTION_NAME, embedding_function=model)

collection.add(
    documents=documents1,
    ids=[f"id{i}" for i in range(len(documents1))],
    metadatas=[{"genre": g} for g in genres1]
)

query_results = collection.query(
    query_texts=["Tìm cho tôi một số đồ ăn ngon!", "Tìm cho tôi một nơi để đi", "Các viện bảo tàng muốn có thứ gì"],
    n_results=2,
)
print(query_results)