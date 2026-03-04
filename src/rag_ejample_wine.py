
import pandas as pd
df = pd.read_csv('top_rated_wines.csv')
df = df[df['variety'].notna()] # remove any NaN values as it blows up serialization
data = df.sample(700).to_dict('records') # Get only 700 records. More records will make it slower to index


print('El numero de filas es: ', len(data))

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

print('Carga el modelo embbeading')
encoder = SentenceTransformer('all-MiniLM-L6-v2') # Model to create embeddings

# create the vector database client
print('Crea en memoria la instancia Qdrant')
# qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
qdrant = QdrantClient(path="qdrant_data")

print('Crea el db vacío para llenar con la data')
# Create collection to store wines
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize!
print('Crea embbeadings usando la columna notes')
qdrant.upload_points(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) # data is the variable holding all the wines
    ]
)


user_prompt = "Suggest me an amazing Malbec wine from Argentina"

# Search time for awesome wines!
print('Hace la buqueda usando el embbeading del user_promp')
hits = qdrant.search(
    collection_name="top_wines",
    query_vector=encoder.encode(user_prompt).tolist(),
    limit=3
)
for hit in hits:
  print(hit.payload, "score:", hit.score)


  # define a variable to hold the search results
print('Genera lista con las n filas más cercanas') 
search_results = [hit.payload for hit in hits]

# Now time to connect to the local large language model
print('Se conecta con el LLM')
from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

print('Arma el prompt con las instrucciones, la pregunta + contexto encontrado')
completion = client.chat.completions.create(
    model="LLaMA_CPP",
    messages=[
        {"role": "system", "content": "You are chatbot, a wine specialist. Your top priority is to help guide users into selecting amazing wine and guide them with their requests."},
        {"role": "user", "content": "Suggest me an amazing Malbec wine from Argentina"},
        {"role": "assistant", "content": str(search_results)}
    ]
)

print('Respuesta: ')
print(completion.choices[0].message)