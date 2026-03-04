
#######################
## La idea de este código es cargar los datos, generar el diccionario como records,
## Genera la base de datos "qdrant" en ath="qdrant_data"
## Luego cargo cargo la data en qdrant, con los embbeadings el ID y el texto


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
