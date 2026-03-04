
###################################
## Este codigo ya toma la base con los embbedings generada en el codigo "make_index"
## Realiza la parte de búsqueda usando el embbeding de la pregunta y los que están en el indice
## Con lo que debuelve la búsquyeda + la pregunta + instrucciones genera el promp y se lo pasa al modelo de lenguaje
## El modelo responde usando la info del contexto  


from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

print('Trae el indice')
qdrant = QdrantClient(path="qdrant_data")

print('Carga el modelo embbeading')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

print('Escribe la pregunta')
user_prompt = "Hay algún vino de uruguay?"

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
    base_url="http://127.0.0.1:1234/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

print("Arma el prompt (pregunta + contexto encontrado)")
context = "\n".join([str(r) for r in search_results])

print('El contexto es: ', context)

print('Tira el prompt con pregunta + contexto')

completion = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct-v0.3",
    messages=[
        {"role": "user", "content": 
            "You are a wine specialist. Use the context below to answer.\n\n"
            f"Question: {user_prompt}\n\n"
            f"Context:\n{context}"
        }
        ],
)

print('Respuesta: ')
print(completion.choices[0].message)