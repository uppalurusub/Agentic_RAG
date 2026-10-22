import chromadb
from app.core.llm_config import VECTOR_DB_PATH

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

print("Database path:", VECTOR_DB_PATH)

collections = client.list_collections()
print("Collections found:", len(collections))

for collection in collections:
    print(
        "Name:", collection.name,
        "| Count:", collection.count(),
    )