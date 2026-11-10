import chromadb
from app.config.settings import settings 
#VECTOR_DB_PATH

client = chromadb.PersistentClient(path=settings.vector_db_path)

print("Database path:", settings.vector_db_path)

collections = client.list_collections()
print("Collections found:", len(collections))

for collection in collections:
    print(
        "Name:", collection.name,
        "| Count:", collection.count(),
    )