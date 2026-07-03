import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load SHL catalog
with open("catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Create Chroma database
client = chromadb.PersistentClient(path="chromadb_data")

collection = client.get_or_create_collection(
    name="shl_assessments"
)

# Add every assessment
for idx, item in enumerate(catalog):

    text = f"""
    Name: {item.get('name','')}
    Description: {item.get('description','')}
    Keys: {' '.join(item.get('keys',[]))}
    Job Levels: {' '.join(item.get('job_levels',[]))}
    """

    embedding = model.encode(text).tolist()

    collection.add(
    ids=[str(idx)],
    embeddings=[embedding],
    documents=[text],
    metadatas=[{
        "name": item.get("name", ""),
        "url": item.get("link", ""),
        "test_type": ", ".join(item.get("keys", []))
    }]

    )

print("Finished creating vector database!")
print("Total assessments:", collection.count())