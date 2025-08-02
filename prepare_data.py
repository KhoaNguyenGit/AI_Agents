from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
import os

load_dotenv()
embedded_model_path = os.getenv("Embedded_Model_Path")
feature_path = os.getenv("features_path")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 

with open(feature_path, "r") as f:
    features = json.load(f)

for feature in features:
    embedding = model.encode(feature["description"])
    feature["embedding"] = embedding.tolist()

with open("knowledge_base_data/features_with_embeddings.json", "w") as f_out:
    json.dump(features, f_out, indent=2)

print("Embedded features saved to features_with_embeddings.json")