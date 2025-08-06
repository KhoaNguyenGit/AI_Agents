from sentence_transformers import SentenceTransformer
import json
import os
from config import features_path, Embedded_Model_Path

embedded_model_path = Embedded_Model_Path
feature_path = features_path
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 

with open(feature_path, "r") as f:
    features = json.load(f)

for feature in features:
    embedding = model.encode(feature["description"])
    feature["embedding"] = embedding.tolist()

with open(embedded_model_path, "w") as f_out:
    json.dump(features, f_out, indent=2)

print("Embedded features saved to features_with_embeddings.json")