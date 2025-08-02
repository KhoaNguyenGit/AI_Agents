from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import json

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "features"

def upsert_features(json_path: str):
    with open(json_path, "r") as f:
        features = json.load(f)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance="Cosine")
    )

    points = [
        PointStruct(
            id=feature["id"],
            vector=feature["embedding"],
            payload={
                "name": feature["name"],
                "description": feature["description"],
                "category": feature["category"],
                "tags": feature["tags"],
                "channels_supported": feature["channels_supported"],
                "industries": feature["industries"]
            }
        )
        for feature in features
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Uploaded {len(points)} feature vectors to Qdrant.")

def check_features_collection():
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print("Collection info:")
        print(collection_info)
    except Exception as e:
        print("Failed to retrieve collection info:", e)

def delete_features_collection():
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print("Failed to delete collection:", e)


# upsert_features("knowledge_base_data/features_with_embeddings.json")
check_features_collection()
# delete_features_collection()