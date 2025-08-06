from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import json
from config import Embedded_Model_Path, LLM_model_path

embedding_model_path = Embedded_Model_Path
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 
LLM_model_path = LLM_model_path


user_input = {
                "user_id": "client-001",
                "pain_point": "We're struggling to collect customer feedback consistently after a purchase.",
                # Optional filters
                # "industry": "retail",
                # "preferred_channels": ["pos", "email"]
            }

embedding = embedding_model.encode(user_input["pain_point"])

client = QdrantClient(host="localhost", port=6333)

must_conditions = []

if user_input.get("industry"):
    must_conditions.append(
        FieldCondition(key="industries", match=MatchValue(value=user_input["industry"]))
    )

if user_input.get("preferred_channels"):
    for ch in user_input["preferred_channels"]:
        must_conditions.append(
            FieldCondition(key="channels_supported", match=MatchValue(value=ch))
        )

search_result = client.search(
    collection_name="features",
    query_vector=embedding.tolist(),
    limit=5,
    query_filter=Filter(must=must_conditions) if must_conditions else None
)

threshold = 0
filtered_features = [
    res.payload | {"_score": res.score}
    for res in search_result if res.score >= threshold
]

top_features = filtered_features[:3]
total_features = len(top_features)

print(f"Total matched features: {total_features}")

if not top_features:
    print("No relevant features found with similarity score above 0.5.")
    prompt = None
else:
    feature_descriptions = "\n".join([
        f"{i+1}. {f['name']} - {f['description']} (Score: {f['_score']:.2f})"
        for i, f in enumerate(top_features)
    ])
    
    print(feature_descriptions)

    prompt = f"""Given the user pain point and a list of matched features, return only the single most relevant feature as a JSON object.

            Respond with ONLY a valid JSON object in this exact format — no explanation, no prefix, nothing else:

            {{
            "feature_name": "...",
            "how_it_helps": "...",
            "product_category": "...",
            "relevance_score": float
            }}

            User pain point:
            {user_input['pain_point']}

            Matched features:
            {feature_descriptions}
            """
    
    # prompt="Who are you ?"
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(LLM_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        temperature=0
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(response)
    
    response_dict = json.loads(response)

    combined = f"Potential Filum.ai Solution: {response_dict['feature_name']} - How it help: {response_dict['how_it_helps']}"
    print(combined)