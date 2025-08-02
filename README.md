# 🧠 AI Feature Recommender using Qwen 2.5-3B + Qdrant

This project leverages the **Qwen 2.5-3B Instruct** language model and **Qdrant vector search engine** to match customer pain points with the most relevant features from a knowledge base.

---

## 📁 Project Structure

```
├── knowledge_base_data/
│   └── features.json                 # Clean feature descriptions with metadata
├── model/Qwen2.5-3B-Instruct/        # Local model files for Qwen 2.5-3B
│   ├── config.json, tokenizer.json, ...
│   └── *.safetensors                 # Model weights
├── features_with_embeddings.json     # Feature vectors (output of prepare_data.py)
├── main.py                           # Main entry point to query the LLM with user input
├── prepare_data.py                   # Converts features into vector embeddings and stores them
├── upload_data.py                    # Uploads vectors to Qdrant collection
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables
```

---

## 🚀 How It Works

1. **Prepare features**  
   `prepare_data.py` encodes feature descriptions into embeddings using the same model family as your LLM.

2. **Upload to Qdrant**  
   `upload_data.py` sends these embeddings into a Qdrant collection (`features`), allowing fast similarity search.

3. **Query via LLM**  
   `main.py`:
   - Takes a user pain point as input
   - Uses vector search to find the top matching features
   - Prompts the Qwen 2.5-3B model to select and explain the best match

---

## 🛠️ Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download model weights**
   - mkdir model
   - cd model/
   - Place the Qwen 2.5-3B-Instruct model files into the `model/Qwen2.5-3B-Instruct/` directory.
   - Ensure files include `.safetensors`, `config.json`, `tokenizer.json`, etc.

3. **Start Qdrant (optional if using local setup)**

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

---

## 📦 Example Usage

```bash
# Step 1: Prepare data
python prepare_data.py

# Step 2: Upload to Qdrant
python upload_data.py

# Step 3: Run main agent
python main.py
```

---

## 📚 Sample Feature JSON Format

```json
{
  "feature_name": "Automated Post-Purchase Surveys",
  "how_it_helps": "Collects timely customer feedback directly after a transaction.",
  "product_category": "Customer Feedback Collection"
}
```

---

## 🤖 Model Info

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Model**: Qwen 2.5-3B Instruct
- **Framework**: HuggingFace Transformers
- **Backend**: CUDA / CPU fallback
- **Embedding**: Could be same as LLM or from sentence-transformers

---

## 📬 Contact

Built by [Your Name].  
For questions, contact: your.email@example.com