import os, sklearn
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
print("scikit-learn:", sklearn.__version__)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding length:", len(model.encode("hello world")))
print("Anthropic key found:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("OpenAI key found:", bool(os.getenv("OPENAI_API_KEY")))
