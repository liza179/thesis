from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_model(static=False):
    if static:
        tokenizer = Tokenizer.from_pretrained("BAAI/bge-m3")
        static_embedding = StaticEmbedding(tokenizer, embedding_dim=512)
        return SentenceTransformer(modules=[static_embedding], device="cpu")
    else:
        return SentenceTransformer("BAAI/bge-m3")

def encode(text, model, max_length=8192):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=int(max_length*0.05))
    chunks = splitter.split_text(text)
    embeddings = []
    chunk_weights = []
    for chunk in chunks:
        embeddings.append(model.encode(chunk))
        chunk_weights.append(len(chunk))
    
    embeddings = np.stack(embeddings, axis=0)
    weights = np.array(chunk_weights)
    weights = weights / weights.sum()
    return np.average(embeddings, axis=0, weights=weights)

input_file = "parsed/bol0200.txt"
text = open(input_file).read()
model = get_model(static=True)
embeddings = encode(text, model)
print(embeddings.shape)