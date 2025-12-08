from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import hashlib
from pathlib import Path

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


def _text_hash(text: str) -> str:
    """Generate a hash for text content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def encode_with_cache(text, model, cache_file='.embedding_cache.pkl', max_length=8192):
    """
    Encode text with persistent disk caching.

    Args:
        text: Text to embed
        model: Sentence transformer model
        cache_file: Path to cache file (None to disable caching)
        max_length: Max chunk size for encoding

    Returns:
        Embedding vector (numpy array)
    """
    # If no cache, just encode directly
    if cache_file is None:
        return encode(text, model, max_length=max_length)

    # Load cache
    cache_path = Path(cache_file)
    cache = {}
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

    # Check if cached
    text_key = _text_hash(text)
    if text_key in cache:
        return cache[text_key]

    # Encode and cache
    emb = encode(text, model, max_length=max_length)
    cache[text_key] = emb

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    return emb


if __name__ == "__main__":
    input_file = "parsed/bol0200.txt"
    text = open(input_file).read()
    model = get_model(static=True)
    embeddings = encode(text, model)
    print(embeddings.shape)