import chromadb
from app.core.config import settings

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        if settings.chroma_mode.lower() == "server":
            _client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        else:
            _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection("kuriq_courses")
    return _collection
