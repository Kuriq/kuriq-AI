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


def detect_embedding_dimension() -> int:
    """ChromaDB 에 저장된 임베딩 차원 감지"""
    collection = get_collection()
    
    # 샘플 ID 가져오기
    sample = collection.get(include=["embeddings"], limit=1)
    
    # ChromaDB 는 embeddings 를 리스트로 반환하지만, NumPy 배열일 수 있음
    embeddings = sample.get("embeddings", [])
    
    # embeddings 가 비어있지 않은지 확인 (NumPy 배열 호환)
    if embeddings is not None and len(embeddings) > 0:
        # 첫 번째 임베딩의 차원 반환
        first_embedding = embeddings[0]
        # NumPy 배열이면 shape 사용, 리스트면 len 사용
        if hasattr(first_embedding, 'shape'):
            return int(first_embedding.shape[0])
        return len(first_embedding)
    
    # 데이터가 없으면 기본값 384 (sentence-transformers)
    return 384
