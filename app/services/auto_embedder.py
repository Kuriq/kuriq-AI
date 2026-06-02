"""
임베딩 자동 감지 및 생성 모듈
ChromaDB 에 저장된 임베딩 차원을 감지하여 자동으로 맞는 모델 사용
"""
import logging
from typing import Literal

logger = logging.getLogger(__name__)

EmbeddingBackend = Literal["sentence_transformers", "openai"]

# 감지된 설정 (런타임에 결정)
_detected_dimension: int | None = None
_backend: EmbeddingBackend | None = None
_sentence_model = None
_openai_client = None


def detect_backend(dimension: int) -> EmbeddingBackend:
    """임베딩 차원에 맞는 백엔드 결정"""
    if dimension == 384:
        return "sentence_transformers"
    elif dimension == 1536:
        return "openai"
    else:
        logger.warning(f"알 수 없는 임베딩 차원: {dimension}, sentence-transformers 사용")
        return "sentence_transformers"


def get_backend() -> EmbeddingBackend:
    """백엔드 자동 감지 (싱글톤)"""
    global _backend, _detected_dimension
    
    if _backend is None:
        from app.core.chroma import detect_embedding_dimension
        _detected_dimension = detect_embedding_dimension()
        _backend = detect_backend(_detected_dimension)
        logger.info(f"[임베딩] ChromaDB 임베딩 차원 감지: {_detected_dimension} → {_backend} 사용")
    
    return _backend


def get_sentence_model():
    """sentence-transformers 모델 로드 (싱글톤)"""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("[임베딩] sentence-transformers 모델 로드 완료")
    return _sentence_model


def get_openai_client():
    """OpenAI 클라이언트 로드 (싱글톤)"""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        from app.core.config import settings
        _openai_client = OpenAI(api_key=settings.openai_api_key)
        logger.info("[임베딩] OpenAI 클라이언트 초기화 완료")
    return _openai_client


def embed_text(text: str) -> list[float]:
    """텍스트 임베딩 생성 (자동 백엔드 선택)"""
    backend = get_backend()
    
    logger.debug(f"[auto_embedder] embed_text 호출 — backend={backend}, text={text[:50]}...")
    
    if backend == "sentence_transformers":
        model = get_sentence_model()
        embedding = model.encode(text, convert_to_numpy=True)
        # 1D 배열로 평탄화 후 Python 리스트로 변환 (ChromaDB 호환)
        result = embedding.flatten().tolist()
        logger.debug(f"[auto_embedder] 임베딩 생성 완료 — type={type(result)}, dim={len(result)}")
        return result
    else:  # openai
        client = get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        result = response.data[0].embedding
        logger.debug(f"[auto_embedder] 임베딩 생성 완료 — type={type(result)}, dim={len(result)}")
        return result


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """배치 임베딩 생성 (자동 백엔드 선택)"""
    backend = get_backend()
    
    if backend == "sentence_transformers":
        model = get_sentence_model()
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        # 2D 배열을 리스트 of 리스트로 변환 (ChromaDB 호환)
        return [emb.flatten().tolist() for emb in embeddings]
    else:  # openai
        client = get_openai_client()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            all_embeddings.extend([e.embedding for e in response.data])
        return all_embeddings
