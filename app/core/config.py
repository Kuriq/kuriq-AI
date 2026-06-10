from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    llm_timeout: int = 30
    internal_secret_key: str
    rag_top_k: int = 20
    chroma_mode: str = "server"
    chroma_path: str = "./chroma_db"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    crawler_data_base_url: str = "http://localhost:8001"
    crawler_data_timeout: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
