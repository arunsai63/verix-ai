from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = Field(default="VerixAI", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    vector_store_type: str = Field(default="chromadb", env="VECTOR_STORE_TYPE")
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="verixai_documents", env="CHROMA_COLLECTION_NAME")
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    
    # LLM Provider Selection
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")
    embedding_provider: str = Field(default="ollama", env="EMBEDDING_PROVIDER")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="llama3.2", env="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", env="OLLAMA_EMBEDDING_MODEL")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # Claude Configuration
    claude_api_key: Optional[str] = Field(default=None, env="CLAUDE_API_KEY")
    claude_chat_model: str = Field(default="claude-3-opus-20240229", env="CLAUDE_CHAT_MODEL")
    
    # Legacy support
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    
    max_upload_size: int = Field(default=104857600, env="MAX_UPLOAD_SIZE")
    allowed_extensions: List[str] = Field(
        default=["pdf", "docx", "pptx", "html", "txt", "md", "csv", "xlsx"],
        env="ALLOWED_EXTENSIONS"
    )
    
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    upload_directory: str = Field(default="./uploads", env="UPLOAD_DIR")
    datasets_directory: str = Field(default="./datasets", env="DATASETS_DIR")
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")  # Alias for compatibility
    datasets_dir: str = Field(default="./datasets", env="DATASETS_DIR")  # Alias for compatibility

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()