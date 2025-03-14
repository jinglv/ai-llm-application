# @Time: 2025/3/13 14:42
# @Author: lvjing
import os

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """
    环境变量配置读取
    """
    # DeepSeek配置
    deepseek_api_key: str = Field(default=os.getenv("DEEPSEEK_API_KEY"), description="Deepseek API key")
    deepseek_api_base: str = Field(default=os.getenv("DEEPSEEK_API_BASE"), description="Deepseek API base")
    deepseek_model_name: str = Field(default=os.getenv("DEEPSEEK_MODEL_NAME"), description="Deepseek model name")
    # LLM配置
    llm_api_key: str = Field(default=os.getenv("LLM_API_KEY"), description="LLM API key")
    llm_api_base: str = Field(default=os.getenv("LLM_API_BASE"), description="LLM API base")
    llm_model_name: str = Field(default=os.getenv("LLM_MODEL_NAME"), description="LLM model name")
    # 本地嵌入模型Embedding
    local_embedding_model_name: str = Field(default=os.getenv("LOCAL_EMBEDDING_MODEL_NAME"),
                                            description="Local embedding model name")
    # 远程嵌入模型Embedding
    remote_embedding_model_name: str = Field(default=os.getenv("REMOTE_EMBEDDING_MODEL_NAME"),
                                             description="Remote embedding model name")
    remote_embedding_model_url: str = Field(default=os.getenv("REMOTE_EMBEDDING_MODEL_URL"),
                                            description="Remote embedding model URL")
    # 向量化数据库的尺寸大小，跟使用embedding模型相关，查询模型最大的尺寸
    embedding_model_dim: int = Field(default=512, description="Embedding model dimension")
    # 月之暗面LLM配置
    moonshot_api_key: str = Field(default=os.getenv("MOONSHOT_API_KEY"), description="Moonshot API key")
    # Milvus配置
    milvus_uri: str = Field(default=os.getenv("MILVUS_URI"), description="Milvus URI")
