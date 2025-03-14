# @Time: 2025/3/13 14:47
# @Author: lvjing
from typing import Dict

from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.openai import OpenAI as LLamaIndexOpenAI
from openai import OpenAI

from .config import Configuration

configuration = Configuration()
# -------------------------LLM Settings  Start-------------------------
#  需要支持新的模型参考如下格式添加即可
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS

QWEN_MODELS_V3: Dict[str, int] = {
    "deepseek-v3": 64000,
}
ALL_AVAILABLE_MODELS.update(QWEN_MODELS_V3)
CHAT_MODELS.update(QWEN_MODELS_V3)


# -------------------------LLM Settings  End-------------------------

def deepseek_llm(**kwargs):
    return DeepSeek(model=configuration.deepseek_model_name,
                    api_key=configuration.deepseek_api_key,
                    api_base=configuration.deepseek_api_base, **kwargs)


def llama_index_llm(**kwargs):
    return LLamaIndexOpenAI(model=configuration.llm_model_name,
                            api_key=configuration.llm_api_key,
                            api_base=configuration.llm_api_base, **kwargs)


def moonshot_llm(**kwargs):
    return OpenAI(api_key=configuration.moonshot_api_key,
                  base_url="https://api.moonshot.cn/v1", **kwargs)


def openai_llm(**kwargs):
    return OpenAI(api_key=configuration.llm_api_key,
                  base_url=configuration.llm_api_base, **kwargs)


# ------------------------Embedding Settings------------------------

# 本地嵌入模型
def local_bge_small_embed_model(**kwargs):
    # 终端执行： pip install llama-index-embeddings-huggingface llama-index-embeddings-instructor
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name=configuration.local_embedding_model_name,
                                       **kwargs)
    return embed_model


# 在线嵌入模型
def ollama_nomic_embed_model(**kwargs):
    # 终端执行： pip install llama-index-embeddings-ollama
    from llama_index.embeddings.ollama import OllamaEmbedding

    ollama_embedding = OllamaEmbedding(
        model_name=configuration.remote_embedding_model_name,
        base_url=configuration.remote_embedding_model_url,
        **kwargs
    )
    return ollama_embedding


# -------------------------Setting Default LLM Start------------------------
from llama_index.core import Settings

Settings.embed_model = local_bge_small_embed_model()
Settings.llm = deepseek_llm()

# -------------------------Setting Default LLM End------------------------
