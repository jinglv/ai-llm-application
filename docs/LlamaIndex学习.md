# LlamaIndex学习

LlamaIndex官方文档：https://docs.llamaindex.ai/en/stable/



## 1.什么是LlamaIndex

LlamaIndex是一个为基于大型语言模型（LLM）的应用提供数据摄取、结构化和访问私有或领域特定数据的数据框架。它的核心目标是简化和优化LLM在外部数据源中的查询过程，使得LLM在处理大规模数据集时更加高效和智能。LlamaIndex提供了以下工具：

1. **数据连接器（Data connectors）**：能够从原始来源和格式中摄取现有数据，这些数据源可能是APIs、PDFs、SQL数据库等。
   - LlamaIndex数据源
     - 数据库
     - 文件（pdf\ppt\word\...）
   - LlamaIndex提供读取文件的方法
     - SimpleDirectoryReader: https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader/
       - 可根据不同文件类型选择数据连接器，不需要传指定的数据格式
     - DoclingReader:https://docs.llamaindex.ai/en/stable/examples/data_connectors/DoclingReaderDemo/
       - docling的官方文档：https://github.com/docling-project/docling
2. **数据索引（Data indexes）**：将数据结构化为LLM易于使用和性能良好的中间表示形式。
3. **引擎（Engines）**：提供对数据的自然语言访问。例如，查询引擎是强大的检索接口，用于知识增强的输出；聊天引擎是与数据进行多消息、“来回”交互的对话接口。
4. **数据代理（Data agents）**：由LLM驱动的知识工作者，通过工具增强，从简单的辅助功能到API集成等。
5. **应用集成（Application integrations）**：将LlamaIndex重新整合回你的生态系统中，这可以是LangChain、Flask、Docker、ChatGPT等。

LlamaIndex特别适用于需要处理大量文档、数据库或非结构化数据的场景，例如企业内部知识库、智能客服系统等。它还支持上下文增强（context augmentation），这是一种使LLM能够访问私有数据以解决手头问题的方法。LlamaIndex提供了构建任何上下文增强用例的工具，从原型到生产，允许你摄取、解析、索引和处理数据，并快速实现结合数据访问与LLM提示的复杂查询工作流。



##  2. 安装需要的依赖包

```
pip install llama-index
pip install llama-index-embeddings-instructor
pip install llama-index-embeddings-huggingface
pip install torch sentence-transformers
```



## 3. 导入依赖包

```python
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,Settings,get_response_synthesizer,StorageContext,load_index_from_storage,Document
from typing import Dict
from llama_index.llms.openai import OpenAI as DeepSeeK
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
```



## 4. 设置LLM

```python
DEEPSEEK_MODELS: Dict[str, int] = {
    "deepseek-chat": 128000,
}
ALL_AVAILABLE_MODELS.update(DEEPSEEK_MODELS)
CHAT_MODELS.update(DEEPSEEK_MODELS)

llm = DeepSeeK(api_key="sk-xxxxxxxxxx",
                 model="deepseek-chat",
                 api_base="https://api.deepseek.com/v1",
                 temperature=0.5)
Settings.llm = llm
```



## 5. 设置嵌入模型

如何选择合适的 Embedding 模型:

-  https://mp.weixin.qq.com/s/ihGgXmm8pMAp-j2v8NhqbQ
- https://huggingface.co/spaces/mteb/leaderboard

注意：选择Embedding模型一次处理Token的最大数值



**bge-small-zh-v1.5模型**

默认512维空间

​	- 单次处理的最大Token是512

```python
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model
```



**bge-large-zh-v1.5模型**

```python
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
Settings.embed_model = embed_model
```



**bce-embedding-base_v1模型**：中英双语，及其跨语种embedding模型for RAG

- https://www.modelscope.cn/models/maidalun/bce-embedding-base_v1



## 6. 什么是Document？

Document是一个容器，用来保存来自各种来源的数据，比如PDF、API 输出或从数据库中检索到的数据。

### 加载单个pdf文档

```python
documents = SimpleDirectoryReader(input_files=["data/SQL题库.pdf"]).load_data() # SQL题库.pdf有3页
print(len(documents)) # 输出3
print(documents)
```

PDF默认加载文件一页是一个Document对象



### 加载单个文本文档

```python
documents = SimpleDirectoryReader(input_files=["data/围城.txt"]).load_data()
print(len(documents))
print(documents)
```

记事本txt文档处理，不管文档由多长，默认都只处理加载为一个Document对象



### 加载多个文本文档

```python
documents = SimpleDirectoryReader(input_dir="data", required_exts=[".txt"]).load_data()
print(len(documents))
print(documents)
```

- 指定目录下文件格式



```python
documents = SimpleDirectoryReader(input_dir="data").load_data()
print(len(documents))
print(documents)
```

- 加载指定目录下所有文件



## 7. LLmaindex中的Node是什么？

LlamaIndex 中的 Node 对象表示源文档的“块”或部分。 可能是一个文本块、一幅图像或其他类型的数据。类似于 Documents，Nodes 也包含与其他节点的元数据和关系信息。

在 LlamaIndex 中，Nodes 被认为是一等公民。 这意味着可以直接定义 Nodes 及其所有属性。或者也可以使用 NodeParser 类将源Document解析为Node。默认情况下，从文档派生的每个节点都会继承相同的元数据。例如，文档中的file_name字段会传播到每个节点。

1. 将整个Document对象发送至索引，默认的LLmaindex索引方式 
2. 在索引之前将Document转换为Node对象：当你的文档很长且希望在索引之前将其拆分成较小块（或节点）时，这种方法很实用。



### LLamaindex默认索引方式

```python
documents = SimpleDirectoryReader(input_files=["data/lanzhou.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
print(len(documents))
print(documents)
print(index)
```



## 8. LlamaIndex 中的 VectorStoreIndex 是什么？

VectorStoreIndex 是一种索引类型，它使用文本的向量表示以实现有效检索相关上下文。

它构建在 VectorStore 之上，后者是一种存储向量并允许快速最近邻搜索的数据结构。 VectorStoreIndex 接收 IndexNode 对象，这些对象代表了原始文档的块。 它使用一个嵌入模型（在ServiceContext中指定）将这些节点的文本内容转换成向量表示。然后这些向量被存储在VectorStore中。

在查询时，VectorStoreIndex 可以快速检索出针对特定查询最相关的节点。 它通过使用相同的嵌入模型将查询转换为向量，然后在 VectorStore 中执行最近邻搜索来实现这一点。



## 9. LlamaIndex 中的 retriever 是什么？

在LlamaIndex中，"retriever"是指从索引中检索相关上下文的组件。具体来说，它是一个负责从索引中获取与用户查询相关的IndexNode对象的系统。这些IndexNode对象代表了存储在索引中的原始文档块，它们可以是文本块、图像或其他类型的数据。

在LlamaIndex的架构中，retriever通常与一个查询引擎（如RetrieverQueryEngine）一起工作。RetrieverQueryEngine使用retriever来从VectorStoreIndex（一种基于向量的索引类型）中检索相关的IndexNode对象。这个过程涉及到将用户的查询转换为向量表示，然后在VectorStore中执行最近邻搜索，以找到与查询最相关的节点。

retriever在LlamaIndex中的作用可以总结如下：

1. **检索相关节点**：根据用户的查询，retriever从索引中检索出最相关的IndexNode对象。
2. **支持RAG用例**：对于“检索增强生成”（RAG）的应用场景，retriever是关键组件，它允许系统快速检索与查询相关的上下文信息。
3. **提高响应质量**：通过提供相关的上下文，retriever帮助生成更准确和信息丰富的响应。

在实际应用中，retriever的性能和准确性对于整个系统的效能至关重要，因为它直接影响到检索结果的相关性和最终生成的响应的质量。



## 10. LlamaIndex 中的 RetrieverQueryEngine 是什么？

LlamaIndex 中的 RetrieverQueryEngine 是一种查询引擎，它使用一个检索器从索引中获取相关的上下文，给定用户查询。 它主要用于和检索器一起工作，比如从 VectorStoreIndex 创建的 VectorStoreRetriever。 RetrieverQueryEngine 接受一个检索器和一个响应合成器作为输入。 检索器负责从索引中获取相关的 IndexNode 对象，而响应合成器则根据检索到的节点和用户查询生成自然语言响应。



## 11. LlamaIndex 中的 NodePostProcessor(Node后处理器) 是什么？

### NodePostProcessor定义

在 LlamaIndex 的上下文中，NodePostProcessor 是一个在数据节点（Node）被处理之后进行额外操作的组件。这些操作可以包括但不限于数据的清洗、增强、过滤或转换等，目的是确保最终提供给LLM的数据是最优的，从而提高模型响应的质量。NodePostProcessor 是通过继承自BaseNodePostprocessor接口实现的，该接口定义了_postprocess_nodes 方法，此方法接受一系列节点并返回经过处理后的节点列表。 在 LlamaIndex 中，NodePostProcessor 的主要作用是：

1. **数据清洗和隐私保护**：NodePostProcessor 可用于对节点数据进行清洗，如移除无关的信息或者进行数据格式标准化。此外，它还能用于隐私信息的掩蔽（masking）。例如，NERPIINodePostprocessor 就是用来识别并掩蔽个人身份信息（PII）的 NodePostProcessor。通过使用命名实体识别（NER）模型，它可以检测并替换掉节点中的 PII 。
2. **数据增强**：另一个重要的功能是数据增强，这可以通过添加前后文相关的其他节点来实现，以此来丰富LLM的上下文理解。例如，PrevNextNodePostprocessor 可以用来向前或向后扩展节点，以便为模型提供更多的上下文信息，这对于提高问答系统的准确性和连贯性尤其有用 。
3. **结果重排**：在一些场景下，NodePostProcessor 也可以用来对检索到的结果进行重排。通过重新评估和排序检索出的文档片段，可以确保最相关的信息优先呈现给用户。这种重排可以通过自定义 Rerank 功能的 NodePostProcessor 来实现，例如在 RAG（Retrieval-Augmented Generation）系统中 。
4. **动态捕捉视频后处理**：虽然这不是 LlamaIndex 的直接应用场景，但从更广泛的视角来看，NodePostProcessor 的概念也被应用到了诸如动态捕捉视频数据的后处理中。在这种情况下，它被用来处理节点数据以创建运动捕捉视频 。尽管这超出了 LlamaIndex 的原始设计范围，但它展示了 NodePostProcessor 概念的灵活性和可扩展性。



### 相似性节点后处理器 SimilarityPostprocessor

用于删除低于相似性分数阈值的节点。



### 其它后处理器 Node Postprocessor

https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#metadatareplacementpostprocessor