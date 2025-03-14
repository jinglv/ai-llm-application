# RAG知识库系统

LlamaIndex + Chainlit + PostgresSQL + milvus + minio构建RAG知识库系统

- LlamaIndex -- 知识库构建及检索
- Chainlit -- 快速构建前端面
- PostgresSQL -- 数据库
- Milvus -- 向量数据库
- Minio -- 对象存储

Chainlit官方文档：https://docs.chainlit.io/get-started/overview

LlamaIndex官方文档：https://docs.llamaindex.ai/en/stable/

## 环境搭建

### 1. 安装依赖包

python版本：3.13.2

升级pip：pip install --upgrade pip

```
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-llms-deepseek
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-ollama
pip install llama-index-vector-stores-milvus
pip install llama-index-readers-file
pip install chainlit
pip install unicorn
```

### 2. Docker安装启动postgres

```
docker pull postgres

docker run --name postgre-sql -p 5432:5432 -e POSTGRES_PASSWORD=123456 -e LANG="C.UTF-8" -d postgres
```

- 用户名：postgres
- 密码：123456

### 3. Docker安装启动minio

```
docker pull minio/minio

docker run -p 9000:9000 -p 9001:9001 --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  -v /Users/jeanlv/data/minio/data:/data \
  -v /Users/jeanlv/data/minio/config:/root/.minio \
  -d minio/minio server /data --console-address ":9001"
```

- 访问地址：http://localhost:9000
- 用户名：minioadmin
- 密码：minioadmin

### 4.Docker安装启动milvus

文档：https://milvus.io/docs/install-overview.md

```
# 通过docker-compose
mkdir milvus_compose
cd milvus_compose

wget https://github.com/milvus-io/milvus/releases/download/v2.5.6/milvus-standalone-docker-compose.yml -O docker-compose.yml

docker-compose up -d
```

## RAG优化及核心知识

- 入库优化（最关键）
    - 数据入库前的高质量（难点）
- 召回优化
    - top_k
    - 重排