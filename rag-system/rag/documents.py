# @Time: 2025/3/13 15:03
# @Author: lvjing
import os
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, Document

from rag.base import BaseRAG
from rag.utils import extract_text_from_llm


class DocumentRAG(BaseRAG):

    async def load_data(self):
        """
        加载数据，该函数需要优化文件内容的识别、清洗
        :return:
        """
        docs = []
        for file in self.files:
            # 对图片及文档通过Moonshot大模型进行OCR识别
            # 改造成离线识别，使用本地的OCR工具
            contents = extract_text_from_llm(file)
            temp_file = datetime.now().strftime("%Y%m%d%H%M%S") + ".txt"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(contents)
                f_name = temp_file
            data = SimpleDirectoryReader(input_files=[f_name]).load_data()
            doc = Document(text="\n\n".join([d.text for d in data[0:]]), metadata={"path": file})
            docs.append(doc)
            os.remove(f_name)
        return docs
