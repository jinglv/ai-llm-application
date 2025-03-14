# @Time: 2025/3/13 15:02
# @Author: lvjing
from pathlib import Path

from conf.settings import moonshot_llm


def extract_text_from_llm(file_path) -> str:
    """
    提取文件中的文本
    （月之暗面提取文本模型是免费的）
    :param file_path: 文件路径
    :return:
    """
    client = moonshot_llm()
    file_object = client.files.create(file=Path(file_path), purpose="file-extract")
    file_content = client.files.content(file_id=file_object.id).json()
    return file_content.get("content")
