# @Time: 2025/3/14 15:21
# @Author: lvjing
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse

load_dotenv(verbose=True)

from rag.documents import DocumentRAG
from utils.r import R

app = FastAPI()


@app.post("/files/")
async def create_upload_files(files: list[UploadFile], collection_name: str = Form()):
    file_list = []
    for file in files:
        file_path = os.path.join("documents", os.path.basename(file.filename))
        with open(file_path, "wb+") as f:
            f.write(await file.read())
            file_list.append(file_path)
    rag = DocumentRAG(files=file_list)
    await rag.create_remote_index(collection_name=collection_name)
    return R.ok("index success")


@app.get("/")
async def main():
    content = """
        <body>
        <form action="/files/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input name="collection_name" type="text">
        <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)


# mount_chainlit(app=app, target="web_ui.py", path="/chainlit")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
