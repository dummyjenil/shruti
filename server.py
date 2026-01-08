from fastapi import FastAPI, UploadFile, File
import os
import shutil
import uuid
from shruti import ShrutiASR
import asyncio
app = FastAPI()
model = ShrutiASR().to("cuda")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    lang: str = "hi"
):
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return await asyncio.to_thread(model.forward,file_path,64,lang,False)
    finally:
        os.remove(file_path)
