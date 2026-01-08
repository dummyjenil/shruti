from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
from shruti import ShrutiASR
import asyncio

app = FastAPI()

model = ShrutiASR(True, True).to("cuda")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    lang: str = "hi"
):
    # unique filename
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        # save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # run model in thread (GPU safe for FastAPI)
        srt, speaker = await asyncio.to_thread(
            model.forward,
            file_path,
            4,
            lang,
            False
        )

        return JSONResponse({
            "status": "success",
            "srt": srt,
            "speaker": speaker
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
