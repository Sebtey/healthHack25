from fastapi import FastAPI, File, UploadFile
import io
from machine_learning import get_model
import uvicorn
import logging
import traceback

app = FastAPI()

model = get_model()
logger = logging.getLogger("uvicorn.error")

def log_item(i):
    logger.info(i)
    return i


@app.get("/")  # Ensure this is present
async def root():
    return {"message": "Welcome to the prediction API. Upload mp4 file with /upload to get started!"}


@app.post("/upload")
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Read file contents into memory
    mp4_bytes = await file.read()

    # Convert to BytesIO object
    mp4_buffer = io.BytesIO(mp4_bytes)
    result = None
    try:
        result = model(mp4_bytes)
    except Exception as e:
        log_item("".join(traceback.format_exception(etype=None, value=e, tb=e.__traceback__)))
        result = (str(e), str(e.__cause__))

    return {"filename": file.filename, "size": len(mp4_bytes), "model_Result": result}

# Run server
# Save this as main.py and run: uvicorn main:app --host 0.0.0.0 --port 8000
