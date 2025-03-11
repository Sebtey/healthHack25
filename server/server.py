from fastapi import FastAPI, File, UploadFile
import io

app = FastAPI()


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

    return {"filename": file.filename, "size": len(mp4_bytes)}

# Run server
# Save this as main.py and run: uvicorn main:app --host 0.0.0.0 --port 8000
