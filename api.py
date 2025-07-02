import os
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np
import tempfile
import shutil
from kokoro_tts import (
    validate_language, validate_voice, extract_chapters_from_epub, PdfParser,
    chunk_text, process_chunk_sequential, merge_chunks_to_chapters
)

# TODO: Refactor kokoro-tts script into a module for API use

app = FastAPI(title="Kokoro TTS API", version="1.0")

# Allow CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Kokoro model
class KokoroSingleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        return cls._instance

@app.get("/v1/audio/voices")
def get_voices():
    kokoro = KokoroSingleton.get_instance()
    voices = sorted(kokoro.get_voices())
    return {"voices": voices}

@app.get("/v1/audio/languages")
def get_languages():
    kokoro = KokoroSingleton.get_instance()
    languages = sorted(kokoro.get_languages())
    return {"languages": languages}

@app.post("/v1/audio/merge")
def merge_audio(
    split_output: str = Form(...),
    format: str = Form("wav")
):
    # Use a temp file for the merged output
    temp_dir = tempfile.mkdtemp()
    try:
        merge_chunks_to_chapters(split_output, format)
        # Find merged files in split_output
        merged_files = [f for f in os.listdir(split_output) if f.endswith(f'.{format}')]
        if not merged_files:
            raise HTTPException(status_code=404, detail="No merged files found.")
        # Return the first merged file (or customize as needed)
        merged_path = os.path.join(split_output, merged_files[0])
        return FileResponse(merged_path, media_type=f"audio/{format}", filename=os.path.basename(merged_path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/v1/audio/speech")
async def tts_speech(
    request: Request,
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    voice: Optional[str] = Form("af_sarah"),
    lang: Optional[str] = Form("en-us"),
    speed: Optional[float] = Form(1.0),
    format: Optional[str] = Form("wav")
):
    """
    OpenAI-compatible TTS endpoint. Accepts either JSON (with 'input') or multipart/form-data (with 'file' or 'text').
    - If 'file' is present, it is used (EPUB, PDF, TXT supported).
    - If 'input' (or 'text') is present, it is used as the text to synthesize.
    - All other parameters (voice, lang, speed, format) are supported in both modes.
    """
    kokoro = KokoroSingleton.get_instance()
    content_type = request.headers.get("content-type", "")
    input_text = None
    upload_file = None
    # Handle JSON requests
    if content_type.startswith("application/json"):
        data = await request.json()
        input_text = data.get("input") or data.get("text")
        voice = data.get("voice", voice)
        lang = data.get("lang", lang)
        speed = float(data.get("speed", speed))
        format = data.get("format", format)
    # Handle multipart/form-data
    else:
        # If file is present, use it
        if file is not None:
            upload_file = file
        # If text is present, use it
        elif text is not None:
            input_text = text
    # If both file and text/input are missing, return error
    if upload_file is None and not input_text:
        raise HTTPException(status_code=400, detail="No input text or file provided.")
    # Validate language and voice
    lang = validate_language(lang, kokoro)
    voice = validate_voice(voice, kokoro)
    # If file is present, process it
    if upload_file is not None:
        filename = upload_file.filename.lower()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name
        if filename.endswith('.epub'):
            chapters = extract_chapters_from_epub(tmp_path)
            input_text = '\n'.join([c['content'] for c in chapters])
        elif filename.endswith('.pdf'):
            parser = PdfParser(tmp_path)
            chapters = parser.get_chapters()
            input_text = '\n'.join([c['content'] for c in chapters])
        elif filename.endswith('.txt'):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
        else:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        os.unlink(tmp_path)
    # Chunk and synthesize
    chunks = chunk_text(input_text, initial_chunk_size=1000)
    all_samples = []
    sample_rate = None
    for chunk in chunks:
        samples, sr = process_chunk_sequential(chunk, kokoro, voice, speed, lang)
        if samples is not None:
            if sample_rate is None:
                sample_rate = sr
            all_samples.extend(samples)
    if not all_samples or sample_rate is None:
        raise HTTPException(status_code=500, detail="TTS synthesis failed.")
    # Write to buffer
    buf = io.BytesIO()
    sf.write(buf, np.array(all_samples), sample_rate, format=format.upper())
    buf.seek(0)
    media_type = f"audio/{'wav' if format == 'wav' else 'mpeg'}"
    return StreamingResponse(buf, media_type=media_type, headers={
        "Content-Disposition": f"attachment; filename=output.{format}"
    }) 