"""
Student Answer Grader - FastAPI Backend
Deploy this on Vast.ai (GPU instance with at least 16GB VRAM)
Model: Qwen2.5-VL-7B-Instruct (vision-language model)
"""

import io
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Model config ───────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

model     = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    global model, processor
    logger.info(f"Loading {MODEL_ID} on {DEVICE}…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Student Answer Grader API", lifespan=lifespan)

# Allow requests from your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your Vercel URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─── Schemas ────────────────────────────────────────────────────────────────
class GradeResponse(BaseModel):
    transcription: str
    assessment:    str
    score:         Optional[str]
    feedback:      str


# ─── Helpers ────────────────────────────────────────────────────────────────
TRANSCRIPTION_PROMPT = """You are reviewing a scanned or photographed student answer sheet.

Step 1 – TRANSCRIPTION
Read every word the student wrote and transcribe it exactly, preserving structure (headings, numbered points, equations, etc.). Label this section "TRANSCRIPTION:".

Step 2 – ASSESSMENT
Evaluate the student's answer on these dimensions:
• Accuracy      – Is the content correct? Identify any errors.
• Completeness  – Did the student address all parts of the question?
• Clarity       – Is the writing legible and well-organised?
• Depth         – Does the answer show understanding beyond surface recall?

Label this section "ASSESSMENT:".

Step 3 – SCORE
Give a score out of 10 with a one-line justification. Label this "SCORE:".

Step 4 – FEEDBACK
Write 2–4 sentences of constructive feedback the student can act on. Label this "FEEDBACK:"."""


def parse_model_output(raw: str) -> GradeResponse:
    """Split model output into structured fields."""
    sections = {"TRANSCRIPTION": "", "ASSESSMENT": "", "SCORE": "", "FEEDBACK": ""}
    current = None
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper().rstrip(":")
        if upper in sections:
            current = upper
            continue
        if current:
            sections[current] += line + "\n"

    return GradeResponse(
        transcription=sections["TRANSCRIPTION"].strip() or raw,
        assessment=sections["ASSESSMENT"].strip(),
        score=sections["SCORE"].strip() or None,
        feedback=sections["FEEDBACK"].strip(),
    )


def run_inference(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": TRANSCRIPTION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}


@app.post("/grade", response_model=GradeResponse)
async def grade_answer(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    raw_bytes = await file.read()
    if len(raw_bytes) > 20 * 1024 * 1024:      # 20 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB).")

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    logger.info(f"Grading image: {image.size}, {file.content_type}")
    raw_output = run_inference(image)
    logger.info("Inference complete.")

    return parse_model_output(raw_output)
