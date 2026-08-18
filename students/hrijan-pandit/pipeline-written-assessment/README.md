# Pipeline: Written Assessment (AnswerLens)

## Overview
AnswerLens is an AI-powered web application that accepts a photo or scanned image of a student's handwritten answer sheet and automatically produces a transcription, assessment, score out of 10, and written feedback.

## Architecture
- **Backend**: FastAPI + Qwen2.5-VL-7B-Instruct (vision-language model) deployed on Vast.ai GPU instance, exposed via Cloudflare tunnel
- **Frontend**: Next.js 14 web application deployed on Vercel

## Installation

### Backend (Vast.ai or any GPU instance with ≥16GB VRAM)
```bash
git clone https://github.com/HrijP/Deployed.git
cd Deployed/backend
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000 &
```

### Frontend (Local or Vercel)
```bash
cd Deployed/frontend
npm install
# Create .env.local with:
# NEXT_PUBLIC_BACKEND_URL=https://YOUR_TUNNEL_URL
npm run dev
```

## Dependencies
- Python: fastapi, uvicorn, transformers, torch, Pillow, python-multipart
- Node.js: Next.js 14, React 18, TypeScript

## Model
- Qwen2.5-VL-7B-Instruct (Hugging Face: Qwen/Qwen2.5-VL-7B-Instruct)
- Vision-language model capable of reading handwritten text in images

## API Endpoints
- GET /health — returns model status and device info
- POST /grade — accepts image file, returns JSON with transcription, assessment, score, feedback

## Input/Output
- Input: JPG, PNG, or WEBP image of a student answer sheet (max 20MB)
- Output:
```json
{
  "transcription": "...",
  "assessment": "...",
  "score": "8/10 ...",
  "feedback": "..."
}
```

## Deployment
See DEPLOY.md for complete step-by-step instructions covering Vast.ai instance setup, Cloudflare tunnel configuration, and Vercel deployment.

## Live Demo
- Frontend: https://deployed-hrijanns-projects.vercel.app
- GitHub: https://github.com/HrijP/Deployed
