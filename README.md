# AeroVantage X9 — PDF-grounded RAG Technical Assistant

## Architecture
`50-page PDF -> PyMuPDF page extraction -> overlapping 850-character chunks (160 overlap) -> sentence-transformers embeddings -> persistent ChromaDB -> semantic retrieval and lexical reranking -> optional OpenAI LLM -> answer with page sources`.

## Setup (Mac / VS Code)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```
The first run downloads an embedding model and indexes the manual. Later runs reuse `chroma_db/` without regenerating embeddings. Optional: set `OPENAI_API_KEY` in `.env` for LLM answers. Without a key, an offline extractive mode prints source excerpts. Never commit `.env` or the persistent database.

## Required function
`answer(question: str) -> dict` in `main.py` returns `{'answer': str, 'sources': [page_number_or_chunk_id, ...]}`.

## Grounding rules
The only knowledge corpus is `manual.pdf`. If no relevant supported evidence is retrieved, return exactly `[not found]`. Keep source page numbers. A retrieved context is not a guarantee of entailment: inspect answers for unsupported claims, especially negative questions. The optional LLM has a strict abstention instruction, but automated grounding is not perfect.

## Five assessment questions and expected PDF-supported answers
1. **Q1:** Primary IMU Magnetometer Calibration Fault and Compass Interference Lockout, code `ERR_IMU_MAG_07` (page 12).
2. **Q2:** `[not found]` — manual does not explicitly address underwater sonar mapping at 50 m.
3. **Q3:** `3.45 Volts per cell` triggers forced emergency RTH (page 29).
4. **Q4:** `38 knots` (`70 km/h` or `19.5 m/s`) sustained wind (page 44).
5. **Q5:** `[not found]` — CEO of DJI Innovations is not established in this manual.

## Assessment actual outputs
Run `python main.py` and paste its five *actual* generated answers and sources here. Compare them against expected answers above; do not claim success without running and checking. Capture a terminal screenshot for the portal.

## GitHub
```bash
git init
git add .
git commit -m "feat: implement PDF-grounded AeroVantage X9 RAG assistant"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```
