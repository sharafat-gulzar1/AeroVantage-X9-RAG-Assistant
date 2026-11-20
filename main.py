"""PDF-only, page-cited RAG assistant. Run: python main.py"""
import os
import re
from pathlib import Path
import fitz
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
BASE = Path(__file__).resolve().parent
PDF = BASE / 'manual.pdf'
DB = BASE / 'chroma_db'
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CLIENT = chromadb.PersistentClient(path=str(DB))
COLLECTION = CLIENT.get_or_create_collection('aerovantage_x9')

QUESTIONS = [
    'What does the diagnostic LED pattern of three rapid red flashes followed by two solid amber flashes indicate on the X9 flight controller?',
    'Can the AeroVantage X9 be used for underwater sonar mapping at a depth of 50 metres?',
    'At what minimum per-cell voltage, in volts, does the X9 trigger a forced emergency Return-to-Home (RTH) landing?',
    'What is the maximum sustained wind speed at which the X9 is certified to operate safely?',
    'Who is the current Chief Executive Officer of DJI Innovations?',
]

def ingest():
    if COLLECTION.count():
        print(f'Reusing persistent ChromaDB ({COLLECTION.count()} chunks)')
        return
    docs = []
    with fitz.open(PDF) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            text = re.sub(r'\s+', ' ', page.get_text()).strip()
            # Page-aware overlapping character chunks, never crossing page boundaries.
            chunk_size, overlap = 850, 160
            for start in range(0, len(text), chunk_size - overlap):
                chunk = text[start:start + chunk_size]
                if len(chunk) < 30:
                    continue
                docs.append((f'p{page_no}_c{start}', chunk, page_no))
    for offset in range(0, len(docs), 64):
        batch = docs[offset:offset + 64]
        embeddings = MODEL.encode([d[1] for d in batch], normalize_embeddings=True).tolist()
        COLLECTION.add(ids=[d[0] for d in batch], documents=[d[1] for d in batch],
                       metadatas=[{'page': d[2]} for d in batch], embeddings=embeddings)
    print(f'Indexed {len(docs)} chunks from 50 pages')

def answer(question: str) -> dict:
    """Returns {'answer': str, 'sources': [page_or_chunk_id, ...]}."""
    ingest()
    embedding = MODEL.encode([question], normalize_embeddings=True).tolist()
    result = COLLECTION.query(query_embeddings=embedding, n_results=min(10, COLLECTION.count()),
                              include=['documents', 'metadatas', 'distances'])
    candidates = list(zip(result['documents'][0], result['metadatas'][0], result['distances'][0]))
    # Drop boilerplate and favor question-specific evidence, not mere repeated headings.
    stop = {'what','does','the','and','for','with','from','this','that','which','when','where','who','can','used','at','in','of','to','is','a','an','on','be','it','its','x9','aerovantage','minimum','maximum'}
    words = {w for w in re.findall(r'[a-z0-9]+', question.lower()) if len(w)>2 and w not in stop}
    scored = []
    for passage, meta, distance in candidates:
        lower = passage.lower()
        overlap = sum(w in lower for w in words)
        informative = 'SECTION ' in passage and any(c.isdigit() for c in passage)
        scored.append((overlap + (2 if informative else 0) - distance, passage, meta))
    scored.sort(reverse=True)
    best = scored[:4]
    # Strict abstention: do not infer unsupported external facts or unsupported capabilities.
    if not best or max(x[0] for x in best) < 2.0 or not any('SECTION ' in x[1] for x in best):
        return {'answer': '[not found]', 'sources': []}
    context = '\n\n'.join(f"[Page {meta['page']}] {passage}" for _, passage, meta in best)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = ('Answer ONLY from the supplied PDF excerpts. If the exact answer is not explicitly '
                  'supported, reply exactly [not found]. Do not use outside knowledge. '
                  'Answer concisely, preserving exact numerical values and error codes.\n\n'
                  f'EXCERPTS:\n{context}\n\nQUESTION: {question}')
        response = client.chat.completions.create(model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    messages=[{'role':'user','content':prompt}], temperature=0)
        generated = (response.choices[0].message.content or '').strip()
        if generated == '[not found]':
            return {'answer': '[not found]', 'sources': []}
        return {'answer': generated, 'sources': sorted({f"page {m['page']}" for _, _, m in best})}
    # Offline fallback: grounded source excerpt rather than inventing an LLM response.
    top = next(((p,m) for _,p,m in best if 'SECTION ' in p), None)
    if top is None:
        return {'answer': '[not found]', 'sources': []}
    passage, meta = top
    section = passage.split('SECTION ',1)[1]
    excerpt = 'SECTION ' + section.split('Operational Compliance Note')[0].strip()
    # Offline evidence gate: at least two distinctive question terms must occur in the factual section.
    if sum(w in excerpt.lower() for w in words) < 2:
        return {'answer': '[not found]', 'sources': []}
    return {'answer': excerpt, 'sources': [f"page {meta['page']}"]}

if __name__ == '__main__':
    for index, question in enumerate(QUESTIONS, 1):
        print(f'\nQ{index}: {question}\n{answer(question)}')
