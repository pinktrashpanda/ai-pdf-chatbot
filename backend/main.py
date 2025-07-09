import os
import tempfile
import pdfplumber
import camelot
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import json

app = FastAPI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("pdf_chunks")

def extract_pdf_text(pdf_path):
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_chunks.append({
                    "type": "text",
                    "page": i + 1,
                    "content": text.strip()
                })
    return text_chunks

def extract_pdf_tables(pdf_path):
    tables = []
    try:
        all_tables = camelot.read_pdf(pdf_path, pages="all")
        for t in all_tables:
            tables.append({
                "type": "table",
                "page": t.page,
                "content": t.df.to_csv(index=False)
            })
    except Exception as e:
        print(f"Table extraction failed: {e}")
    return tables

def chunk_text(text_chunks, max_length=1000):
    chunks = []
    for chunk in text_chunks:
        content = chunk["content"]
        parts = [p.strip() for p in content.split("\n\n") if p.strip()]
        buffer = ""
        for part in parts:
            if len(buffer) + len(part) < max_length:
                buffer += " " + part
            else:
                chunks.append({
                    "type": chunk["type"],
                    "page": chunk["page"],
                    "content": buffer.strip()
                })
                buffer = part
        if buffer:
            chunks.append({
                "type": chunk["type"],
                "page": chunk["page"],
                "content": buffer.strip()
            })
    return chunks

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text_chunks = extract_pdf_text(tmp_path)
    table_chunks = extract_pdf_tables(tmp_path)
    all_chunks = chunk_text(text_chunks) + table_chunks

    for chunk in all_chunks:
        content = chunk["content"]
        chunk_id = str(uuid.uuid4())
        emb = embedder.encode(content)
        metadata = {
            "type": chunk["type"],
            "page": chunk["page"],
            "content": content
        }
        collection.add(
            embeddings=[emb.tolist()],
            documents=[content],
            metadatas=[metadata],
            ids=[chunk_id]
        )
    os.remove(tmp_path)
    return JSONResponse({"status": "processed", "chunks": len(all_chunks)})

@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    q_emb = embedder.encode(query)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=5
    )
    retrieved = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = "\n\n".join(
        f"(Page {md['page']} - {md['type']}): {doc}" for doc, md in zip(retrieved, metadatas)
    )
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Answer the question using ONLY the following context (from a PDF):\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=512
    )
    answer = response.choices[0].text.strip()
    return JSONResponse({"answer": answer, "context": context})
