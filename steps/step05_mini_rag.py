"""Step 5: RAG ทั้งวงจรใน ~40 บรรทัด ไม่ใช้ OpenSearch (ใช้ numpy แทน vector DB)

chunk -> embed -> ค้นด้วย cosine -> ใส่ chunk ลงส่วน Context ของ prompt -> LLM ตอบ

    python steps/step05_mini_rag.py
    python steps/step05_mini_rag.py "อหิวาตกโรคติดต่อทางไหน"
"""
import sys

from common import embedder, llm, read_corpus

# 1) Chunking: ตัดทุกไฟล์ตามย่อหน้า (บรรทัดว่าง) — แบบง่ายที่สุด
chunks = []
for name, text in read_corpus().items():
    chunks += [(name, p.strip()) for p in text.split("\n\n") if len(p.strip()) > 50]
print(f"{len(chunks)} chunks")

# 2) Embedding ทุก chunk (ทำครั้งเดียว ของจริงจะเก็บลง vector DB)
model = embedder()
chunk_vecs = model.encode([c for _, c in chunks], normalize_embeddings=True, batch_size=16, show_progress_bar=True)

# 3) Retrieval: embed คำถามด้วยโมเดลเดียวกัน แล้วหา top-k ที่ cosine สูงสุด
question = sys.argv[1] if len(sys.argv) > 1 else "ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน"
q_vec = model.encode(question, normalize_embeddings=True)
scores = chunk_vecs @ q_vec
top = scores.argsort()[::-1][:3]
for rank, i in enumerate(top, 1):
    print(f"\n#{rank} score={scores[i]:.3f} [{chunks[i][0]}]\n{chunks[i][1][:200]}")

# 4) Augmented: เอา chunk ไปใส่ส่วน Context ของ prompt
context = "\n\n".join(f"[{n}] {chunks[i][1]}" for n, i in enumerate(top, 1))
prompt = f"""ตอบคำถามโดยใช้ข้อมูลใน Context เท่านั้น ถ้าไม่มีให้ตอบว่าไม่พบข้อมูล และอ้างอิงเลข [n]

Context:
{context}

Question: {question}"""

# 5) Generation
print("\nA:", llm(prompt))
