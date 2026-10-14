"""Step 6: ส่องข้างใน index ของระบบครู (รันหลังจาก python authenticRAG.py เสร็จแล้ว)

    python steps/step06_inspect_teacher.py

ตอบ 4 คำถามด้วยข้อมูลจริง ไม่ใช่ด้วยการอ่านชื่อตัวแปร:
  1) มีกี่ chunk ขนาดเท่าไร  (โค้ดเขียนว่า chunk_size=256 — จริงไหม?)
  2) BM25 index ใช้ analyzer อะไร และมันตัดภาษาไทยออกมาเป็นอะไร
  3) "context" ที่ LLM สร้างให้แต่ละ chunk หน้าตาเป็นอย่างไร (ภาษาอะไร รู้ไหมว่าอยู่เอกสารไหน)
  4) ค้นด้วย BM25 แล้วเจออะไรบ้าง
"""
import statistics

from common import opensearch

VEC, BM25 = "anthropic-vector-index", "anthropic-bm25-index"
client = opensearch()
for name in (VEC, BM25):
    if not client.indices.exists(index=name):
        raise SystemExit(f"ไม่พบ index {name} — รัน authenticRAG.py ของครูก่อน")

docs = [h["_source"] for h in client.search(index=BM25, body={"size": 1000, "query": {"match_all": {}}})["hits"]["hits"]]
sizes = sorted(len(d["content"]) for d in docs)
print(f"1) {len(docs)} chunks, ความยาว (ตัวอักษร) min={sizes[0]} median={statistics.median(sizes):.0f} max={sizes[-1]}")
print(f"   chunk ที่ยาวเกิน 1,500 ตัวอักษร: {sum(s > 1500 for s in sizes)} chunk")

mapping = client.indices.get_mapping(index=BM25)[BM25]["mappings"]["properties"]["content"]
print(f"\n2) field content ใช้ analyzer: {mapping.get('analyzer', 'standard')}")
sample = docs[0]["content"][:80]
tokens = client.indices.analyze(index=BM25, body={"field": "content", "text": sample})["tokens"]
print(f"   ข้อความ: {sample!r}\n   term ที่เก็บจริง: {[t['token'] for t in tokens][:8]}")

print("\n3) ตัวอย่าง context ที่ LLM สร้าง:")
for d in docs[:3]:
    print(f"   - chunk: {d['content'][:60]!r}...\n     context: {d['contextualized_content'][:150]!r}")

q = "ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน"
hits = client.search(index=BM25, body={"size": 5, "query": {"multi_match": {
    "query": q, "fields": ["content", "contextualized_content"]}}})["hits"]["hits"]
print(f"\n4) BM25 ค้น {q!r} เจอ {len(hits)} chunk")
for h in hits:
    print(f"   score={h['_score']:.2f} {h['_source']['content'][:70]!r}")
