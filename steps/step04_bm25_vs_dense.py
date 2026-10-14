"""Step 4: BM25 vs Dense vs Hybrid (RRF) บน index เล็ก ๆ 6 ประโยค (ต้องเปิด OpenSearch ก่อน)

    python steps/step04_bm25_vs_dense.py
"""
from common import embedder, opensearch

INDEX = "workshop-step04"
docs = [
    "โรคหัดเยอรมันเกิดจากเชื้อไวรัสรูเบลลา อยู่ในตระกูลโทกาวิริดี",
    "หญิงตั้งครรภ์ที่ติดเชื้อหัดเยอรมันในไตรมาสแรก ทารกอาจพิการแต่กำเนิด",
    "อหิวาตกโรคเกิดจากเชื้อแบคทีเรียวิบริโอคอเลอเร",
    "ผู้ป่วยอหิวาตกโรคถ่ายอุจจาระเป็นน้ำซาวข้าว",
    "ต้อกระจกทำให้แก้วตาขุ่น มองเห็นไม่ชัด",
    "ยาโอเมพราโซลใช้ลดกรดในกระเพาะอาหารสำหรับโรคกรดไหลย้อน",
]

client, model = opensearch(), embedder()
if client.indices.exists(index=INDEX):
    client.indices.delete(index=INDEX)
client.indices.create(index=INDEX, body={
    "settings": {"index.knn": True},
    "mappings": {"properties": {
        "text": {"type": "text", "analyzer": "thai"},   # สำหรับ BM25
        "vec": {"type": "knn_vector", "dimension": 1024,  # สำหรับ dense
                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}}}})
vecs = model.encode(docs, normalize_embeddings=True)
for i, (d, v) in enumerate(zip(docs, vecs)):
    client.index(index=INDEX, id=str(i), body={"text": d, "vec": v.tolist()})
client.indices.refresh(index=INDEX)


def bm25(q, k=3):
    hits = client.search(index=INDEX, body={"size": k, "query": {"match": {"text": q}}})["hits"]["hits"]
    return [(h["_id"], round(h["_score"], 2)) for h in hits]


def dense(q, k=3):
    v = model.encode(q, normalize_embeddings=True).tolist()
    hits = client.search(index=INDEX, body={"size": k, "query": {"knn": {"vec": {"vector": v, "k": k}}}})["hits"]["hits"]
    return [(h["_id"], round(h["_score"], 2)) for h in hits]


def rrf(*rankings, k=60):
    """Reciprocal Rank Fusion: ใช้ 'อันดับ' ไม่ใช้ 'คะแนน' เพราะคะแนน BM25 กับ cosine คนละสเกล"""
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(((d, round(s, 4)) for d, s in scores.items()), key=lambda x: -x[1])


for q in ["ยาโอเมพราโซล",                      # ชื่อเฉพาะ: BM25 ถนัด
          "ตาพร่ามัวเหมือนมีหมอกบัง",          # ไม่มีคำตรงกับเอกสาร: BM25 อาจไม่เจอ
          "pregnant woman rubella"]:          # ถามภาษาอังกฤษ: BM25 ไม่เจอแน่นอน dense ยังเจอ
    b, d = bm25(q), dense(q)
    print(f"\nQ: {q}")
    print("  BM25  :", b, "->", docs[int(b[0][0])] if b else "(ไม่เจอเลย)")
    print("  dense :", d, "->", docs[int(d[0][0])])
    print("  RRF   :", rrf(b, d)[:3])

client.indices.delete(index=INDEX)
# สังเกต: คะแนน BM25 (เช่น 3.5) กับ cosine (0.6-0.8) เทียบกันตรง ๆ ไม่ได้ -> RRF ใช้อันดับแทน
# สองวิธีเก่งคนละแบบ จึงรวมกันเป็น hybrid
