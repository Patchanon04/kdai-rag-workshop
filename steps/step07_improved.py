"""Step 7: ระบบที่ปรับปรุงแล้ว (LLM ตัวเดิม เปลี่ยนเฉพาะ chunking / context / index / search / rerank / filter)

    python steps/step07_improved.py ingest                  # สร้าง index (เรียก LLM 1 ครั้งต่อ chunk, cache ไว้)
    python steps/step07_improved.py ask "คำถาม"             # ถาม-ตอบ (mode = filter)
    python steps/step07_improved.py ask "คำถาม" hybrid      # ลองโหมดอื่น: bm25 | dense | hybrid | rerank | filter

แต่ละการปรับปรุงมีหมายเลข [P1]..[P7] ตรงกับหัวข้อในคู่มือ
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ROOT, embedder, llm, opensearch, read_corpus

INDEX = "workshop-improved"
MAX_CHARS = 800
CONTEXT_CACHE = Path(os.environ.get("CONTEXT_CACHE", ROOT / "results" / "contexts.json"))


# [P1] Chunking: ตัดตามหัวข้อ markdown ก่อน แล้วค่อยจำกัดขนาด + overlap + breadcrumb
def chunk_markdown(text, source):
    title, chunks, stack, lines = "", [], [], []

    def flush():
        body = [l for l in lines if l.strip()]
        if not body:
            return
        crumb, buf = " > ".join(stack), []
        for line in body:
            while len(line) > MAX_CHARS:                      # บรรทัดยาวเกิน -> ตัดที่ช่องว่าง
                cut = line.rfind(" ", 0, MAX_CHARS)
                cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
                piece, line = line[:cut], line[cut:].strip()
                buf = emit(buf, piece, crumb, body)
            buf = emit(buf, line, crumb, body)
        if buf:
            chunks.append(make(buf, crumb, body))

    def emit(buf, line, crumb, body):
        if buf and sum(map(len, buf)) + len(line) > MAX_CHARS:
            chunks.append(make(buf, crumb, body))
            buf = buf[-1:] if len(buf[-1]) < 300 else []        # overlap 1 บรรทัด
        return buf + [line]

    def make(buf, crumb, body):
        return {"source": source, "title": title, "breadcrumb": crumb,
                "content": "\n".join(buf), "section": "\n".join(body)}

    for raw in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", raw)
        if m:
            flush()
            lines, level, head = [], len(m.group(1)), m.group(2).strip()
            title = title or head
            stack = stack[: level - 1] + [head]                 # breadcrumb: เอกสาร > หัวข้อ > หัวข้อย่อย
        else:
            lines.append(raw)
    flush()
    return chunks


# [P2] Contextual Retrieval ที่ LLM ได้เห็นเอกสารจริง (สารบัญ + section แม่) และตอบเป็นภาษาไทย
CONTEXT_PROMPT = """<document_title>{title}</document_title>
<document_outline>
{outline}
</document_outline>
<section path="{crumb}">
{section}
</section>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences, in Thai) to situate this chunk within the overall document for the purposes of improving search retrieval. Mention the disease name and what aspect this chunk covers (e.g. cause, symptom, treatment), and key synonyms a reader might search with. Answer only with the succinct context and nothing else."""


def contextualize(chunks, outlines):
    cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8")) if CONTEXT_CACHE.exists() else {}

    def one(c):
        key = c["source"] + "|" + c["content"][:200]
        if key not in cache:
            cache[key] = llm(CONTEXT_PROMPT.format(
                title=c["title"], outline=outlines[c["source"]], crumb=c["breadcrumb"],
                section=c["section"][:3000], chunk=c["content"]), temperature=0.1)
        c["context"] = cache[key]

    CONTEXT_CACHE.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(one, chunks), 1):
            print(f"context {i}/{len(chunks)}", end="\r")
            CONTEXT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print()


def ingest():
    chunks, outlines = [], {}
    for name, text in read_corpus().items():
        outlines[name] = "\n".join(l for l in text.splitlines() if l.startswith("#"))
        chunks += chunk_markdown(text, name)
    print(f"{len(chunks)} chunks")
    contextualize(chunks, outlines)

    client, model = opensearch(), embedder()
    if client.indices.exists(index=INDEX):
        client.indices.delete(index=INDEX)
    thai = {"type": "text", "analyzer": "thai"}                  # [P3] analyzer ภาษาไทย
    client.indices.create(index=INDEX, body={
        "settings": {"index.knn": True},
        "mappings": {"properties": {
            "embedding": {"type": "knn_vector", "dimension": 1024,
                          "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
            "content": thai, "context": thai, "breadcrumb": thai, "source": {"type": "keyword"}}}})
    # [P4] embed breadcrumb + context + chunk (โมเดลเดิม bge-m3)
    texts = [f"{c['breadcrumb']}\n{c['context']}\n{c['content']}" for c in chunks]
    vecs = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    from opensearchpy import helpers
    helpers.bulk(client, ({"_index": INDEX, "_id": f"c{i}", "embedding": v.tolist(),
                           **{k: c[k] for k in ("source", "breadcrumb", "content", "context")}}
                          for i, (c, v) in enumerate(zip(chunks, vecs))))
    client.indices.refresh(index=INDEX)
    print(f"indexed {len(chunks)} chunks into {INDEX}")


# ---------------------------------------------------------------- search
_reranker = None


def bm25(q, k=20):
    r = opensearch().search(index=INDEX, body={"size": k, "_source": {"excludes": ["embedding"]}, "query": {
        "multi_match": {"query": q, "fields": ["content", "context", "breadcrumb^2"], "type": "most_fields"}}})
    return [(h["_id"], h["_source"]) for h in r["hits"]["hits"]]


def dense(q, k=20):
    v = embedder().encode(q, normalize_embeddings=True).tolist()
    r = opensearch().search(index=INDEX, body={"size": k, "_source": {"excludes": ["embedding"]},
                                               "query": {"knn": {"embedding": {"vector": v, "k": k}}}})
    return [(h["_id"], h["_source"]) for h in r["hits"]["hits"]]


def rrf(rankings, k=60):
    """[P5] RRF ที่ rank เริ่ม 1 (สูตรมาตรฐาน) + candidate ฝั่งละ 20"""
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, (doc_id, src) in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            docs[doc_id] = src
    return [(d, docs[d]) for d in sorted(scores, key=scores.get, reverse=True)]


def rerank(q, cands):
    """[P6] cross-encoder อ่าน (คำถาม, chunk) พร้อมกันแล้วให้คะแนนใหม่"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    scores = _reranker.predict([(q, f"{s['breadcrumb']}\n{s['content']}") for _, s in cands])
    return [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]


def relevance_filter(q, cands, threshold=0.2):
    """[P7] ให้ LLM ตัดสินตาม instruction ว่าเกี่ยวข้องไหม แล้วตัดที่ต่ำกว่า threshold ทิ้ง (แนวคิด jev-reranker)"""
    listing = "\n\n".join(f"[{i}] ({s['breadcrumb']})\n{s['content'][:600]}" for i, (_, s) in enumerate(cands))
    out = llm(f"""Instruction: Judge whether each passage contains information that helps answer the question.
A passage about a different disease, or only loosely on the topic, is NOT relevant.
Question: {q}

{listing}

Reply with JSON only: {{"<passage number>": <relevance 0.0-1.0>, ...}}""", temperature=0)
    try:
        # int(re.sub(...)) เพราะบางโมเดลตอบ key เป็น "[0]" แทน "0"
        rel = {int(re.sub(r"\D", "", k)): float(v) for k, v in json.loads(re.search(r"\{.*\}", out, re.S).group()).items()}
    except Exception:
        print(f"  (filter: อ่านคำตอบ LLM ไม่ได้ จึงไม่กรอง: {out[:80]!r})")  # บอกให้รู้ ไม่เงียบ
        return cands                                            # parse ไม่ได้ -> ไม่กรอง ดีกว่าทิ้งข้อมูล
    return [c for i, c in enumerate(cands) if rel.get(i, 1.0) >= threshold] or cands[:1]


def search(q, mode="filter", k=5):
    if mode == "bm25":
        return bm25(q)[:k]
    if mode == "dense":
        return dense(q)[:k]
    fused = rrf([bm25(q), dense(q)])
    if mode == "hybrid":
        return fused[:k]
    ranked = rerank(q, fused[:20])
    return ranked[:k] if mode == "rerank" else relevance_filter(q, ranked[:k])


def answer(q, mode="filter"):
    hits = search(q, mode)
    ctx = "\n\n".join(f"[{i}] แหล่ง: {s['source']} > {s['breadcrumb']}\n{s['content']}"
                      for i, (_, s) in enumerate(hits, 1))
    return hits, llm(f"""You are a question answering assistant. Answer in Thai using ONLY the context below.
Cite sources like [1]. If the context does not contain the answer, say so clearly instead of guessing.

Context:
{ctx}

Question: {q}""", temperature=0.3)


if __name__ == "__main__":
    if sys.argv[1:2] == ["ingest"]:
        ingest()
    elif sys.argv[1:2] == ["ask"] and len(sys.argv) > 2:
        hits, ans = answer(sys.argv[2], *(sys.argv[3:4] or ["filter"]))
        for i, (_, s) in enumerate(hits, 1):
            print(f"[{i}] {s['source']} > {s['breadcrumb']}\n    {s['content'][:100]!r}")
        print("\nA:", ans)
    else:
        print(__doc__)
