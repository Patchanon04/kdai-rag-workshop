"""Step 8: วัดผล retrieval ของระบบครู vs ระบบปรับปรุง ด้วยชุดคำถาม steps/questions.json

chunk ถือว่า "ถูก" ถ้ามีข้อความ evidence (คัดลอกตรงจากเอกสาร) อยู่ข้างใน
-> ไม่ผูกกับวิธีตัด chunk จึงเทียบสองระบบที่ตัด chunk ต่างกันได้

    python steps/step08_evaluate.py          # ไม่รวมโหมด filter (ไม่เรียก LLM)
    python steps/step08_evaluate.py filter   # รวมโหมด filter ด้วย (เรียก LLM ข้อละ 1 ครั้ง)
"""
import json
import sys
from pathlib import Path

import step07_improved as imp
from common import ROOT, embedder, opensearch

QUESTIONS = json.loads((Path(__file__).with_name("questions.json")).read_text(encoding="utf-8"))
K = 5


# --- ระบบของครู: query แบบเดียวกับ onlysearchAuthenticRAG.py (ค้นฝั่งละ 10, RRF rank เริ่ม 0)
def t_bm25(q, k=10):
    hits = opensearch().search(index="anthropic-bm25-index", body={"size": k, "query": {"multi_match": {
        "query": q, "fields": ["content", "contextualized_content"], "type": "best_fields"}}})["hits"]["hits"]
    return [(h["_id"], h["_source"]["content"]) for h in hits]


def t_dense(q, k=10):
    v = embedder().encode(q).tolist()
    hits = opensearch().search(index="anthropic-vector-index", body={
        "size": k, "query": {"knn": {"embedding": {"vector": v, "k": k}}}})["hits"]["hits"]
    return [(h["_id"], h["_source"]["content"]) for h in hits]


def t_hybrid(q):
    scores, text = {}, {}
    for ranking in (t_bm25(q), t_dense(q)):
        for rank, (d, c) in enumerate(ranking):
            scores[d] = scores.get(d, 0) + 1 / (60 + rank)
            text[d] = c
    return [text[d] for d in sorted(scores, key=scores.get, reverse=True)]


SYSTEMS = {
    "ครู: BM25": lambda q: [c for _, c in t_bm25(q)],
    "ครู: dense": lambda q: [c for _, c in t_dense(q)],
    "ครู: hybrid (ระบบเดิม)": t_hybrid,
    **{f"ใหม่: {m}": (lambda m: lambda q: [s["content"] for _, s in imp.search(q, m)])(m)
       for m in ["bm25", "dense", "hybrid", "rerank"]},
}
SYSTEMS["ใหม่: filter (ระบบใหม่)"] = lambda q: [s["content"] for _, s in imp.search(q, "filter")]


def first_hit(texts, evidence):
    """อันดับแรก (1..K) ที่มี evidence อยู่ใน chunk หรือ None ถ้าไม่เจอใน top-K"""
    for rank, t in enumerate(texts[:K], 1):
        if any(e in t for e in evidence):
            return rank
    return None


def evaluate(run, questions=None):
    """วัดระบบเดียว: run(คำถาม) -> list ข้อความ chunk เรียงตามอันดับ
    คืน (ตัวชี้วัด, อันดับรายข้อ) — notebook เรียกฟังก์ชันนี้ตรง ๆ ได้"""
    labelled = [q for q in (questions or QUESTIONS) if q["evidence"]]
    ranks, chars = {}, []
    for q in labelled:
        got = run(q["question"])[:K]
        ranks[q["id"]] = first_hit(got, q["evidence"])
        chars.append(sum(map(len, got)))
    r, n = list(ranks.values()), len(labelled)
    return {"Hit@1": sum(x == 1 for x in r) / n, "Hit@5": sum(x is not None for x in r) / n,
            "MRR@5": sum(1 / x for x in r if x) / n, "chars": sum(chars) / n}, ranks


def main(argv):
    systems = dict(SYSTEMS)
    if "filter" not in argv[0:1]:                  # ไม่ใส่ filter = ไม่วัดชั้นที่ต้องเรียก LLM
        systems.pop("ใหม่: filter (ระบบใหม่)")
    if argv[1:]:  # เลือกเฉพาะบางแถว เช่น  python steps/step08_evaluate.py x teacher   หรือ   ... filter filter
        # ponytail: alias ภาษาอังกฤษ เพราะบาง terminal บน Windows ส่งอาร์กิวเมนต์ภาษาไทยไปเป็น ????
        pats = [{"teacher": "ครู", "new": "ใหม่"}.get(p, p) for p in argv[1:]]
        systems = {n: f for n, f in systems.items() if any(p in n for p in pats)}
    rows = []
    for name, run in systems.items():
        m, _ = evaluate(run)
        rows.append((name, m))
        print(f"{name:28} Hit@1={m['Hit@1']:.2f} Hit@5={m['Hit@5']:.2f} MRR@5={m['MRR@5']:.2f} chars={m['chars']:,.0f}")
    n = sum(1 for q in QUESTIONS if q["evidence"])
    md = [f"# Retrieval metrics ({n} คำถาม, top-{K})\n",
          "| ระบบ | Hit@1 | Hit@5 | MRR@5 | ตัวอักษรที่ส่งให้ LLM |", "|---|---|---|---|---|"]
    md += [f"| {name} | {m['Hit@1']:.2f} | {m['Hit@5']:.2f} | {m['MRR@5']:.2f} | {m['chars']:,.0f} |" for name, m in rows]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nบันทึกตารางที่ results/metrics.md")


if __name__ == "__main__":
    main(sys.argv[1:])
