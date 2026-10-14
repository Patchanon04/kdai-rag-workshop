"""ตรวจว่าเครื่องพร้อมสำหรับเวิร์กช็อปหรือยัง

    python check_setup.py          # ตรวจทุกอย่าง ยกเว้นเรียก LLM จริง
    python check_setup.py --llm    # ตรวจด้วยว่า API key ใช้ได้ (เรียก LLM 1 ครั้ง)
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "steps"))
ok = True


def check(name, fn, hint):
    global ok
    try:
        detail = fn()
        print(f"  [ผ่าน]  {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:
        ok = False
        print(f"  [ไม่ผ่าน] {name}: {str(e)[:150]}\n           วิธีแก้: {hint}")


print("ตรวจความพร้อมของเครื่อง\n")

def python_ok():
    if sys.version_info < (3, 10):
        raise RuntimeError(f"เครื่องนี้เป็น {sys.version.split()[0]}")
    return sys.version.split()[0]


check("Python 3.10 ขึ้นไป", python_ok, "ติดตั้ง Python 3.10-3.12 จาก python.org แล้วรัน setup ใหม่")

for lib in ["opensearchpy", "sentence_transformers", "openai", "numpy", "pandas", "matplotlib", "langchain", "llama_index.core"]:
    check(f"library {lib}", lambda lib=lib: getattr(importlib.import_module(lib), "__version__", ""),
          "เปิด venv แล้วรัน: pip install -r requirements.txt")


def opensearch_ok():
    from common import opensearch
    c = opensearch()
    v = c.info()["version"]["number"]
    tokens = [t["token"] for t in c.indices.analyze(body={"analyzer": "thai", "text": "หญิงตั้งครรภ์"})["tokens"]]
    return f"version {v}, thai analyzer: {tokens}"


check("OpenSearch ที่ localhost:9200", opensearch_ok,
      "เปิด Docker Desktop แล้วรัน: docker compose up -d  (รอ ~30 วินาที)")


def model_ok(name):
    def f():
        from huggingface_hub import try_to_load_from_cache
        if try_to_load_from_cache(name, "config.json") is None:
            raise RuntimeError("ยังไม่ได้ดาวน์โหลด")
        return "อยู่ใน cache แล้ว"
    return f


check("embedding model BAAI/bge-m3", model_ok("BAAI/bge-m3"),
      "รัน setup ใหม่ หรือ: python -c \"from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-m3')\"")
check("reranker BAAI/bge-reranker-v2-m3", model_ok("BAAI/bge-reranker-v2-m3"),
      "รัน setup ใหม่ หรือ: python -c \"from sentence_transformers import CrossEncoder as C; C('BAAI/bge-reranker-v2-m3')\"")


def env_ok():
    import common  # noqa: F401  (อ่าน .env)
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    if not key or "xxxx" in key:
        raise RuntimeError("ยังไม่ได้ใส่ API key ใน .env")
    return f"model = {common.LLM_MODEL}"


check("API key ในไฟล์ .env", env_ok, "เปิดไฟล์ .env แล้วใส่ key (ดู README บทที่ 1.6)")

if "--llm" in sys.argv:
    def llm_ok():
        from common import llm
        return f"LLM ตอบว่า {llm('ตอบตัวเลขอย่างเดียว: 2+3', max_tokens=500)!r}"
    check("เรียก LLM ได้จริง", llm_ok, "ตรวจ key / ชื่อโมเดล / โควตา ใน .env")

print("\nพร้อมเริ่มเวิร์กช็อป" if ok else "\nยังไม่พร้อม แก้ข้อที่ [ไม่ผ่าน] ตามวิธีแก้ แล้วรันไฟล์นี้ใหม่")
sys.exit(0 if ok else 1)
