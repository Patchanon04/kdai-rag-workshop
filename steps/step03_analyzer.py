"""Step 3: Analyzer — OpenSearch ตัดข้อความเป็น "term" ก่อนเก็บใน inverted index (ต้องเปิด OpenSearch ก่อน)

    python steps/step03_analyzer.py
"""
from common import opensearch

client = opensearch()
print("OpenSearch version:", client.info()["version"]["number"])

texts = [
    "Rubella is a contagious viral infection",
    "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์",
    "ภาวะแทรกซ้อนในหญิงตั้งครรภ์ช่วงไตรมาสแรก",
]
for text in texts:
    print("\n" + text)
    for analyzer in ["standard", "thai"]:
        tokens = client.indices.analyze(body={"analyzer": analyzer, "text": text})["tokens"]
        print(f"  {analyzer:8} -> {[t['token'] for t in tokens]}")

# ภาษาอังกฤษ: standard ตัดได้ดีเพราะมีช่องว่างคั่นคำ
# ภาษาไทย: standard ได้ทั้งประโยคเป็น term เดียว -> BM25 จะ match ก็ต่อเมื่อพิมพ์ตรงทั้งประโยค
#          thai ใช้พจนานุกรมตัดเป็นคำ (และทิ้ง stopword อย่าง "จึง", "มี", "กับ")
# ระบบของครูใช้ standard  ->  จำผลของ step นี้ไว้ จะกลับมาใช้ตอนหาจุดอ่อน
