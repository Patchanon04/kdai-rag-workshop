"""Step 2: Embedding คืออะไร — ข้อความความหมายใกล้กัน vector ใกล้กัน (วัดด้วย cosine similarity)

    python steps/step02_embedding.py
"""
import numpy as np

from common import embedder

sentences = [
    "หญิงตั้งครรภ์ติดเชื้อหัดเยอรมัน",      # 0
    "คนท้องเป็นโรคหัดเยอรมัน",              # 1  ความหมายเดียวกับ 0 แต่คำต่างกันเกือบหมด
    "ผู้ป่วยอหิวาตกโรคถ่ายเป็นน้ำซาวข้าว",   # 2  คนละเรื่อง
    "pregnant woman infected with rubella",  # 3  ความหมายเดียวกับ 0 แต่เป็นภาษาอังกฤษ
]

vecs = embedder().encode(sentences, normalize_embeddings=True)
print("vector shape:", vecs.shape)          # (4, 1024) — แต่ละประโยคกลายเป็นตัวเลข 1024 ตัว
print("10 ตัวแรกของประโยค 0:", np.round(vecs[0][:10], 3))

# vector ถูก normalize (ยาว = 1) แล้ว cosine similarity = dot product
sim = vecs @ vecs.T
print("\ncosine similarity (1 = เหมือนกันมาก)")
for i, s in enumerate(sentences):
    print(f"[{i}] " + "  ".join(f"{sim[i, j]:.2f}" for j in range(len(sentences))) + f"   {s}")

# สังเกต: 0-1 และ 0-3 สูง ทั้งที่แทบไม่มีคำซ้ำกัน, 0-2 ต่ำ
# = การค้นแบบ dense/semantic หาเจอแม้ผู้ใช้ใช้คำต่างจากเอกสาร (สิ่งที่ SQL WHERE / lexical ทำไม่ได้)
