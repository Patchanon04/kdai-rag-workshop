# สคริปต์สร้าง notebook และผังภาพรวม (สำหรับผู้สอนที่จะแก้เนื้อหา)

แก้ข้อความในไฟล์ `_build_0X.py` แล้วรันเพื่อสร้าง notebook ใหม่ จากนั้นรัน notebook เพื่อเก็บ output จริง:

```bash
.venv\Scripts\python docs/notebook_src/_build_01.py
```

```bash
.venv\Scripts\jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=rag-workshop notebooks/01_concepts.ipynb
```

ผู้เรียนไม่ต้องใช้โฟลเดอร์นี้
