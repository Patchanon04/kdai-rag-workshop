"""สร้างผังภาพรวม docs/overview_journey.svg และ docs/overview_pipeline.svg (สีตายตัว เปิดได้ทุกที่)"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent          # docs/

FONT = "Tahoma, 'Leelawadee UI', 'Noto Sans Thai', sans-serif"
RAMP = {  # fill, stroke, title, subtitle
    "gray": ("#F1EFE8", "#5F5E5A", "#444441", "#5F5E5A"),
    "purple": ("#EEEDFE", "#534AB7", "#3C3489", "#534AB7"),
    "coral": ("#FAECE7", "#993C1D", "#712B13", "#993C1D"),
    "teal": ("#E1F5EE", "#0F6E56", "#085041", "#0F6E56"),
}


def box(x, y, w, h, ramp, title, sub, dashed=False):
    f, s, t1, t2 = RAMP[ramp]
    d = ' stroke-dasharray="4 3"' if dashed else ""
    cx = x + w / 2
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{f}" stroke="{s}" stroke-width="0.8"{d}/>'
            f'<text x="{cx}" y="{y + h * 0.36}" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="bold" fill="{t1}">{title}</text>'
            f'<text x="{cx}" y="{y + h * 0.72}" text-anchor="middle" dominant-baseline="central" font-size="12" fill="{t2}">{sub}</text>')


def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#888780" stroke-width="1.3" marker-end="url(#a)"/>'


def svg(h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{h}" viewBox="0 0 680 {h}" font-family="{FONT}">'
            f'<title>{title}</title><rect width="680" height="{h}" fill="#ffffff"/>'
            '<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>'
            + body + "</svg>")


# ---------- 1. เส้นทางของเวิร์กช็อป
steps = [("gray", "Setup (บทที่ 0-1)", "setup.ps1 + ใส่ key ใน .env"),
         ("purple", "01_concepts (บทที่ 2-6)", "LLM, embedding, BM25, RRF, mini RAG"),
         ("purple", "02_detective (บทที่ 7-9)", "วัดระบบครู หาจุดอ่อน 6 จุด"),
         ("purple", "03_improve (บทที่ 10-11)", "แก้ P1-P7 ทีละขั้น + วัดผล"),
         ("gray", "README บทที่ 12-14", "เขียนข้อ 3, ส่ง GitHub, นำเสนอ")]
b = []
for i, (r, t, s) in enumerate(steps):
    y = 30 + i * 86
    b.append(box(60, y, 300, 56, r, t, s))
    if i < len(steps) - 1:
        b.append(arrow(210, y + 56, 210, y + 84))
for y, sub in [(202, "รันโค้ดครู ~15 นาที"), (288, "ingest ใหม่ + วัดผล ~30 นาที")]:
    b.append(box(420, y, 220, 56, "coral", "การบ้านก่อนคาบ", sub, dashed=True))
    b.append(arrow(418, y + 28, 364, y + 28))
OUT.joinpath("overview_journey.svg").write_text(svg(470, "".join(b), "เส้นทางของเวิร์กช็อป RAG"), encoding="utf-8")

# ---------- 2. pipeline ระบบครู vs ระบบใหม่
rows = [("ตัด chunk ตามหัวข้อ", "ยาวถึง 8,404 ตัวอักษร", "P1 หัวข้อ + ไม่เกิน 800", "overlap + breadcrumb"),
        ("LLM เขียน context", "เห็นแค่ chunk, ภาษาอังกฤษ", "P2 ให้เห็นสารบัญ + section", "ตอบไทย + คำพ้อง"),
        ("BM25 analyzer standard", "ทั้งประโยคไทยเป็นคำเดียว", "P3 analyzer thai", "BM25 Hit@5 0.07 → 0.89"),
        ("embed chunk + context", "chunk ใหญ่ vector เบลอ", "P4 breadcrumb+context+chunk", "dense Hit@1 0.68 → 0.79"),
        ("RRF ฝั่งละ 10", "rank เริ่ม 0 ผิดสูตร", "P5 RRF ฝั่งละ 20", "rank เริ่ม 1 ตามสูตร"),
        ("ไม่มี rerank", "Hit@1 0.64, MRR 0.78", "P6 cross-encoder rerank", "Hit@1 0.82, MRR 0.90"),
        ("ส่ง 5 chunk ให้ LLM", "~11,100 ตัว, prompt หลวม", "P7 filter + prompt ใหม่", "~1,700 ตัว + อ้างอิง [n]")]
b = ['<text x="175" y="40" text-anchor="middle" font-size="14" font-weight="bold" fill="#444441">ระบบครู (จุดอ่อน)</text>',
     '<text x="505" y="40" text-anchor="middle" font-size="14" font-weight="bold" fill="#444441">ระบบใหม่ (แก้ด้วย)</text>']
for i, (lt, ls, rt, rs) in enumerate(rows):
    y = 60 + i * 66
    b += [box(40, y, 270, 52, "coral", lt, ls), box(370, y, 270, 52, "teal", rt, rs), arrow(312, y + 26, 366, y + 26)]
b.append('<text x="340" y="532" text-anchor="middle" font-size="12" fill="#5F5E5A">LLM ตอบคำถาม: ตัวเดียวกันทั้งสองระบบ (ห้ามเปลี่ยนตามโจทย์)</text>')
OUT.joinpath("overview_pipeline.svg").write_text(svg(560, "".join(b), "Pipeline ระบบครูเทียบระบบใหม่"), encoding="utf-8")
print("wrote docs/overview_journey.svg, docs/overview_pipeline.svg")
