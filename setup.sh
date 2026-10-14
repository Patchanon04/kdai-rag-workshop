#!/usr/bin/env bash
# เตรียมเครื่องสำหรับเวิร์กช็อป RAG (macOS / Linux) — รันครั้งเดียว รันซ้ำได้ไม่พัง
#   bash setup.sh
set -e
cd "$(dirname "$0")"
step() { printf "\n\033[36m[%s/7] %s\033[0m\n" "$1" "$2"; }
fail() { printf "\n\033[31mหยุด: %s\033[0m\n" "$1"; exit 1; }

step 1 "หา Python 3.10-3.12"
PY=""
for c in python3.11 python3.12 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then PY="$c"; break; fi
done
[ -n "$PY" ] || fail "ไม่พบ Python 3.10-3.12 (macOS: brew install python@3.11)"
echo "  ใช้: $PY"

step 2 "สร้าง virtual environment (.venv)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY=.venv/bin/python

step 3 "ติดตั้ง library (ครั้งแรกใช้เวลา 5-15 นาที)"
$VPY -m pip install --upgrade pip --quiet
# Linux: ใช้ PyTorch แบบ CPU (ไม่งั้นได้รุ่น CUDA ขนาดหลาย GB ที่เวิร์กช็อปนี้ไม่ได้ใช้)
if [ "$(uname)" = "Linux" ]; then $VPY -m pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
$VPY -m pip install -r requirements.txt || fail "pip install ไม่สำเร็จ ดู error ด้านบน"

step 4 "ลงทะเบียน Jupyter kernel 'RAG Workshop'"
$VPY -m ipykernel install --user --name rag-workshop --display-name "RAG Workshop (.venv)" >/dev/null

step 5 "เปิด OpenSearch (Docker)"
if curl -sf localhost:9200 >/dev/null; then
  echo "  มี OpenSearch รันอยู่ที่ localhost:9200 แล้ว ใช้ตัวนั้น"
else
  docker info >/dev/null 2>&1 || fail "Docker ยังไม่ทำงาน เปิด Docker Desktop แล้วรัน setup ใหม่"
  docker compose up -d
  printf "  รอ OpenSearch เริ่ม (สูงสุด 2 นาที)"
  for i in $(seq 40); do curl -sf localhost:9200 >/dev/null && break; sleep 3; printf "."; done
  curl -sf localhost:9200 >/dev/null || fail "OpenSearch ไม่ขึ้น: docker logs rag-workshop-opensearch --tail 30"
  echo " พร้อม"
fi

step 6 "ดาวน์โหลด embedding model และ reranker (ครั้งแรก ~4.5 GB)"
$VPY -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3'); print('  โหลดครบ')"

step 7 "เตรียมไฟล์ .env และตรวจความพร้อม"
[ -f .env ] || { cp .env.example .env; echo "  สร้าง .env แล้ว -> เปิดไฟล์ .env ใส่ API key ของตัวเอง"; }
$VPY check_setup.py || true
printf "\n\033[32mขั้นต่อไป: เปิดโฟลเดอร์นี้ใน VS Code > เปิด notebooks/01_concepts.ipynb > เลือก kernel 'RAG Workshop (.venv)'\033[0m\n"
