#!/usr/bin/env bash
# เวิร์กช็อป RAG (macOS / Linux) — คำสั่งเดียว: โหลดเวิร์กช็อป + repo ของครูเอ้ ไปไว้ที่ Desktop แล้วรัน setup
#   curl -fsSL https://raw.githubusercontent.com/Patchanon04/kdai-rag-workshop/main/install.sh | bash
set -e
REPO="Patchanon04/kdai-rag-workshop"
TEACHER="aekanun2020/2025-authenticRAG"
DEST="${RAG_WORKSHOP_DIR:-$HOME/Desktop/rag-workshop}"

# ponytail: Mac ที่ยังไม่ลง Xcode tools มีคำสั่ง git แต่รันไม่ได้ (เด้งให้ติดตั้ง) จึงเช็กด้วยการลองรันจริง
has_git() { git --version >/dev/null 2>&1; }

get_repo() {  # $1 = owner/name, $2 = โฟลเดอร์ปลายทาง
  if [ -d "$2" ]; then
    if [ -d "$2/.git" ] && has_git; then echo "  มีอยู่แล้ว อัปเดต: $2"; git -C "$2" pull --ff-only
    else echo "  มีโฟลเดอร์ $2 อยู่แล้ว ข้าม (ไม่เขียนทับ)"; fi
    return
  fi
  if has_git; then git clone --depth 1 "https://github.com/$1.git" "$2"
  else
    tmp="$(mktemp -d)"
    curl -fsSL "https://github.com/$1/archive/refs/heads/main.tar.gz" | tar -xz -C "$tmp"
    mv "$tmp"/* "$2"; rm -rf "$tmp"
  fi
}

echo "[1/3] โหลดเวิร์กช็อปไปที่ $DEST"; get_repo "$REPO" "$DEST"
echo "[2/3] โหลดโค้ดของครูเอ้ ไปที่ $DEST/teacher"; get_repo "$TEACHER" "$DEST/teacher"
echo "[3/3] รัน setup"
[ -n "$RAG_SKIP_SETUP" ] && { echo "  ข้าม (RAG_SKIP_SETUP)"; exit 0; }
cd "$DEST" && bash setup.sh
