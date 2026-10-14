# เตรียมเครื่องสำหรับเวิร์กช็อป RAG (Windows) — รันครั้งเดียว รันซ้ำได้ไม่พัง
#   คลิกขวาที่โฟลเดอร์ rag-workshop > Open in Terminal แล้วพิมพ์:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Step($n, $msg) { Write-Host "`n[$n/7] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nหยุด: $msg" -ForegroundColor Red; exit 1 }

# 1) หา Python 3.10-3.12 (ใช้ py launcher ก่อน เพราะเลือกรุ่นได้)
Step 1 "หา Python 3.10-3.12"
$pyExe = $null; $pyArg = @()
foreach ($v in "3.11", "3.12", "3.10") {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArg = @("-$v"); break } } catch {}
}
if (-not $pyExe) {
    try { $ok = & python -c "import sys; print((3,10) <= sys.version_info[:2] <= (3,12))"; if ($ok -eq "True") { $pyExe = "python" } } catch {}
}
if (-not $pyExe) { Fail "ไม่พบ Python 3.10-3.12 ติดตั้งจาก python.org (ติ๊ก Add python.exe to PATH) แล้วรันใหม่" }
Write-Host "  ใช้: $pyExe $pyArg"

# 2) venv (มีแล้วข้าม)
Step 2 "สร้าง virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $pyExe @pyArg -m venv .venv } else { Write-Host "  มีอยู่แล้ว ข้าม" }
$vpy = ".\.venv\Scripts\python.exe"

# 3) library (~2-3 GB ครั้งแรก เพราะมี PyTorch)
Step 3 "ติดตั้ง library (ครั้งแรกใช้เวลา 5-15 นาที)"
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install ไม่สำเร็จ ดู error ด้านบน (มักเป็นเน็ตหลุด รันใหม่ได้เลย)" }

# 4) ลงทะเบียน kernel ให้ Jupyter / VS Code เห็น venv นี้
Step 4 "ลงทะเบียน Jupyter kernel 'RAG Workshop'"
& $vpy -m ipykernel install --user --name rag-workshop --display-name "RAG Workshop (.venv)" | Out-Null

# 5) OpenSearch: ถ้ามีตัวที่รันอยู่แล้วที่ port 9200 ใช้ตัวนั้นเลย
Step 5 "เปิด OpenSearch (Docker)"
function OsUp { try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 http://localhost:9200).StatusCode -eq 200 } catch { $false } }
if (OsUp) { Write-Host "  มี OpenSearch รันอยู่ที่ localhost:9200 แล้ว ใช้ตัวนั้น" }
else {
    $global:LASTEXITCODE = 1
    try { docker info *> $null } catch {}
    if ($LASTEXITCODE -ne 0) { Fail "Docker ยังไม่ทำงาน เปิดโปรแกรม Docker Desktop รอจนขึ้น Engine running แล้วรัน setup ใหม่" }
    docker compose up -d
    Write-Host "  รอ OpenSearch เริ่ม (สูงสุด 2 นาที)" -NoNewline
    for ($i = 0; $i -lt 40 -and -not (OsUp); $i++) { Start-Sleep 3; Write-Host "." -NoNewline }
    if (-not (OsUp)) { Fail "OpenSearch ไม่ขึ้น ดูสาเหตุด้วย: docker logs rag-workshop-opensearch --tail 30" }
    Write-Host " พร้อม"
}

# 6) ดาวน์โหลดโมเดล (~2.3 GB + ~2.3 GB ครั้งแรก เก็บใน %USERPROFILE%\.cache\huggingface)
Step 6 "ดาวน์โหลด embedding model และ reranker (ครั้งแรก ~4.5 GB)"
& $vpy -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3'); print('  โหลดครบ')"
if ($LASTEXITCODE -ne 0) { Fail "ดาวน์โหลดโมเดลไม่สำเร็จ (เน็ต/พื้นที่ดิสก์) รันใหม่ได้ จะโหลดต่อจากเดิม" }

# 7) .env + ตรวจรวม
Step 7 "เตรียมไฟล์ .env และตรวจความพร้อม"
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "  สร้าง .env แล้ว -> เปิดไฟล์ .env ใส่ API key ของตัวเอง" -ForegroundColor Yellow }
& $vpy check_setup.py
Write-Host "`nขั้นต่อไป: เปิดโฟลเดอร์นี้ใน VS Code > เปิด notebooks/01_concepts.ipynb > เลือก kernel 'RAG Workshop (.venv)'" -ForegroundColor Green
