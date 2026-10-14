# เวิร์กช็อป RAG — คำสั่งเดียว: โหลดเวิร์กช็อป + repo ของครูเอ้ ไปไว้ที่ Desktop แล้วรัน setup ต่อให้
# เปิด PowerShell แล้ววางบรรทัดนี้:
#   irm https://raw.githubusercontent.com/Patchanon04/kdai-rag-workshop/main/install.ps1 | iex
$ErrorActionPreference = "Stop"
$Repo    = "Patchanon04/kdai-rag-workshop"
$Teacher = "aekanun2020/2025-authenticRAG"
# Desktop จริงของเครื่อง (รองรับกรณี OneDrive ย้าย Desktop ไปไว้ที่อื่น)
$Dest = if ($env:RAG_WORKSHOP_DIR) { $env:RAG_WORKSHOP_DIR } else { Join-Path ([Environment]::GetFolderPath("Desktop")) "rag-workshop" }

function Get-Repo($slug, $target) {
    $hasGit = [bool](Get-Command git -ErrorAction SilentlyContinue)
    if (Test-Path $target) {
        if ($hasGit -and (Test-Path (Join-Path $target ".git"))) {
            Write-Host "  มีอยู่แล้ว อัปเดตเป็นรุ่นล่าสุด: $target"
            git -C $target pull --ff-only
        } else {
            Write-Host "  มีโฟลเดอร์ $target อยู่แล้ว ข้าม (ไม่เขียนทับ)" -ForegroundColor Yellow
        }
        return
    }
    if ($hasGit) {
        git clone --depth 1 "https://github.com/$slug.git" $target
    } else {
        # ไม่มี git: โหลดเป็น zip แทน
        $zip = Join-Path $env:TEMP "rag-$([guid]::NewGuid()).zip"
        $tmp = Join-Path $env:TEMP "rag-$([guid]::NewGuid())"
        Invoke-WebRequest -UseBasicParsing "https://github.com/$slug/archive/refs/heads/main.zip" -OutFile $zip
        Expand-Archive $zip $tmp
        Move-Item (Get-ChildItem $tmp | Select-Object -First 1).FullName $target
        Remove-Item $zip, $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "[1/3] โหลดเวิร์กช็อปไปที่ $Dest" -ForegroundColor Cyan
Get-Repo $Repo $Dest

Write-Host "[2/3] โหลดโค้ดของครูเอ้ ($Teacher) ไปที่ $Dest\teacher" -ForegroundColor Cyan
Get-Repo $Teacher (Join-Path $Dest "teacher")

Write-Host "[3/3] รัน setup (ติดตั้ง library, OpenSearch, โมเดล)" -ForegroundColor Cyan
if ($env:RAG_SKIP_SETUP) { Write-Host "  ข้าม (RAG_SKIP_SETUP)"; return }
Set-Location $Dest
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Dest "setup.ps1")
