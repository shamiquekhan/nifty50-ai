Write-Host "============================================================" -ForegroundColor Red
Write-Host " NIFTY50 AUTO-UPDATE SYSTEM" -ForegroundColor Red
Write-Host " Starting automated data updates and model fine-tuning..." -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Red
Write-Host ""

python src\auto_update.py
