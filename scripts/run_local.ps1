# PowerShell version of run_local.sh for Windows

$ErrorActionPreference = "Stop"

Write-Host "Starting Qwen Megakernel Services..." -ForegroundColor Green

$REPO_ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $REPO_ROOT

function Test-Port {
    param([int]$Port)
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    return $connection.TcpTestSucceeded
}

$jobs = @()

# Start LLM service
if (-not (Test-Port -Port 8000)) {
    Write-Host "Starting LLM service on port 8000..." -ForegroundColor Green
    $llmJob = Start-Job -ScriptBlock {
        Set-Location $using:REPO_ROOT\services\llm_megakernel
        python server.py
    }
    $jobs += $llmJob
    Start-Sleep -Seconds 2
} else {
    Write-Host "Port 8000 is already in use. Skipping LLM service." -ForegroundColor Yellow
}

# Start TTS service
if (-not (Test-Port -Port 8001)) {
    Write-Host "Starting TTS service on port 8001..." -ForegroundColor Green
    $ttsJob = Start-Job -ScriptBlock {
        Set-Location $using:REPO_ROOT\services\tts_qwen3
        python server.py
    }
    $jobs += $ttsJob
    Start-Sleep -Seconds 2
} else {
    Write-Host "Port 8001 is already in use. Skipping TTS service." -ForegroundColor Yellow
}

# Wait for services
Write-Host "Waiting for services to start..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Run demo
Write-Host "Running Pipecat demo..." -ForegroundColor Green
Set-Location "$REPO_ROOT\pipecat_demo"
python app.py

# Cleanup
Write-Host "`nShutting down services..." -ForegroundColor Yellow
$jobs | Stop-Job
$jobs | Remove-Job
Write-Host "Done" -ForegroundColor Green
