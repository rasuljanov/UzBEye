param([ValidateRange(1,65535)][int]$Port = 8000)
$root = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $root
try {
    $python = Join-Path $root '.venv\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $python)) { throw 'Set up the project virtual environment from README.md.' }
    if (!(Test-Path -LiteralPath 'frontend/dist/index.html')) { throw 'Build the Vue frontend with npm run build in frontend.' }
    & $python -m uvicorn src.api:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
