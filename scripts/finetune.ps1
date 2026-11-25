[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidateSet('B', 'C')]
    [string]$Model,

    [ValidateRange(1, 1000)]
    [int]$Epochs = 5,

    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]*$')]
    [string]$RunName = 'finetune',

    [string]$Manifest = 'data/refcocog/training_bilingual.json',

    [switch]$RankingLoss
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
$dataset = Join-Path $root 'data/refcocog'
$manifestPath = if ([IO.Path]::IsPathRooted($Manifest)) { $Manifest } else { Join-Path $root $Manifest }
$model = $Model.ToUpperInvariant()
$output = Join-Path $root "runs/$RunName/$model"
$rankingArgs = @()
if ($RankingLoss) {
    if ($model -ne 'C') {
        throw '-RankingLoss is only supported for Model C.'
    }
    $rankingArgs = @('--ranking-loss')
}

if (!(Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'The project virtual environment is missing. See README.md for setup.'
}
if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "The bilingual manifest is missing at $manifestPath. See README.md for dataset preparation."
}
if (Test-Path -LiteralPath $output) {
    throw "Run directory already exists at $output. Choose a different -RunName."
}
if (!$PSCmdlet.ShouldProcess($output, "Fine-tune model $model with an epoch limit of $Epochs")) {
    return
}

$module = if ($model -eq 'B') { 'scripts.train_translate' } else { 'scripts.train_tahrirchi' }
Push-Location -LiteralPath $root
try {
    Write-Host "Starting Model $model. Training log: $(Join-Path $output 'training.jsonl')"
    & $python -u -m $module --root $dataset --manifest $manifestPath --out $output --epochs $Epochs @rankingArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
