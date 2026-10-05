param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArguments)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $projectRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { throw "Project runtime missing: $runtime" }
$oldLocation = Get-Location
$saved = @{}
foreach ($key in @('PATH','TORCH_HOME','TORCH_EXTENSIONS_DIR','MAX_JOBS','PYTHONIOENCODING')) {
    $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
try {
    Set-Location -LiteralPath $projectRoot
    $env:PATH = (Join-Path $projectRoot '.venv-gpu\Scripts') + ';' + $env:PATH
    $env:TORCH_HOME = Join-Path $projectRoot 'outputs\torch_cache'
    $env:TORCH_EXTENSIONS_DIR = Join-Path $projectRoot 'outputs\torch_extensions'
    $env:MAX_JOBS = '4'
    $env:PYTHONIOENCODING = 'utf-8'
    if (-not $CliArguments) { $CliArguments = @('doctor','--check-upstream') }
    & $runtime -m splatad_drive @CliArguments
    $result = $LASTEXITCODE
} finally {
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    Set-Location -LiteralPath $oldLocation
}
exit $result
