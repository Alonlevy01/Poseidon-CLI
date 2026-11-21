<#
.SYNOPSIS
    Launch Poseidon-CLI inside Docker mounted on your current working directory (just like `claude`).
.EXAMPLE
    # Add to your PowerShell $PROFILE to run `poseidon` from any project folder:
    Set-Alias poseidon "c:\Users\alonl\Desktop\Projects\assignment2\Poseidon-CLI\scripts\poseidon.ps1"
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PoseidonArgs
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $RepoRoot ".env"
$WorkDir = (Get-Location).Path

$DockerArgs = @(
    "run", "--rm", "-it",
    "--group-add", "0",
    "-v", "${WorkDir}:/workspace",
    "-v", "/var/run/docker.sock:/var/run/docker.sock",
    "-v", "poseidon-cli_poseidon_model_cache:/home/poseidon/.cache/fastembed",
    "-w", "/workspace"
)

if (Test-Path $EnvFile) {
    $DockerArgs += @("--env-file", $EnvFile)
}

$DockerArgs += "poseidon-cli:dev"
if ($PoseidonArgs) {
    $DockerArgs += $PoseidonArgs
}

& docker @DockerArgs
exit $LASTEXITCODE
