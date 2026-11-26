<#
.SYNOPSIS
    Launch Poseidon-CLI or Poseidon Web UI natively (zero Docker required for CLI/Web UI).
    Docker is used only when running isolated `--network none` unit tests (`run_tests` / `poseidon test`).

.EXAMPLES
    poseidon                          # Launch interactive CLI in current folder
    poseidon --plan                   # Launch CLI in read-only Plan Mode
    poseidon ui                       # Launch Web UI in foreground on http://localhost:2701
    poseidon ui -d                    # Launch Web UI detached (terminal stays free!)
    poseidon ui -d --port 2702        # Launch a second detached Web UI on port 2702
    poseidon ui stop                  # Stop detached native Poseidon Web UI processes
    poseidon doctor                   # Check environment & provider health
#>
[string[]]$PoseidonArgs = @($args)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvExe = Join-Path $RepoRoot ".venv\Scripts\poseidon.exe"
$EnvFile = Join-Path $RepoRoot ".env"
$WorkDir = (Get-Location).Path

# Ensure native virtualenv exists
if (-not (Test-Path $VenvExe)) {
    Write-Host "Poseidon native virtualenv not found at $VenvExe. Running 'uv sync --extra dev'..." -ForegroundColor Cyan
    Push-Location $RepoRoot
    try {
        uv sync --extra dev
    }
    finally {
        Pop-Location
    }
    if (-not (Test-Path $VenvExe)) {
        Write-Host "Error: Could not find or build $VenvExe. Run 'uv sync --extra dev' inside $RepoRoot." -ForegroundColor Red
        exit 1
    }
}

# Load API keys from repo .env into process env (without overwriting already-set env vars)
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $idx = $line.IndexOf("=")
            $k = $line.Substring(0, $idx).Trim()
            $v = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
            if ($k -and -not [Environment]::GetEnvironmentVariable($k, "Process")) {
                # Auto-fix legacy Docker host.docker.internal URLs when running natively
                if (($k -eq "OLLAMA_BASE_URL" -or $k -eq "LMSTUDIO_BASE_URL") -and $v -like "*host.docker.internal*") {
                    $v = $v -replace "host\.docker\.internal", "localhost"
                }
                [Environment]::SetEnvironmentVariable($k, $v, "Process")
            }
        }
    }
}

function Get-PoseidonUiPort {
    param([string[]]$ArgList)
    $port = 2701
    for ($i = 0; $i -lt $ArgList.Count; $i++) {
        $arg = $ArgList[$i]
        if (($arg -eq "--port" -or $arg -eq "-p") -and ($i + 1) -lt $ArgList.Count) {
            if ($ArgList[$i + 1] -match '^\d+$') {
                $port = [int]$ArgList[$i + 1]
            }
        }
        elseif ($arg -match '^--port=(\d+)$') {
            $port = [int]$Matches[1]
        }
    }
    return $port
}

# Handle `poseidon ui ...` lifecycle commands natively
if ($PoseidonArgs -and $PoseidonArgs.Count -gt 0 -and $PoseidonArgs[0] -eq "ui") {
    $SubArgs = @()
    if ($PoseidonArgs.Count -gt 1) {
        $SubArgs = $PoseidonArgs[1..($PoseidonArgs.Count - 1)]
    }

    if ($SubArgs.Count -gt 0 -and $SubArgs[0] -eq "stop") {
        $procs = Get-CimInstance Win32_Process -Filter "Name='poseidon.exe' OR Name='python.exe'" |
            Where-Object { $_.CommandLine -like "*poseidon*ui*" }
        if (-not $procs) {
            Write-Host "No running native Poseidon Web UI processes found." -ForegroundColor DarkGray
            exit 0
        }
        foreach ($p in $procs) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped Poseidon Web UI (PID $($p.ProcessId))." -ForegroundColor Green
        }
        exit 0
    }

    $Detach = $false
    $NoOpen = $false
    $HasWorkspace = $false
    $ForwardArgs = @("ui")
    $Port = Get-PoseidonUiPort -ArgList $SubArgs

    foreach ($arg in $SubArgs) {
        if ($arg -eq "-d" -or $arg -eq "--detach") {
            $Detach = $true
            continue
        }
        if ($arg -eq "--no-open") {
            $NoOpen = $true
            continue
        }
        if ($arg -eq "--workspace" -or $arg -eq "-w" -or $arg -like "--workspace=*") {
            $HasWorkspace = $true
        }
        $ForwardArgs += $arg
    }

    if (-not $HasWorkspace -and -not ($SubArgs -contains "--help") -and -not ($SubArgs -contains "-h")) {
        $ForwardArgs += @("--workspace", $WorkDir)
    }

    if ($Detach) {
        Start-Process -FilePath $VenvExe -ArgumentList $ForwardArgs -WindowStyle Hidden | Out-Null
        $url = "http://localhost:$Port"
        Write-Host "Poseidon Web UI started natively in background: $url" -ForegroundColor Green
        Write-Host "  Workspace: $WorkDir" -ForegroundColor Cyan
        Write-Host "  Stop with: poseidon ui stop" -ForegroundColor DarkGray
        if (-not $NoOpen) {
            Start-Sleep -Milliseconds 900
            Start-Process $url
        }
        exit 0
    }

    & $VenvExe @ForwardArgs
    exit $LASTEXITCODE
}

# Standard native CLI execution (pass current folder as --workspace if not specified on chat/root)
& $VenvExe @PoseidonArgs
exit $LASTEXITCODE
