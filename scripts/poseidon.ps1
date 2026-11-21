<#
.SYNOPSIS
    Launch Poseidon-CLI or Poseidon Web UI inside Docker mounted on your current directory.

.EXAMPLES
    poseidon                          # Launch interactive CLI in current folder
    poseidon --plan                   # Launch CLI in read-only Plan Mode
    poseidon ui                       # Launch Web UI in foreground on http://localhost:2701
    poseidon ui -d                    # Launch Web UI detached (terminal stays free!)
    poseidon ui -d --port 2702        # Launch a second detached Web UI on port 2702
    poseidon ui status                # Show all running Poseidon Web UI instances
    poseidon ui stop                  # Stop Web UI on default port 2701
    poseidon ui stop --port 2702      # Stop Web UI on port 2702
    poseidon ui stop --all            # Stop all running Poseidon Web UI instances
#>
# Note: intentionally uses automatic `$args` (no `[Parameter()]` block) so PowerShell
# does not hijack short flags like `-d` (`-Debug`), `-v`, or `-w` as common parameters.
[string[]]$PoseidonArgs = @($args)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $RepoRoot ".env"
$WorkDir = (Get-Location).Path

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

# Handle `poseidon ui ...` lifecycle commands (detach, multi-port, status, stop)
if ($PoseidonArgs -and $PoseidonArgs.Count -gt 0 -and $PoseidonArgs[0] -eq "ui") {
    $SubArgs = @()
    if ($PoseidonArgs.Count -gt 1) {
        $SubArgs = $PoseidonArgs[1..($PoseidonArgs.Count - 1)]
    }

    # 1. `poseidon ui status`
    if ($SubArgs.Count -gt 0 -and $SubArgs[0] -eq "status") {
        $rows = docker ps --filter "label=poseidon.ui=true" --format "{{.Names}}`t{{.Ports}}`t{{.Status}}"
        $composeRow = docker ps --filter "name=poseidon-cli-poseidon-ui" --format "{{.Names}}`t{{.Ports}}`t{{.Status}}"
        $allRows = @($rows) + @($composeRow) | Where-Object { $_ -and $_.Trim() -ne "" } | Select-Object -Unique
        if (-not $allRows -or $allRows.Count -eq 0) {
            Write-Host "No Poseidon Web UI containers are currently running." -ForegroundColor DarkGray
            exit 0
        }
        Write-Host "Running Poseidon Web UI instance(s):" -ForegroundColor Cyan
        foreach ($r in $allRows) {
            $parts = $r -split "`t"
            $urlHint = ""
            if ($parts[1] -match ':(\d+)->') {
                $urlHint = " -> http://localhost:$($Matches[1])"
            }
            Write-Host "  - $($parts[0])  ($($parts[2]))$urlHint" -ForegroundColor Green
        }
        exit 0
    }

    # 2. `poseidon ui stop [--port N | --all]`
    if ($SubArgs.Count -gt 0 -and $SubArgs[0] -eq "stop") {
        $stopAll = ($SubArgs -contains "--all") -or ($SubArgs -contains "-a")
        if ($stopAll) {
            $ids = @(docker ps -q --filter "label=poseidon.ui=true") + @(docker ps -q --filter "name=poseidon-cli-poseidon-ui")
            $ids = $ids | Where-Object { $_ -and $_.Trim() -ne "" } | Select-Object -Unique
            if (-not $ids -or $ids.Count -eq 0) {
                Write-Host "No running Poseidon Web UI containers found." -ForegroundColor DarkGray
                exit 0
            }
            docker stop @ids | Out-Null
            Write-Host "Stopped $($ids.Count) Poseidon Web UI container(s)." -ForegroundColor Green
            exit 0
        }

        $targetPort = Get-PoseidonUiPort -ArgList $SubArgs
        $containerName = "poseidon-ui-$targetPort"
        $found = docker ps -q --filter "name=^/${containerName}$"
        if ($found) {
            docker stop $containerName | Out-Null
            Write-Host "Stopped $containerName (port $targetPort)." -ForegroundColor Green
            exit 0
        }
        if ($targetPort -eq 2701) {
            $composeFound = docker ps -q --filter "name=^/poseidon-cli-poseidon-ui-1$"
            if ($composeFound) {
                docker stop poseidon-cli-poseidon-ui-1 | Out-Null
                Write-Host "Stopped poseidon-cli-poseidon-ui-1 (port 2701)." -ForegroundColor Green
                exit 0
            }
        }
        Write-Host "No running Poseidon Web UI found on port $targetPort." -ForegroundColor Yellow
        exit 1
    }

    # 3. `poseidon ui [-d|--detach] [--port N] [--keep-alive] [--no-open] ...`
    $Detach = $false
    $NoOpen = $false
    $ForwardArgs = @("ui")
    $Port = Get-PoseidonUiPort -ArgList $SubArgs
    $HasPortFlag = $false

    for ($i = 0; $i -lt $SubArgs.Count; $i++) {
        $arg = $SubArgs[$i]
        if ($arg -eq "-d" -or $arg -eq "--detach") {
            $Detach = $true
            continue
        }
        if ($arg -eq "--no-open") {
            $NoOpen = $true
            continue
        }
        if ($arg -eq "--port" -or $arg -eq "-p" -or $arg -like "--port=*") {
            $HasPortFlag = $true
        }
        $ForwardArgs += $arg
    }
    if (-not $HasPortFlag) {
        $ForwardArgs += @("--port", "$Port")
    }

    $ContainerName = "poseidon-ui-$Port"
    $DockerArgs = @("run", "--rm")
    if ($Detach) {
        $DockerArgs += "-d"
    } else {
        $DockerArgs += "-it"
    }
    $DockerArgs += @(
        "--name", $ContainerName,
        "--label", "poseidon.ui=true",
        "--label", "poseidon.port=$Port",
        "--label", "poseidon.workspace=$WorkDir",
        "--group-add", "0",
        "--add-host", "host.docker.internal:host-gateway",
        "-e", "POSEIDON_HOST_WORKSPACE=$WorkDir",
        "-p", "${Port}:${Port}",
        "-v", "${WorkDir}:/workspace",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", "poseidon-cli_poseidon_model_cache:/home/poseidon/.cache/fastembed",
        "-v", "poseidon-cli_poseidon_secrets:/home/poseidon/.poseidon_secrets",
        "-w", "/workspace"
    )
    if (Test-Path $EnvFile) {
        $DockerArgs += @("--env-file", $EnvFile)
    }
    $DockerArgs += "poseidon-cli:dev"
    $DockerArgs += $ForwardArgs

    if ($Detach) {
        $cid = & docker @DockerArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        $Url = "http://localhost:$Port"
        # Wait briefly until Uvicorn is listening before opening the browser
        for ($try = 0; $try -lt 24; $try++) {
            try {
                $resp = Invoke-WebRequest -Uri "$Url/api/state" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
                if ($resp.StatusCode -eq 200) { break }
            } catch {
                Start-Sleep -Milliseconds 250
            }
        }
        Write-Host "Poseidon Web UI started ($ContainerName)" -ForegroundColor Cyan
        Write-Host "  URL:       $Url" -ForegroundColor Green
        Write-Host "  Workspace: $WorkDir" -ForegroundColor DarkGray
        Write-Host "  Shutdown:  close the browser tab (auto-stops in 8s), click Stop, or run 'poseidon ui stop --port $Port'" -ForegroundColor DarkGray
        if (-not $NoOpen) {
            Start-Process $Url
        }
        exit 0
    }

    & docker @DockerArgs
    exit $LASTEXITCODE
}

# Standard CLI invocation (`poseidon`, `poseidon -p "..."`, `git diff | poseidon -p "..."`)
$TtyFlags = if ([Console]::IsInputRedirected) { @("-i") } else { @("-it") }
$DockerArgs = @("run", "--rm") + $TtyFlags + @(
    "--group-add", "0",
    "--add-host", "host.docker.internal:host-gateway",
    "-e", "POSEIDON_HOST_WORKSPACE=$WorkDir",
    "-v", "${WorkDir}:/workspace",
    "-v", "/var/run/docker.sock:/var/run/docker.sock",
    "-v", "poseidon-cli_poseidon_model_cache:/home/poseidon/.cache/fastembed",
    "-v", "poseidon-cli_poseidon_secrets:/home/poseidon/.poseidon_secrets",
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
