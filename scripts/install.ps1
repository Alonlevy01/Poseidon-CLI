<#
.SYNOPSIS
    Install (or uninstall) global `poseidon` command shims into %USERPROFILE%\.local\bin
    so `poseidon` and `poseidon ui` work from EVERY directory and terminal (PowerShell, CMD, Git Bash).

.EXAMPLES
    .\scripts\install.ps1              # Install globally
    .\scripts\install.ps1 -Uninstall   # Remove global shims
#>
param(
    [switch]$Uninstall
)

$BinDir = Join-Path $env:USERPROFILE ".local\bin"
$CmdShim = Join-Path $BinDir "poseidon.cmd"
$ShShim = Join-Path $BinDir "poseidon"
$LauncherScript = Join-Path $PSScriptRoot "poseidon.ps1"

if ($Uninstall) {
    foreach ($file in @($CmdShim, $ShShim)) {
        if (Test-Path $file) {
            Remove-Item -Force $file
            Write-Host "Removed $file" -ForegroundColor Yellow
        }
    }
    Write-Host "Poseidon global shims removed." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
}

# 1. Windows CMD + PowerShell shim (works everywhere on PATH without ExecutionPolicy friction)
$CmdContent = @"
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$LauncherScript" %*
"@
Set-Content -Path $CmdShim -Value $CmdContent -Encoding Ascii

# 2. POSIX shell shim (so Git Bash / WSL-adjacent shells also find `poseidon`)
$ShContent = @"
#!/usr/bin/env sh
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$LauncherScript" "`$@"
"@
Set-Content -Path $ShShim -Value $ShContent -Encoding Ascii -NoNewline

# 3. Ensure %USERPROFILE%\.local\bin is on the User PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$PathEntries = @()
if ($UserPath) {
    $PathEntries = $UserPath -split ";" | Where-Object { $_ -and $_.Trim() -ne "" }
}
$AlreadyOnPath = $PathEntries | Where-Object { $_.TrimEnd("\") -ieq $BinDir.TrimEnd("\") }
if (-not $AlreadyOnPath) {
    $NewUserPath = if ($UserPath) { "$BinDir;$UserPath" } else { $BinDir }
    [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
    if ($env:Path -notlike "*$BinDir*") {
        $env:Path = "$BinDir;$env:Path"
    }
    Write-Host "Added $BinDir to your User PATH." -ForegroundColor Cyan
}

Write-Host "Installed global Poseidon command:" -ForegroundColor Green
Write-Host "  $CmdShim" -ForegroundColor Cyan
Write-Host "  $ShShim" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run 'poseidon' or 'poseidon ui -d' from ANY directory and ANY terminal!" -ForegroundColor Green
