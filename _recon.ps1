# Recon for environment migration
$ErrorActionPreference = 'SilentlyContinue'

function Show-DirSize([string]$Path, [string]$Label) {
    if (Test-Path $Path) {
        $s = (Get-ChildItem $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
              Measure-Object -Property Length -Sum).Sum
        Write-Output ("{0,-14} {1,10:N2} GB   {2}" -f $Label, ($s / 1GB), $Path)
    } else {
        Write-Output ("{0,-14} {1,10}      {2}" -f $Label, 'NOT-FOUND', $Path)
    }
}

Write-Output "=== Directory sizes ==="
Show-DirSize 'C:\Users\27871\anaconda3\pkgs' 'C:pkgs'
Show-DirSize 'D:\Miniconda3\pkgs' 'D:pkgs'
Show-DirSize "$env:LOCALAPPDATA\pip\Cache" 'pip-cache'
Show-DirSize 'D:\Miniconda3\envs' 'D:envs'

Write-Output ""
Write-Output "=== Drives ==="
Get-PSDrive -PSProvider FileSystem |
    Where-Object { $_.Name -in 'C', 'D' } |
    ForEach-Object { Write-Output ("{0}: free {1:N1} GB" -f $_.Name, ($_.Free / 1GB)) }

Write-Output ""
Write-Output "=== .condarc files ==="
foreach ($f in @("$env:USERPROFILE\.condarc", 'D:\Miniconda3\.condarc', 'C:\Users\27871\anaconda3\.condarc')) {
    if (Test-Path $f) {
        Write-Output "--- $f ---"
        Get-Content $f | ForEach-Object { Write-Output "  $_" }
    } else {
        Write-Output "--- $f (absent)"
    }
}

Write-Output ""
Write-Output "=== pip config ==="
foreach ($f in @("$env:APPDATA\pip\pip.ini", "$env:LOCALAPPDATA\pip\pip.ini", 'C:\ProgramData\pip\pip.ini', 'D:\Miniconda3\pip.ini')) {
    if (Test-Path $f) {
        Write-Output "--- $f ---"
        Get-Content $f | ForEach-Object { Write-Output "  $_" }
    }
}
Write-Output "(end)"
