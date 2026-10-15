# Lance VoxLingo : backend FastAPI (port 8000) + frontend React (port 5173).
# Usage, depuis la racine du projet :  .\start.ps1
# Ctrl+C dans cette fenêtre arrête les deux serveurs.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# Python de l'environnement conda "dubbing"
$condaBase = $null
foreach ($candidate in @($env:CONDA_PREFIX_1, $env:CONDA_PREFIX, "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\miniconda3")) {
    if ($candidate -and (Test-Path (Join-Path $candidate 'envs\dubbing\python.exe'))) { $condaBase = $candidate; break }
}
if (-not $condaBase) {
    Write-Host "Environnement conda 'dubbing' introuvable. Voir README.md (Installation)." -ForegroundColor Red
    exit 1
}
$python = Join-Path $condaBase 'envs\dubbing\python.exe'

# Dépendances du frontend au premier lancement
if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
    Write-Host 'Installation des dépendances du frontend...' -ForegroundColor Cyan
    Push-Location (Join-Path $root 'frontend'); npm install; Pop-Location
}

# Le backend tourne dans sa propre fenêtre (réduite) : au rechargement automatique,
# uvicorn envoie Ctrl+C à toute sa console, ce qui arrêterait aussi le frontend.
# Ses journaux sont aussi écrits dans data/logs/voxlingo.log.
Write-Host 'Démarrage du backend  : http://localhost:8000 (fenêtre « VoxLingo API » réduite)' -ForegroundColor Cyan
$backend = Start-Process -FilePath $python -WindowStyle Minimized `
    -ArgumentList '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000', '--reload', '--reload-exclude', 'tests/*' `
    -WorkingDirectory (Join-Path $root 'backend') -PassThru

Write-Host 'Démarrage du frontend : http://localhost:5173' -ForegroundColor Cyan
$frontend = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'npm run dev' `
    -WorkingDirectory (Join-Path $root 'frontend') -NoNewWindow -PassThru

try {
    Wait-Process -Id $backend.Id, $frontend.Id
} finally {
    Write-Host 'Arrêt des serveurs...' -ForegroundColor Yellow
    foreach ($p in @($backend, $frontend)) {
        if ($p -and -not $p.HasExited) { & taskkill /PID $p.Id /T /F | Out-Null }
    }
}
