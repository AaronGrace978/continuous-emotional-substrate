@echo off
setlocal
cd /d "%~dp0desktop"
where node >nul 2>&1
if errorlevel 1 (
  echo Node.js is required. Install it from https://nodejs.org and run this again.
  pause
  exit /b 1
)
findstr /c:"\"version\": \"45." "node_modules\electron\package.json" >nul 2>&1
if errorlevel 1 (
  echo Installing Electron...
  call npm install --no-fund --no-audit
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
)
call npm start
