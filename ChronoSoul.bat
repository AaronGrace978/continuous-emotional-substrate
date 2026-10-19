@echo off
setlocal
cd /d "%~dp0desktop"
where node >nul 2>&1
if errorlevel 1 (
  echo Node.js is required. Install it from https://nodejs.org and run this again.
  pause
  exit /b 1
)
if not exist "node_modules\electron\package.json" (
  echo Installing Electron...
  call npm install
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
)
call npm start
