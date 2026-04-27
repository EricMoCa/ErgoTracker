@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo  ErgoTracker - Setup (Windows)
echo ============================================================

:: Verify Python 3.11+
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.11+ desde python.org
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%

:: Create virtual environment
if not exist ".venv" (
    echo [INFO] Creando entorno virtual...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip --quiet

:: Install all requirements
echo [INFO] Instalando dependencias Python...
for %%f in (requirements\base.txt requirements\pose.txt requirements\ergo.txt requirements\llm.txt requirements\reports.txt requirements\advanced.txt requirements\api.txt) do (
    if exist "%%f" (
        echo   -> %%f
        pip install -r %%f --quiet
    )
)

:: Node/npm for frontend
where npm >nul 2>&1
if errorlevel 1 (
    echo [WARN] npm no encontrado. Frontend no sera instalado.
    echo        Instala Node.js 18+ desde nodejs.org para habilitar la UI
) else (
    echo [INFO] Instalando dependencias del frontend...
    cd frontend && npm install --silent && cd ..
    echo [OK] Frontend listo
)

echo.
echo ============================================================
echo  Setup completado. Para iniciar ErgoTracker:
echo    scripts\start.bat
echo ============================================================
pause
