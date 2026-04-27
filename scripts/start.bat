@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo  ErgoTracker - Inicio
echo ============================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Entorno virtual no encontrado. Ejecuta primero scripts\setup.bat
    pause & exit /b 1
)
call .venv\Scripts\activate.bat

:: Start API in background
echo [INFO] Iniciando API en http://localhost:8000 ...
start "ErgoTracker API" /min cmd /c "call .venv\Scripts\activate.bat && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: Wait for API to be ready
echo [INFO] Esperando a que la API este lista...
:wait_loop
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 goto wait_loop
echo [OK] API lista

:: Start frontend if npm available
where npm >nul 2>&1
if not errorlevel 1 (
    if exist "frontend\package.json" (
        echo [INFO] Iniciando frontend en http://localhost:5173 ...
        start "ErgoTracker UI" /min cmd /c "cd frontend && npm run dev"
        timeout /t 3 /nobreak >nul
        echo [OK] Frontend disponible en http://localhost:5173
    )
)

echo.
echo ============================================================
echo  ErgoTracker en ejecucion:
echo    API:      http://localhost:8000
echo    Docs API: http://localhost:8000/docs
echo    Frontend: http://localhost:5173
echo ============================================================
echo  Presiona cualquier tecla para detener todos los servicios
pause >nul

:: Cleanup
echo [INFO] Deteniendo servicios...
taskkill /f /fi "WINDOWTITLE eq ErgoTracker API" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq ErgoTracker UI" >nul 2>&1
echo [OK] Servicios detenidos
