@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo  ErgoTracker - Suite de Tests
echo ============================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Entorno virtual no encontrado. Ejecuta primero scripts\setup.bat
    pause & exit /b 1
)
call .venv\Scripts\activate.bat

:: Parse arguments
set MODULE=%1
set EXTRA=%2

if "%MODULE%"=="" (
    echo [INFO] Ejecutando todos los tests (excluye GPU e integracion)...
    pytest --tb=short -m "not gpu and not integration" -q
) else if "%MODULE%"=="all" (
    echo [INFO] Ejecutando TODOS los tests (incluye marcadores)...
    pytest --tb=short -q
) else (
    echo [INFO] Ejecutando tests de modulo: %MODULE%
    pytest %MODULE%\ --tb=short -v %EXTRA%
)

echo.
echo ============================================================
echo  Tests completados. Codigo de salida: %ERRORLEVEL%
echo ============================================================
if "%1"=="" pause
