#!/usr/bin/env bash
set -e

echo "============================================================"
echo " ErgoTracker - Inicio"
echo "============================================================"

if [ ! -f ".venv/bin/activate" ]; then
    echo "[ERROR] Entorno virtual no encontrado. Ejecuta primero: bash scripts/setup.sh"
    exit 1
fi
source .venv/bin/activate

cleanup() {
    echo ""
    echo "[INFO] Deteniendo servicios..."
    kill "$API_PID" 2>/dev/null || true
    kill "$UI_PID" 2>/dev/null || true
    echo "[OK] Servicios detenidos"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start API
echo "[INFO] Iniciando API en http://localhost:8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API
echo "[INFO] Esperando a que la API esté lista..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "[OK] API lista"
        break
    fi
    sleep 1
done

# Start frontend
if command -v npm &>/dev/null && [ -f "frontend/package.json" ]; then
    echo "[INFO] Iniciando frontend en http://localhost:5173 ..."
    cd frontend && npm run dev &
    UI_PID=$!
    cd ..
    echo "[OK] Frontend disponible en http://localhost:5173"
fi

echo ""
echo "============================================================"
echo " ErgoTracker en ejecución:"
echo "   API:      http://localhost:8000"
echo "   Docs API: http://localhost:8000/docs"
echo "   Frontend: http://localhost:5173"
echo "============================================================"
echo " Ctrl+C para detener"

wait
