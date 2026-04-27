#!/usr/bin/env bash
set -e

echo "============================================================"
echo " ErgoTracker - Suite de Tests"
echo "============================================================"

if [ ! -f ".venv/bin/activate" ]; then
    echo "[ERROR] Entorno virtual no encontrado. Ejecuta primero: bash scripts/setup.sh"
    exit 1
fi
source .venv/bin/activate

MODULE="${1:-}"
EXTRA="${2:-}"

if [ -z "$MODULE" ]; then
    echo "[INFO] Ejecutando todos los tests (excluye GPU e integración)..."
    pytest --tb=short -m "not gpu and not integration" -q
elif [ "$MODULE" = "all" ]; then
    echo "[INFO] Ejecutando TODOS los tests..."
    pytest --tb=short -q
else
    echo "[INFO] Ejecutando tests de módulo: $MODULE"
    pytest "$MODULE/" --tb=short -v $EXTRA
fi

echo ""
echo "============================================================"
echo " Tests completados"
echo "============================================================"
