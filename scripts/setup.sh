#!/usr/bin/env bash
set -e

echo "============================================================"
echo " ErgoTracker - Setup (Linux/macOS)"
echo "============================================================"

# Verify Python 3.11+
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 no encontrado. Instala Python 3.11+"
    exit 1
fi

PYVER=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "[OK] Python $PYVER"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[INFO] Creando entorno virtual..."
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip --quiet

# Install all requirements
echo "[INFO] Instalando dependencias Python..."
for f in requirements/base.txt requirements/pose.txt requirements/ergo.txt \
          requirements/llm.txt requirements/reports.txt requirements/advanced.txt \
          requirements/api.txt; do
    if [ -f "$f" ]; then
        echo "  -> $f"
        pip install -r "$f" --quiet
    fi
done

# Node/npm for frontend
if ! command -v npm &>/dev/null; then
    echo "[WARN] npm no encontrado. Frontend no será instalado."
    echo "       Instala Node.js 18+ para habilitar la UI"
else
    echo "[INFO] Instalando dependencias del frontend..."
    cd frontend && npm install --silent && cd ..
    echo "[OK] Frontend listo"
fi

echo ""
echo "============================================================"
echo " Setup completado. Para iniciar ErgoTracker:"
echo "   bash scripts/start.sh"
echo "============================================================"
