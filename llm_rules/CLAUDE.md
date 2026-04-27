# Task C — llm_rules/

## Objetivo
Implementar el módulo de extracción de reglas ergonómicas desde documentos PDF usando Gemma 3 4B via Ollama. Completamente independiente del pipeline de visión. Produce archivos JSON de reglas que son consumidos por `ergo_engine`.

## Dependencia previa
**Requiere que `schemas/` esté completo.** Importa ÚNICAMENTE de `schemas`. NO depende de ningún otro módulo del proyecto.

## Contexto de GPU (CRÍTICO)
Este módulo usa la GPU (4 GB VRAM) para inferencia LLM. Al terminar `extract()`, **debe liberar la VRAM explícitamente** antes de retornar, para que `pose_pipeline` o `advanced_pipeline` puedan usar la GPU después.

## Archivos a crear

```
llm_rules/
├── __init__.py           # Expone: RuleExtractor
├── pdf_extractor.py      # Extracción de texto desde PDF con PyMuPDF
├── rule_extractor.py     # Extracción de reglas con Gemma/Ollama
├── rule_cache.py         # Caché SHA-256 para evitar re-procesar PDFs
├── gpu_manager.py        # Liberación de VRAM tras inferencia LLM
├── conftest.py           # Fixtures (PDF sintético, mock de Ollama)
└── tests/
    ├── test_pdf_extractor.py
    ├── test_rule_extractor.py
    ├── test_rule_cache.py
    └── test_gpu_manager.py
```

## Implementación por archivo

### gpu_manager.py
```python
import requests
import os
from loguru import logger

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class GPUManager:
    """
    Gestiona la liberación de VRAM después de usar el LLM.
    Patrón: cargar Gemma → extraer reglas → liberar → pipeline de visión usa GPU libre.
    """

    @staticmethod
    def release_llm_vram(model_name: str = "gemma3:4b") -> None:
        """
        Fuerza a Ollama a descargar el modelo de la GPU enviando keep_alive=0.
        Llama siempre al finalizar cualquier operación LLM.
        """
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10
            )
            resp.raise_for_status()
            logger.info(f"GPU VRAM liberada: modelo {model_name} descargado de Ollama")
        except Exception as e:
            logger.warning(f"No se pudo liberar VRAM via Ollama: {e}")

        # Purgar caché CUDA residual si PyTorch está disponible
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() ejecutado")
        except ImportError:
            pass  # PyTorch no requerido para este módulo

    @staticmethod
    def is_ollama_running() -> bool:
        """Verifica si el servidor Ollama está activo."""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def is_model_available(model_name: str) -> bool:
        """Verifica si el modelo está descargado en Ollama."""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(model_name in m for m in models)
        except Exception:
            return False
```

### pdf_extractor.py
```python
import fitz  # PyMuPDF
from pathlib import Path
from loguru import logger

class PDFExtractor:
    """
    Extrae texto de PDFs usando PyMuPDF.
    Soporta PDFs con texto embebido (la mayoría de normativas digitales).
    Para PDFs escaneados, aplica OCR opcional con pytesseract.
    """

    def __init__(self, use_ocr: bool = False):
        self.use_ocr = use_ocr

    def extract_text(self, pdf_path: str) -> str:
        """
        Extrae todo el texto del PDF como string continuo.
        Retorna texto limpio (sin headers/footers repetitivos si es posible).
        """

    def extract_pages(self, pdf_path: str) -> list[dict]:
        """
        Extrae texto página a página.
        Retorna: [{"page": 1, "text": "...", "word_count": 150}, ...]
        """

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 2000,
        overlap: int = 200
    ) -> list[str]:
        """
        Divide el texto en chunks para enviar al LLM.
        Usa separadores de párrafo (\n\n) para chunking semántico.
        chunk_size: máximo de caracteres por chunk
        overlap: solapamiento entre chunks para no perder contexto
        """

    def compute_hash(self, pdf_path: str) -> str:
        """SHA-256 del archivo PDF para usar como clave de caché."""
```

### rule_cache.py
```python
import json
import hashlib
from pathlib import Path
from schemas import ErgonomicRule
from loguru import logger

CACHE_DIR = Path(".ergo_cache")

class RuleCache:
    """
    Caché local de reglas extraídas por PDF.
    Si el hash SHA-256 del PDF coincide, devuelve las reglas guardadas
    sin volver a llamar a Gemma. Crítico para evitar re-gastar VRAM.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, pdf_hash: str) -> list[ErgonomicRule] | None:
        """Retorna reglas cacheadas o None si no existen."""
        cache_file = self.cache_dir / f"{pdf_hash}.json"
        if cache_file.exists():
            logger.info(f"Cache hit para PDF hash {pdf_hash[:8]}...")
            data = json.loads(cache_file.read_text())
            return [ErgonomicRule(**r) for r in data]
        return None

    def set(self, pdf_hash: str, rules: list[ErgonomicRule]) -> None:
        """Guarda reglas en caché."""
        cache_file = self.cache_dir / f"{pdf_hash}.json"
        cache_file.write_text(
            json.dumps([r.model_dump() for r in rules], ensure_ascii=False, indent=2)
        )
        logger.info(f"Reglas guardadas en caché: {len(rules)} reglas para hash {pdf_hash[:8]}...")

    def save_profile(self, name: str, rules: list[ErgonomicRule], output_path: str) -> None:
        """
        Guarda las reglas como perfil ergonómico reutilizable en JSON.
        Este archivo es el que consume ergo_engine/llm_rule_analyzer.py
        """
        profile = {
            "profile_name": name,
            "rules": [r.model_dump() for r in rules]
        }
        Path(output_path).write_text(
            json.dumps(profile, ensure_ascii=False, indent=2)
        )
```

### rule_extractor.py
```python
import json
import requests
import os
from schemas import ErgonomicRule, RiskLevel
from .pdf_extractor import PDFExtractor
from .rule_cache import RuleCache
from .gpu_manager import GPUManager
from loguru import logger

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "gemma3:4b"

# JSON Schema que Ollama debe respetar en el output
RULE_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "description": {"type": "string"},
            "joint": {"type": "string"},
            "condition": {"type": "string"},
            "risk_level": {"type": "string", "enum": ["NEGLIGIBLE","LOW","MEDIUM","HIGH","VERY_HIGH"]},
            "action": {"type": "string"},
            "source": {"type": "string"}
        },
        "required": ["id","description","joint","condition","risk_level","action","source"]
    }
}

SYSTEM_PROMPT = """Eres un experto en ergonomía industrial. Analiza el siguiente fragmento de normativa
y extrae TODAS las reglas ergonómicas que impliquen ángulos articulares, posturas o umbrales de riesgo.

Para cada regla encontrada, devuelve un objeto JSON con:
- id: identificador único (ej: "R-001")
- description: descripción clara de la regla
- joint: nombre del ángulo afectado. DEBE ser uno de: trunk_flexion, trunk_lateral_bending,
  trunk_rotation, neck_flexion, shoulder_elevation_left, shoulder_elevation_right,
  elbow_flexion_left, elbow_flexion_right, wrist_flexion_left, wrist_flexion_right,
  wrist_deviation_left, wrist_deviation_right, knee_flexion_left, knee_flexion_right
- condition: expresión simple como "angle > 60" o "angle < 30"
- risk_level: uno de NEGLIGIBLE, LOW, MEDIUM, HIGH, VERY_HIGH
- action: acción correctiva recomendada
- source: indica de dónde viene la regla (nombre del PDF y página si está disponible)

Si un fragmento no contiene reglas ergonómicas con ángulos, devuelve un array vacío [].
RESPONDE ÚNICAMENTE CON JSON VÁLIDO, sin explicaciones adicionales."""

class RuleExtractor:
    """
    Extrae reglas ergonómicas de PDFs usando Gemma 3 4B via Ollama.

    Flujo:
    1. Verificar caché (si PDF ya fue procesado, retornar reglas guardadas)
    2. Extraer texto del PDF con PyMuPDF
    3. Dividir en chunks de ~2000 caracteres
    4. Enviar cada chunk a Gemma con structured output (JSON Schema)
    5. Consolidar y deduplicar reglas
    6. Liberar VRAM (SIEMPRE, incluso si hay error)
    7. Guardar en caché y retornar
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.pdf_extractor = PDFExtractor()
        self.cache = RuleCache()
        self.gpu_manager = GPUManager()

    def extract(self, pdf_path: str, profile_name: str = "custom") -> list[ErgonomicRule]:
        """
        Punto de entrada principal. SIEMPRE llama a release_llm_vram() al finalizar.
        """
        pdf_hash = self.pdf_extractor.compute_hash(pdf_path)

        # 1. Verificar caché
        cached = self.cache.get(pdf_hash)
        if cached:
            return cached

        rules = []
        try:
            # 2. Extraer y chunkear texto
            text = self.pdf_extractor.extract_text(pdf_path)
            chunks = self.pdf_extractor.chunk_text(text)
            logger.info(f"PDF dividido en {len(chunks)} chunks para procesamiento")

            # 3. Procesar cada chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Procesando chunk {i+1}/{len(chunks)}...")
                chunk_rules = self._extract_from_chunk(chunk, pdf_path)
                rules.extend(chunk_rules)

            # 4. Deduplicar por descripción similar
            rules = self._deduplicate(rules)
            logger.info(f"Extracción completada: {len(rules)} reglas únicas")

            # 5. Guardar en caché
            self.cache.set(pdf_hash, rules)
            self.cache.save_profile(profile_name, rules, f"{profile_name}_rules.json")

        finally:
            # 6. SIEMPRE liberar VRAM (en finally para garantizar ejecución)
            self.gpu_manager.release_llm_vram(self.model)

        return rules

    def _extract_from_chunk(self, chunk: str, source_file: str) -> list[ErgonomicRule]:
        """Envía un chunk a Ollama y parsea el resultado JSON."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Fragmento de normativa:\n\n{chunk}"}
            ],
            "stream": False,
            "temperature": 0,           # Determinista para extracción
            "format": RULE_JSON_SCHEMA  # Structured output de Ollama
        }
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()

        content = resp.json()["message"]["content"]
        raw_rules = json.loads(content)

        # Añadir source file a cada regla
        for r in raw_rules:
            if "source" not in r or not r["source"]:
                r["source"] = f"PDF:{Path(source_file).name}"

        return [ErgonomicRule(**r) for r in raw_rules if self._is_valid_joint(r.get("joint",""))]

    def _is_valid_joint(self, joint: str) -> bool:
        """Filtra reglas con joints que no existen en JointAngles."""
        from schemas import JointAngles
        return joint in JointAngles.model_fields

    def _deduplicate(self, rules: list[ErgonomicRule]) -> list[ErgonomicRule]:
        """Elimina reglas con el mismo joint y condition."""
        seen = set()
        unique = []
        for r in rules:
            key = (r.joint, r.condition)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
```

## Interfaz Pública (`__init__.py`)
```python
from .rule_extractor import RuleExtractor
from .gpu_manager import GPUManager

__all__ = ["RuleExtractor", "GPUManager"]
```

## Tests

### conftest.py
```python
@pytest.fixture
def sample_pdf(tmp_path):
    """Crea un PDF mínimo con texto ergonómico sintético para tests."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), """
    Artículo 4.2 - Posturas de trabajo
    La flexión del tronco no debe superar los 45 grados durante más de 2 horas.
    Los ángulos de elevación del hombro superiores a 60 grados representan
    riesgo alto y deben evitarse. La flexión del cuello debe mantenerse por
    debajo de 20 grados durante operaciones prolongadas.
    """)
    path = tmp_path / "test_normativa.pdf"
    doc.save(str(path))
    return str(path)

@pytest.fixture
def mock_ollama(monkeypatch):
    """Mock de Ollama para tests sin servidor real."""
    def mock_post(*args, **kwargs):
        class MockResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"message": {"content": json.dumps([{
                    "id": "R-001",
                    "description": "Flexión de tronco máx 45°",
                    "joint": "trunk_flexion",
                    "condition": "angle > 45",
                    "risk_level": "HIGH",
                    "action": "Reducir ángulo o rotar al operario",
                    "source": "PDF:test_normativa.pdf"
                }])}}
        return MockResponse()
    monkeypatch.setattr("requests.post", mock_post)
```

### test_rule_extractor.py
- `test_extract_returns_list_of_rules`: con mock Ollama y PDF sintético
- `test_vram_always_released`: verificar que `release_llm_vram` se llama incluso si Ollama falla
- `test_cache_hit_skips_llm`: procesar el mismo PDF dos veces → segunda vez no llama a requests.post
- `test_invalid_joint_filtered`: reglas con joint inválido no aparecen en el resultado

## Dependencias

```
# requirements/llm.txt
pymupdf>=1.23.0
requests>=2.31.0
loguru>=0.7.0
pytest>=8.0.0
```

## Comandos

```bash
cd llm_rules
pip install -r ../requirements/llm.txt

# Tests con mock (no requiere Ollama real):
pytest tests/ -v

# Test de integración real (requiere Ollama corriendo con gemma3:4b):
OLLAMA_HOST=http://localhost:11434 pytest tests/test_rule_extractor.py -v -m integration
```

## Configuración Ollama

El agente no instala Ollama, pero puede documentar el setup:
```bash
# El usuario debe tener Ollama instalado y el modelo descargado:
ollama pull gemma3:4b
# Verificar que corre:
curl http://localhost:11434/api/tags
```

## NO HACER

- No importar de `pose_pipeline`, `ergo_engine`, `api`, `reports`, o `advanced_pipeline`
- No implementar análisis ergonómico (solo extracción de reglas)
- No cargar modelos ONNX de visión
- No procesar frames de video
- No modificar archivos en `schemas/`
