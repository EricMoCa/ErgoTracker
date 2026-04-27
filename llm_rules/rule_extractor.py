"""
llm_rules/rule_extractor.py
Extracción de reglas ergonómicas de PDFs usando Gemma 3 4B via Ollama.
"""
import json
import os
from pathlib import Path

import requests
from loguru import logger

from schemas import ErgonomicRule, JointAngles

from .gpu_manager import GPUManager
from .pdf_extractor import PDFExtractor
from .rule_cache import RuleCache

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
            "risk_level": {
                "type": "string",
                "enum": ["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
            },
            "action": {"type": "string"},
            "source": {"type": "string"},
        },
        "required": ["id", "description", "joint", "condition", "risk_level", "action", "source"],
    },
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

        rules: list[ErgonomicRule] = []
        try:
            # 2. Extraer y chunkear texto
            text = self.pdf_extractor.extract_text(pdf_path)
            chunks = self.pdf_extractor.chunk_text(text)
            logger.info(f"PDF dividido en {len(chunks)} chunks para procesamiento")

            # 3. Procesar cada chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Procesando chunk {i + 1}/{len(chunks)}...")
                chunk_rules = self._extract_from_chunk(chunk, pdf_path)
                rules.extend(chunk_rules)

            # 4. Deduplicar por joint y condition
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
                {"role": "user", "content": f"Fragmento de normativa:\n\n{chunk}"},
            ],
            "stream": False,
            "temperature": 0,           # Determinista para extracción
            "format": RULE_JSON_SCHEMA,  # Structured output de Ollama
        }
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()

        content = resp.json()["message"]["content"]
        raw_rules = json.loads(content)

        # Añadir source file a cada regla
        for r in raw_rules:
            if "source" not in r or not r["source"]:
                r["source"] = f"PDF:{Path(source_file).name}"

        return [
            ErgonomicRule(**r)
            for r in raw_rules
            if self._is_valid_joint(r.get("joint", ""))
        ]

    def _is_valid_joint(self, joint: str) -> bool:
        """Filtra reglas con joints que no existen en JointAngles."""
        return joint in JointAngles.model_fields

    def _deduplicate(self, rules: list[ErgonomicRule]) -> list[ErgonomicRule]:
        """Elimina reglas con el mismo joint y condition."""
        seen: set[tuple[str, str]] = set()
        unique: list[ErgonomicRule] = []
        for r in rules:
            key = (r.joint, r.condition)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
