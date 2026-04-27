"""
llm_rules/rule_cache.py
Caché SHA-256 para evitar re-procesar PDFs con Ollama.
"""
import json
from pathlib import Path

from loguru import logger

from schemas import ErgonomicRule

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
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return [ErgonomicRule(**r) for r in data]
        return None

    def set(self, pdf_hash: str, rules: list[ErgonomicRule]) -> None:
        """Guarda reglas en caché."""
        cache_file = self.cache_dir / f"{pdf_hash}.json"
        cache_file.write_text(
            json.dumps([r.model_dump() for r in rules], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            f"Reglas guardadas en caché: {len(rules)} reglas para hash {pdf_hash[:8]}..."
        )

    def save_profile(
        self, name: str, rules: list[ErgonomicRule], output_path: str
    ) -> None:
        """
        Guarda las reglas como perfil ergonómico reutilizable en JSON.
        Este archivo es el que consume ergo_engine/llm_rule_analyzer.py
        """
        profile = {
            "profile_name": name,
            "rules": [r.model_dump() for r in rules],
        }
        Path(output_path).write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Perfil '{name}' guardado en {output_path} con {len(rules)} reglas")
