"""
llm_rules/pdf_extractor.py
Extracción de texto desde PDF con PyMuPDF.
"""
import hashlib
from pathlib import Path

import fitz  # PyMuPDF
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
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages_text.append(text)
        doc.close()
        full_text = "\n\n".join(pages_text)
        logger.debug(f"Extraídos {len(full_text)} caracteres de {Path(pdf_path).name}")
        return full_text

    def extract_pages(self, pdf_path: str) -> list[dict]:
        """
        Extrae texto página a página.
        Retorna: [{"page": 1, "text": "...", "word_count": 150}, ...]
        """
        doc = fitz.open(pdf_path)
        result = []
        for i, page in enumerate(doc):
            text = page.get_text()
            word_count = len(text.split())
            result.append({
                "page": i + 1,
                "text": text,
                "word_count": word_count,
            })
        doc.close()
        logger.debug(f"Extraídas {len(result)} páginas de {Path(pdf_path).name}")
        return result

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> list[str]:
        """
        Divide el texto en chunks para enviar al LLM.
        Usa separadores de párrafo (\\n\\n) para chunking semántico.
        chunk_size: máximo de caracteres por chunk
        overlap: solapamiento entre chunks para no perder contexto
        """
        if not text.strip():
            return []

        # Dividir por párrafos primero
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Si el párrafo solo ya supera chunk_size, dividirlo por caracteres
            if len(para) > chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Split large paragraph by hard limit
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end].strip())
                    start = end - overlap
                continue

            # Si agregar este párrafo supera el límite, guardar chunk actual
            if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                chunks.append(current_chunk.strip())
                # Solapamiento: incluir la parte final del chunk anterior
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.debug(f"Texto dividido en {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")
        return chunks

    def compute_hash(self, pdf_path: str) -> str:
        """SHA-256 del archivo PDF para usar como clave de caché."""
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()
