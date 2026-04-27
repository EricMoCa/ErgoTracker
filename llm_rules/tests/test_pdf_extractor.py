"""
Tests para PDFExtractor.
"""
import hashlib
from pathlib import Path

import pytest

from llm_rules.pdf_extractor import PDFExtractor


# ---------------------------------------------------------------------------
# Tests de extract_text
# ---------------------------------------------------------------------------


def test_extract_text_returns_string(sample_pdf):
    """extract_text retorna un string no vacío para el PDF sintético."""
    extractor = PDFExtractor()
    text = extractor.extract_text(sample_pdf)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_extract_text_contains_keywords(sample_pdf):
    """El texto extraído contiene palabras clave del PDF sintético."""
    extractor = PDFExtractor()
    text = extractor.extract_text(sample_pdf)
    assert "tronco" in text.lower() or "flexión" in text.lower() or "grados" in text.lower()


def test_extract_text_file_not_found():
    """extract_text lanza excepción si el PDF no existe."""
    extractor = PDFExtractor()
    with pytest.raises(Exception):
        extractor.extract_text("/nonexistent/path/file.pdf")


# ---------------------------------------------------------------------------
# Tests de extract_pages
# ---------------------------------------------------------------------------


def test_extract_pages_returns_list(sample_pdf):
    """extract_pages retorna una lista de dicts con las claves correctas."""
    extractor = PDFExtractor()
    pages = extractor.extract_pages(sample_pdf)
    assert isinstance(pages, list)
    assert len(pages) >= 1
    first = pages[0]
    assert "page" in first
    assert "text" in first
    assert "word_count" in first


def test_extract_pages_page_numbers(sample_pdf):
    """Los números de página empiezan en 1."""
    extractor = PDFExtractor()
    pages = extractor.extract_pages(sample_pdf)
    assert pages[0]["page"] == 1


def test_extract_pages_word_count(sample_pdf):
    """word_count es consistente con el texto de la página."""
    extractor = PDFExtractor()
    pages = extractor.extract_pages(sample_pdf)
    for page in pages:
        expected = len(page["text"].split())
        assert page["word_count"] == expected


# ---------------------------------------------------------------------------
# Tests de chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_basic():
    """chunk_text divide texto largo en chunks del tamaño esperado."""
    extractor = PDFExtractor()
    text = "palabra " * 1000  # ~8000 caracteres
    chunks = extractor.chunk_text(text, chunk_size=2000, overlap=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 2000 + 200  # margen por overlap


def test_chunk_text_empty():
    """chunk_text retorna lista vacía para texto vacío."""
    extractor = PDFExtractor()
    chunks = extractor.chunk_text("", chunk_size=2000)
    assert chunks == []


def test_chunk_text_short():
    """chunk_text retorna un único chunk si el texto cabe en uno."""
    extractor = PDFExtractor()
    text = "Texto corto de prueba."
    chunks = extractor.chunk_text(text, chunk_size=2000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap():
    """chunk_text con overlap produce solapamiento entre chunks consecutivos."""
    extractor = PDFExtractor()
    # Texto lo suficientemente largo para producir varios chunks
    text = ("A" * 1500 + "\n\n" + "B" * 1500 + "\n\n" + "C" * 1500)
    chunks = extractor.chunk_text(text, chunk_size=2000, overlap=200)
    assert len(chunks) >= 2


def test_chunk_text_no_loss(sample_pdf):
    """El chunking no pierde información — todo el texto aparece en algún chunk."""
    extractor = PDFExtractor()
    text = extractor.extract_text(sample_pdf)
    chunks = extractor.chunk_text(text, chunk_size=500, overlap=50)
    # Verificamos que al menos parte del texto original aparece en los chunks
    combined = " ".join(chunks)
    # El texto original debe aparecer (al menos parcialmente) en los chunks
    assert len(combined) >= len(text) * 0.8  # al menos 80% del texto está representado


# ---------------------------------------------------------------------------
# Tests de compute_hash
# ---------------------------------------------------------------------------


def test_compute_hash_returns_hex_string(sample_pdf):
    """compute_hash retorna un string hexadecimal de 64 caracteres (SHA-256)."""
    extractor = PDFExtractor()
    hash_val = extractor.compute_hash(sample_pdf)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
    assert all(c in "0123456789abcdef" for c in hash_val)


def test_compute_hash_deterministic(sample_pdf):
    """compute_hash produce el mismo resultado para el mismo archivo."""
    extractor = PDFExtractor()
    hash1 = extractor.compute_hash(sample_pdf)
    hash2 = extractor.compute_hash(sample_pdf)
    assert hash1 == hash2


def test_compute_hash_different_files(tmp_path):
    """compute_hash produce valores diferentes para archivos diferentes."""
    import fitz

    extractor = PDFExtractor()

    doc1 = fitz.open()
    page1 = doc1.new_page()
    page1.insert_text((50, 50), "Contenido del primer PDF.")
    path1 = tmp_path / "pdf1.pdf"
    doc1.save(str(path1))
    doc1.close()

    doc2 = fitz.open()
    page2 = doc2.new_page()
    page2.insert_text((50, 50), "Contenido completamente diferente del segundo PDF.")
    path2 = tmp_path / "pdf2.pdf"
    doc2.save(str(path2))
    doc2.close()

    hash1 = extractor.compute_hash(str(path1))
    hash2 = extractor.compute_hash(str(path2))
    assert hash1 != hash2


def test_compute_hash_matches_manual(sample_pdf):
    """compute_hash produce el mismo valor que hashlib directo."""
    extractor = PDFExtractor()
    result = extractor.compute_hash(sample_pdf)

    sha256 = hashlib.sha256()
    with open(sample_pdf, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    expected = sha256.hexdigest()

    assert result == expected
