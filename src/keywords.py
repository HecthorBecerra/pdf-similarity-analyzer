"""
keywords.py — Modulo de extraccion de palabras clave.

Responsable de extraer las palabras clave y frases clave mas
representativas de documentos de texto plano usando KeyBERT respaldado
por un modelo multilingue de sentence-transformers. No se realiza
lectura de PDFs ni calculo de similitud aqui.

El modelo KeyBERT se instancia una vez al cargar el modulo para que
las llamadas repetidas no paguen el costo de carga del modelo de nuevo.
"""

import logging
from typing import Final

from keybert import KeyBERT

logger = logging.getLogger(__name__)

# -- Modelo a nivel de modulo (se carga una sola vez) ───────────────
_MODEL_NAME: Final[str] = "paraphrase-multilingual-MiniLM-L12-v2"

logger.info("Cargando modelo KeyBERT '%s' …", _MODEL_NAME)
kw_model: Final[KeyBERT] = KeyBERT(model=_MODEL_NAME)
logger.info("Modelo KeyBERT cargado exitosamente.")

_MIN_TEXT_LENGTH: Final[int] = 50


def extract_keywords(
    text: str,
    top_n: int = 25,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[str]:
    """Extraer las top-N palabras clave mas representativas de *text*.

    Usa la seleccion Max-Marginal-Relevance (MMR) con un factor de
    diversidad de 0.5 para reducir redundancia entre las palabras
    clave retornadas.

    Args:
        text: El texto de entrada a analizar. Debe tener al menos 50 caracteres.
        top_n: Numero de palabras clave / frases clave a retornar.
        ngram_range: El tamano (min, max) de n-gramas para frases candidatas.

    Returns:
        Una lista de cadenas de palabras clave ordenadas por relevancia
        (las puntuaciones se descartan).

    Raises:
        ValueError: Si *text* esta vacio o tiene menos de 50 caracteres.
    """
    _validate_text(text)

    logger.info(
        "Extrayendo top %d palabras clave (ngram_range=%s) …", top_n, ngram_range
    )

    results: list[tuple[str, float]] = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words=None,
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )

    keywords: list[str] = [kw for kw, _score in results]
    logger.info("Se extrajeron %d palabras clave.", len(keywords))
    return keywords


def extract_keywords_with_scores(
    text: str,
    top_n: int = 25,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[tuple[str, float]]:
    """Extraer palabras clave junto con sus puntuaciones de relevancia.

    Se comporta de forma identica a :func:`extract_keywords` pero
    preserva las puntuaciones de similitud coseno asignadas por el modelo.

    Args:
        text: El texto de entrada a analizar. Debe tener al menos 50 caracteres.
        top_n: Numero de palabras clave / frases clave a retornar.
        ngram_range: El tamano (min, max) de n-gramas para frases candidatas.

    Returns:
        Una lista de tuplas ``(palabra_clave, puntuacion)`` ordenadas en
        orden descendente por *puntuacion*. Las puntuaciones son flotantes
        en el rango ``[0.0, 1.0]``.

    Raises:
        ValueError: Si *text* esta vacio o tiene menos de 50 caracteres.
    """
    _validate_text(text)

    logger.info(
        "Extrayendo top %d palabras clave con puntuaciones (ngram_range=%s) …",
        top_n,
        ngram_range,
    )

    results: list[tuple[str, float]] = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words=None,
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )

    # Asegurar orden descendente por puntuacion (KeyBERT usualmente retorna
    # ordenado, pero lo garantizamos aqui).
    results.sort(key=lambda pair: pair[1], reverse=True)

    logger.info("Se extrajeron %d palabras clave con puntuaciones.", len(results))
    return results


def filter_keywords(
    keywords: list[str],
    min_length: int = 3,
) -> list[str]:
    """Remover palabras clave cortas y duplicadas preservando el orden.

    Args:
        keywords: La lista de palabras clave a filtrar.
        min_length: Longitud minima de caracteres que una palabra clave
            debe tener para ser conservada. Por defecto ``3``.

    Returns:
        Una nueva lista conteniendo solo palabras clave unicas cuya
        longitud es al menos *min_length*, en su orden original.
    """
    seen: set[str] = set()
    filtered: list[str] = []

    for kw in keywords:
        normalised = kw.strip()
        if len(normalised) >= min_length and normalised not in seen:
            seen.add(normalised)
            filtered.append(normalised)

    logger.info(
        "Palabras clave filtradas: %d → %d (min_length=%d).",
        len(keywords),
        len(filtered),
        min_length,
    )
    return filtered


# -- Funciones auxiliares internas ───────────────────────────────────

def _validate_text(text: str) -> None:
    """Lanzar ``ValueError`` si *text* es demasiado corto para una extraccion significativa.

    Args:
        text: La cadena de texto candidata.

    Raises:
        ValueError: Si *text* esta vacio o tiene menos de 50 caracteres.
    """
    if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
        raise ValueError(
            f"El texto debe tener al menos {_MIN_TEXT_LENGTH} caracteres "
            f"para una extraccion significativa de palabras clave (se recibieron "
            f"{len(text.strip())} caracteres)."
        )


# -- Prueba rapida ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    sample_en = (
        "Artificial intelligence and machine learning are transforming "
        "the way we analyse data. Deep learning models, especially "
        "transformer architectures, have achieved remarkable results in "
        "natural language processing, computer vision, and speech "
        "recognition. Transfer learning allows practitioners to fine-tune "
        "pre-trained models on domain-specific tasks with limited labelled "
        "data, significantly reducing training time and computational cost."
    )

    sample_es = (
        "La inteligencia artificial y el aprendizaje automatico estan "
        "transformando la forma en que analizamos los datos. Los modelos "
        "de aprendizaje profundo, especialmente las arquitecturas de "
        "transformadores, han logrado resultados notables en el "
        "procesamiento del lenguaje natural, la vision por computadora y "
        "el reconocimiento de voz. El aprendizaje por transferencia "
        "permite a los profesionales ajustar modelos preentrenados en "
        "tareas especificas del dominio con datos etiquetados limitados, "
        "reduciendo significativamente el tiempo de entrenamiento y el "
        "costo computacional."
    )

    # -- Demo en ingles ─────────────────────────────────────────────
    print("\n=== Palabras clave en ingles ===")
    kw_en = extract_keywords(sample_en, top_n=10)
    for k in kw_en:
        print(f"  • {k}")

    print("\n=== Palabras clave en ingles con puntuaciones ===")
    kw_en_scores = extract_keywords_with_scores(sample_en, top_n=10)
    for k, s in kw_en_scores:
        print(f"  • {k:<35s} {s:.4f}")

    # -- Demo en espanol ────────────────────────────────────────────
    print("\n=== Palabras clave en espanol ===")
    kw_es = extract_keywords(sample_es, top_n=10)
    for k in kw_es:
        print(f"  • {k}")

    print("\n=== Palabras clave en espanol con puntuaciones ===")
    kw_es_scores = extract_keywords_with_scores(sample_es, top_n=10)
    for k, s in kw_es_scores:
        print(f"  • {k:<35s} {s:.4f}")

    # -- Demo de filtrado ───────────────────────────────────────────
    print("\n=== Palabras clave en ingles filtradas (min_length=5) ===")
    filtered = filter_keywords(kw_en, min_length=5)
    for k in filtered:
        print(f"  • {k}")

    # -- Demo de manejo de errores ──────────────────────────────────
    print("\n=== Demo de ValueError (texto corto) ===")
    try:
        extract_keywords("muy corto")
    except ValueError as exc:
        print(f"  ValueError capturado: {exc}")
