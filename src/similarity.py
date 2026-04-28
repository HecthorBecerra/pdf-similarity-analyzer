"""
similarity.py — Modulo de calculo de similitud de textos.

Responsable de calcular la similitud entre dos documentos usando dos
metodos complementarios:

* **Similitud Jaccard** sobre conjuntos de palabras clave (solapamiento lexico).
* **Similitud semantica** via distancia coseno de embeddings de
  sentence-transformer.

Una combinacion ponderada de ambas puntuaciones se retorna dentro de un
dataclass :class:`SimilarityResult`. No se realiza lectura de PDFs,
extraccion de palabras clave ni visualizacion aqui.

El modelo SentenceTransformer se instancia una vez al cargar el modulo
para evitar costos de carga repetidos.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# -- Modelo a nivel de modulo (se carga una sola vez) ───────────────
_MODEL_NAME: Final[str] = "paraphrase-multilingual-MiniLM-L12-v2"

logger.info("Cargando modelo SentenceTransformer '%s' …", _MODEL_NAME)
_model: Final[SentenceTransformer] = SentenceTransformer(_MODEL_NAME)
logger.info("Modelo SentenceTransformer cargado exitosamente.")


# -- Dataclass de resultado ─────────────────────────────────────────

@dataclass
class SimilarityResult:
    """Contenedor para los resultados de un analisis de similitud entre dos documentos.

    Attributes:
        jaccard_score: Indice de Jaccard (0.0 – 1.0) sobre conjuntos de palabras clave.
        semantic_score: Similitud coseno (0.0 – 1.0) de embeddings de oraciones.
        combined_score: Promedio ponderado de *jaccard_score* y *semantic_score*.
        shared_keywords: Palabras clave presentes en ambos conjuntos.
        exclusive_to_a: Palabras clave encontradas solo en el conjunto A.
        exclusive_to_b: Palabras clave encontradas solo en el conjunto B.
        jaccard_pct: Cadena de porcentaje legible, ej. ``"34.5%"``.
        semantic_pct: Cadena de porcentaje legible, ej. ``"72.1%"``.
        combined_pct: Cadena de porcentaje legible, ej. ``"53.3%"``.
    """

    jaccard_score: float
    semantic_score: float
    combined_score: float
    shared_keywords: list[str] = field(default_factory=list)
    exclusive_to_a: list[str] = field(default_factory=list)
    exclusive_to_b: list[str] = field(default_factory=list)
    jaccard_pct: str = ""
    semantic_pct: str = ""
    combined_pct: str = ""


# -- API publica ────────────────────────────────────────────────────

def jaccard_similarity(
    keywords_a: list[str],
    keywords_b: list[str],
) -> float:
    """Calcular el indice de Jaccard entre dos listas de palabras clave.

    La comparacion es **insensible a mayusculas**: todas las palabras
    clave se normalizan a minusculas antes de aplicar las operaciones
    de conjuntos.

    .. math::

        J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

    Args:
        keywords_a: Primera lista de palabras clave.
        keywords_b: Segunda lista de palabras clave.

    Returns:
        El coeficiente de similitud de Jaccard como un flotante en ``[0.0, 1.0]``.
        Retorna ``0.0`` cuando ambas listas estan vacias.
    """
    set_a: set[str] = {kw.lower().strip() for kw in keywords_a}
    set_b: set[str] = {kw.lower().strip() for kw in keywords_b}

    intersection: set[str] = set_a & set_b
    union: set[str] = set_a | set_b

    if not union:
        logger.warning("Ambas listas de palabras clave estan vacias — retornando 0.0.")
        return 0.0

    score = len(intersection) / len(union)
    logger.info(
        "Similitud Jaccard: |A∩B|=%d, |A∪B|=%d → %.4f",
        len(intersection),
        len(union),
        score,
    )
    return score


def semantic_similarity(
    text_a: str,
    text_b: str,
    max_chars: int = 2000,
) -> float:
    """Calcular la similitud coseno de embeddings de sentence-transformer.

    Ambos textos se truncan a *max_chars* caracteres antes de codificar
    para que el modelo se mantenga dentro de sus limites practicos de entrada.

    Args:
        text_a: Texto del primer documento.
        text_b: Texto del segundo documento.
        max_chars: Numero maximo de caracteres a conservar de cada texto
            antes de codificar. Por defecto ``2000``.

    Returns:
        Similitud coseno como un flotante en ``[0.0, 1.0]``.

    Raises:
        ValueError: Si *text_a* o *text_b* esta vacio o en blanco.
    """
    if not text_a or not text_a.strip():
        raise ValueError("text_a esta vacio o en blanco.")
    if not text_b or not text_b.strip():
        raise ValueError("text_b esta vacio o en blanco.")

    truncated_a: str = text_a[:max_chars]
    truncated_b: str = text_b[:max_chars]

    logger.info(
        "Codificando textos (len_a=%d, len_b=%d, max_chars=%d) …",
        len(truncated_a),
        len(truncated_b),
        max_chars,
    )

    embeddings: NDArray[np.float64] = _model.encode(
        [truncated_a, truncated_b],
        convert_to_numpy=True,
    )

    sim_matrix: NDArray[np.float64] = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1),
    )

    score: float = float(sim_matrix[0, 0])

    # Limitar a [0.0, 1.0] — artefactos de redondeo pueden empujar ligeramente fuera.
    score = max(0.0, min(1.0, score))

    logger.info("Similitud semantica: %.4f", score)
    return score


def compute_similarity(
    keywords_a: list[str],
    keywords_b: list[str],
    text_a: str,
    text_b: str,
    jaccard_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> SimilarityResult:
    """Ejecutar el analisis completo de similitud y retornar un resultado estructurado.

    Calcula tanto :func:`jaccard_similarity` como
    :func:`semantic_similarity`, luego produce un promedio ponderado
    (``combined_score``) y rellena un :class:`SimilarityResult`
    con todas las metricas y desgloses de palabras clave.

    Args:
        keywords_a: Palabras clave extraidas del documento A.
        keywords_b: Palabras clave extraidas del documento B.
        text_a: Texto crudo del documento A.
        text_b: Texto crudo del documento B.
        jaccard_weight: Peso asignado a la puntuacion Jaccard en la
            metrica combinada. Por defecto ``0.4``.
        semantic_weight: Peso asignado a la puntuacion semantica en la
            metrica combinada. Por defecto ``0.6``.

    Returns:
        Una instancia :class:`SimilarityResult` completamente poblada.

    Raises:
        ValueError: Si los pesos no suman ``1.0`` (dentro de tolerancia
            de punto flotante) o si alguno de los textos esta vacio.
    """
    if not math.isclose(jaccard_weight + semantic_weight, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"jaccard_weight ({jaccard_weight}) + semantic_weight "
            f"({semantic_weight}) deben sumar 1.0, "
            f"se obtuvo {jaccard_weight + semantic_weight:.10f}."
        )

    logger.info("Calculando similitud (peso Jaccard=%.2f, peso Semantico=%.2f) …",
                jaccard_weight, semantic_weight)

    # -- Jaccard ────────────────────────────────────────────────────
    j_score: float = jaccard_similarity(keywords_a, keywords_b)

    # -- Semantica ──────────────────────────────────────────────────
    s_score: float = semantic_similarity(text_a, text_b)

    # -- Combinada ──────────────────────────────────────────────────
    c_score: float = jaccard_weight * j_score + semantic_weight * s_score

    # -- Desglose de conjuntos de palabras clave ────────────────────
    set_a: set[str] = {kw.lower().strip() for kw in keywords_a}
    set_b: set[str] = {kw.lower().strip() for kw in keywords_b}

    shared: list[str] = sorted(set_a & set_b)
    only_a: list[str] = sorted(set_a - set_b)
    only_b: list[str] = sorted(set_b - set_a)

    result = SimilarityResult(
        jaccard_score=j_score,
        semantic_score=s_score,
        combined_score=c_score,
        shared_keywords=shared,
        exclusive_to_a=only_a,
        exclusive_to_b=only_b,
        jaccard_pct=f"{j_score * 100:.1f}%",
        semantic_pct=f"{s_score * 100:.1f}%",
        combined_pct=f"{c_score * 100:.1f}%",
    )

    logger.info(
        "Similitud calculada — Jaccard: %s, Semantica: %s, Combinada: %s",
        result.jaccard_pct,
        result.semantic_pct,
        result.combined_pct,
    )
    return result


# -- Prueba rapida ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Textos de ejemplo minimos codificados directamente
    text_en = (
        "Artificial intelligence and machine learning are transforming "
        "the way we analyse data. Deep learning models have achieved "
        "remarkable results in natural language processing and computer "
        "vision."
    )
    text_es = (
        "La inteligencia artificial y el aprendizaje automatico estan "
        "transformando la forma en que analizamos los datos. Los modelos "
        "de aprendizaje profundo han logrado resultados notables en el "
        "procesamiento del lenguaje natural y la vision por computadora."
    )

    kw_en = [
        "artificial intelligence", "machine learning", "deep learning",
        "natural language processing", "computer vision", "data analysis",
    ]
    kw_es = [
        "inteligencia artificial", "aprendizaje automatico",
        "aprendizaje profundo", "procesamiento lenguaje natural",
        "vision computadora", "deep learning",
    ]

    # -- Solo Jaccard ───────────────────────────────────────────────
    print("\n=== Similitud Jaccard ===")
    j = jaccard_similarity(kw_en, kw_es)
    print(f"  Puntuacion: {j:.4f}")

    # -- Solo semantica ─────────────────────────────────────────────
    print("\n=== Similitud semantica ===")
    s = semantic_similarity(text_en, text_es)
    print(f"  Puntuacion: {s:.4f}")

    # -- Resultado combinado completo ───────────────────────────────
    print("\n=== SimilarityResult completo ===")
    result = compute_similarity(kw_en, kw_es, text_en, text_es)
    print(f"  Jaccard    : {result.jaccard_pct}")
    print(f"  Semantica  : {result.semantic_pct}")
    print(f"  Combinada  : {result.combined_pct}")
    print(f"  Compartidas: {result.shared_keywords}")
    print(f"  Solo en A  : {result.exclusive_to_a}")
    print(f"  Solo en B  : {result.exclusive_to_b}")

    # -- Demo de manejo de errores ──────────────────────────────────
    print("\n=== Demo de ValueError (texto vacio) ===")
    try:
        semantic_similarity("", "algun texto")
    except ValueError as exc:
        print(f"  ValueError capturado: {exc}")

    print("\n=== Demo de ValueError (pesos incorrectos) ===")
    try:
        compute_similarity(kw_en, kw_es, text_en, text_es,
                           jaccard_weight=0.5, semantic_weight=0.6)
    except ValueError as exc:
        print(f"  ValueError capturado: {exc}")
