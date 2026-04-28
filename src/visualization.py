"""
visualization.py — Modulo de visualizacion de resultados.

Responsable de generar y guardar salidas visuales (diagramas de Venn,
graficos de barras y reportes de texto plano) a partir de un
:class:`SimilarityResult`. No se realiza lectura de PDFs, extraccion
de palabras clave ni calculo de similitud aqui.
"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib_venn import venn2

from src.similarity import SimilarityResult

logger = logging.getLogger(__name__)

# -- Paleta de colores ──────────────────────────────────────────────
_COLOUR_A: str = "#5b9bd5"       # azul suave
_COLOUR_B: str = "#ed7d31"       # naranja calido
_COLOUR_SHARED: str = "#70ad47"  # verde suave
_COLOUR_SEMANTIC: str = "#9b59b6"  # morado
_COLOUR_COMBINED: str = "#2c3e50"  # gris oscuro


# -- Funcion auxiliar ───────────────────────────────────────────────

def _ensure_parent(path: Path) -> None:
    """Crear el directorio padre de *path* si no existe.

    Args:
        path: Ruta de archivo cuyo directorio padre se debe garantizar.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _truncate_list(items: list[str], max_items: int) -> str:
    """Unir hasta *max_items* cadenas con saltos de linea.

    Si la lista es mas larga que *max_items*, se agrega una linea
    final con ``"…"``.

    Args:
        items: Cadenas a unir.
        max_items: Numero maximo de elementos a conservar.

    Returns:
        Una cadena separada por saltos de linea adecuada para
        etiquetas de diagramas de Venn.
    """
    if len(items) <= max_items:
        return "\n".join(items) if items else ""
    return "\n".join(items[:max_items]) + "\n…"


# -- API publica ────────────────────────────────────────────────────

def plot_venn_diagram(
    result: SimilarityResult,
    label_a: str = "PDF 1",
    label_b: str = "PDF 2",
    output_path: str = "outputs/venn.png",
    max_keywords_shown: int = 6,
) -> str:
    """Crear un diagrama de Venn de dos conjuntos del solapamiento de palabras clave y guardarlo.

    El circulo izquierdo muestra palabras clave exclusivas del documento A,
    el circulo derecho muestra palabras clave exclusivas del documento B,
    y la interseccion muestra las palabras clave compartidas. Cada region
    muestra hasta *max_keywords_shown* entradas (una por linea).

    Args:
        result: Un :class:`SimilarityResult` conteniendo los desgloses
            de palabras clave a graficar.
        label_a: Etiqueta para mostrar del primer documento.
        label_b: Etiqueta para mostrar del segundo documento.
        output_path: Ruta de destino del archivo para la imagen PNG guardada.
        max_keywords_shown: Maximo de palabras clave renderizadas por region
            antes de truncar con ``"…"``.

    Returns:
        La ruta absoluta del archivo guardado como cadena.
    """
    out = Path(output_path)
    _ensure_parent(out)

    # -- Construir cadenas de etiquetas ─────────────────────────────
    left_text = _truncate_list(result.exclusive_to_a, max_keywords_shown)
    right_text = _truncate_list(result.exclusive_to_b, max_keywords_shown)
    centre_text = _truncate_list(result.shared_keywords, max_keywords_shown)

    # -- Dibujar Venn ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    v = venn2(
        subsets=(
            len(result.exclusive_to_a),
            len(result.exclusive_to_b),
            len(result.shared_keywords),
        ),
        set_labels=(label_a, label_b),
        set_colors=(_COLOUR_A, _COLOUR_B),
        alpha=0.55,
        ax=ax,
    )

    # Inyectar texto de palabras clave en cada region (si la region existe)
    if v.get_label_by_id("10"):
        v.get_label_by_id("10").set_text(left_text)
        v.get_label_by_id("10").set_fontsize(8)
    if v.get_label_by_id("01"):
        v.get_label_by_id("01").set_text(right_text)
        v.get_label_by_id("01").set_fontsize(8)
    if v.get_label_by_id("11"):
        v.get_label_by_id("11").set_text(centre_text)
        v.get_label_by_id("11").set_fontsize(8)

    ax.set_title(
        f"Solapamiento de palabras clave: {label_a} vs {label_b}",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0.5,
        -0.05,
        f"Similitud combinada: {result.combined_pct}",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        style="italic",
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)

    abs_path = str(out.resolve())
    logger.info("Diagrama de Venn guardado en '%s'.", abs_path)
    return abs_path


def plot_score_bars(
    result: SimilarityResult,
    label_a: str = "PDF 1",
    label_b: str = "PDF 2",
    output_path: str = "outputs/scores.png",
) -> str:
    """Crear un grafico de barras horizontales de las puntuaciones de similitud y guardarlo.

    Se renderizan tres barras, cada una rellenada proporcionalmente (0 → 1)
    y anotada con su valor porcentual:

    * **Jaccard** (solapamiento de palabras clave)
    * **Semantica** (nivel de significado)
    * **Combinada** (promedio ponderado)

    Args:
        result: Un :class:`SimilarityResult` conteniendo las puntuaciones.
        label_a: Etiqueta para mostrar del primer documento.
        label_b: Etiqueta para mostrar del segundo documento.
        output_path: Ruta de destino del archivo para la imagen PNG guardada.

    Returns:
        La ruta absoluta del archivo guardado como cadena.
    """
    out = Path(output_path)
    _ensure_parent(out)

    categories: list[str] = ["Jaccard (palabras clave)", "Semantica (significado)", "Combinada"]
    scores: list[float] = [
        result.jaccard_score,
        result.semantic_score,
        result.combined_score,
    ]
    pcts: list[str] = [result.jaccard_pct, result.semantic_pct, result.combined_pct]
    colours: list[str] = [_COLOUR_A, _COLOUR_SEMANTIC, _COLOUR_COMBINED]

    fig, ax = plt.subplots(figsize=(9, 4))

    y_positions = range(len(categories))
    bars = ax.barh(
        y_positions,
        scores,
        color=colours,
        edgecolor="#dddddd",
        height=0.55,
    )

    # Anotar cada barra con su etiqueta de porcentaje
    for bar, pct in zip(bars, pcts):
        width = bar.get_width()
        ax.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            pct,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Puntuacion (0 – 1)", fontsize=10)
    ax.set_title(
        f"Puntuaciones de similitud: {label_a} vs {label_b}",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)

    abs_path = str(out.resolve())
    logger.info("Grafico de barras guardado en '%s'.", abs_path)
    return abs_path


def generate_report(
    result: SimilarityResult,
    label_a: str = "PDF 1",
    label_b: str = "PDF 2",
    output_path: str = "outputs/report.txt",
) -> str:
    """Escribir un reporte resumen en texto plano del analisis de similitud.

    El reporte incluye la marca de tiempo de generacion, etiquetas de
    documentos, las tres puntuaciones de similitud, las 10 principales
    palabras clave compartidas y hasta 5 palabras clave exclusivas
    por documento.

    Args:
        result: Un :class:`SimilarityResult` conteniendo las metricas
            y desgloses de palabras clave.
        label_a: Etiqueta para mostrar del primer documento.
        label_b: Etiqueta para mostrar del segundo documento.
        output_path: Ruta de destino del archivo para el reporte.

    Returns:
        La ruta absoluta del archivo guardado como cadena.
    """
    out = Path(output_path)
    _ensure_parent(out)

    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    shared_top10: list[str] = result.shared_keywords[:10]
    excl_a_top5: list[str] = result.exclusive_to_a[:5]
    excl_b_top5: list[str] = result.exclusive_to_b[:5]

    lines: list[str] = [
        "=" * 60,
        "  REPORTE DE ANALISIS DE SIMILITUD DE PDFs",
        "=" * 60,
        f"  Generado    : {timestamp}",
        f"  Documento A : {label_a}",
        f"  Documento B : {label_b}",
        "-" * 60,
        "",
        "  PUNTUACIONES",
        f"    Jaccard   (solapamiento palabras clave) : {result.jaccard_pct}",
        f"    Semantica (significado)                 : {result.semantic_pct}",
        f"    Combinada (promedio ponderado)           : {result.combined_pct}",
        "",
        "-" * 60,
        f"  PALABRAS CLAVE COMPARTIDAS (top {len(shared_top10)})",
    ]
    if shared_top10:
        for kw in shared_top10:
            lines.append(f"    • {kw}")
    else:
        lines.append("    (ninguna)")

    lines.append("")
    lines.append("-" * 60)
    lines.append(f"  EXCLUSIVAS DE {label_a} (top {len(excl_a_top5)})")
    if excl_a_top5:
        for kw in excl_a_top5:
            lines.append(f"    • {kw}")
    else:
        lines.append("    (ninguna)")

    lines.append("")
    lines.append(f"  EXCLUSIVAS DE {label_b} (top {len(excl_b_top5)})")
    if excl_b_top5:
        for kw in excl_b_top5:
            lines.append(f"    • {kw}")
    else:
        lines.append("    (ninguna)")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    report_text: str = "\n".join(lines)

    with out.open("w", encoding="utf-8") as fh:
        fh.write(report_text)

    abs_path = str(out.resolve())
    logger.info("Reporte de texto guardado en '%s'.", abs_path)
    return abs_path


# -- Prueba rapida ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Construir un SimilarityResult de prueba para demostracion
    mock_result = SimilarityResult(
        jaccard_score=0.345,
        semantic_score=0.721,
        combined_score=0.345 * 0.4 + 0.721 * 0.6,
        shared_keywords=[
            "machine learning", "deep learning", "neural networks",
            "data science", "artificial intelligence",
        ],
        exclusive_to_a=[
            "computer vision", "image recognition", "convolutional networks",
            "object detection", "segmentation",
        ],
        exclusive_to_b=[
            "natural language processing", "text mining", "transformers",
            "sentiment analysis", "word embeddings",
        ],
        jaccard_pct="34.5%",
        semantic_pct="72.1%",
        combined_pct=f"{(0.345 * 0.4 + 0.721 * 0.6) * 100:.1f}%",
    )

    # -- Generar todas las salidas ──────────────────────────────────
    print("\n=== Generando diagrama de Venn ===")
    venn_path = plot_venn_diagram(
        mock_result,
        label_a="Paper A",
        label_b="Paper B",
        output_path="outputs/venn_demo.png",
    )
    print(f"  Guardado en: {venn_path}")

    print("\n=== Generando grafico de barras ===")
    bars_path = plot_score_bars(
        mock_result,
        label_a="Paper A",
        label_b="Paper B",
        output_path="outputs/scores_demo.png",
    )
    print(f"  Guardado en: {bars_path}")

    print("\n=== Generando reporte de texto ===")
    report_path = generate_report(
        mock_result,
        label_a="Paper A",
        label_b="Paper B",
        output_path="outputs/report_demo.txt",
    )
    print(f"  Guardado en: {report_path}")

    # Imprimir contenido del reporte en consola
    print("\n" + Path(report_path).read_text(encoding="utf-8"))
