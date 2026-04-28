"""
main.py — Punto de entrada CLI para el Analizador de Similitud de PDFs.

Orquesta el pipeline completo de analisis de principio a fin:

1. Parsear argumentos de linea de comandos.
2. Extraer texto del PDF A y del PDF B.
3. Extraer palabras clave de cada texto.
4. Calcular similitud (Jaccard + Semantica).
5. Generar diagrama de Venn.
6. Generar grafico de barras de puntuaciones.
7. Generar reporte de texto.
8. Imprimir un resumen formateado en la consola.

Este es el **unico** archivo que importa de todos los modulos de ``src/``.
Aqui no reside logica de negocio — solo llamadas de orquestacion.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.extractor import extract_text, get_text_preview
from src.keywords import extract_keywords, filter_keywords
from src.similarity import SimilarityResult, compute_similarity
from src.visualization import generate_report, plot_score_bars, plot_venn_diagram


def _build_parser() -> argparse.ArgumentParser:
    """Construir y retornar el parser de argumentos CLI.

    Returns:
        Un :class:`argparse.ArgumentParser` completamente configurado.
    """
    parser = argparse.ArgumentParser(
        prog="pdf-similarity-analyzer",
        description=(
            "Analizar la similitud tematica entre dos documentos PDF "
            "usando extraccion de palabras clave y embeddings de sentence-transformer."
        ),
    )

    # -- Requeridos ──────────────────────────────────────────────────
    parser.add_argument(
        "--pdf-a",
        type=str,
        required=True,
        help="Ruta al primer archivo PDF.",
    )
    parser.add_argument(
        "--pdf-b",
        type=str,
        required=True,
        help="Ruta al segundo archivo PDF.",
    )

    # -- Opcionales ─────────────────────────────────────────────────
    parser.add_argument(
        "--label-a",
        type=str,
        default=None,
        help="Nombre para mostrar del PDF A (por defecto: nombre del archivo sin extension).",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default=None,
        help="Nombre para mostrar del PDF B (por defecto: nombre del archivo sin extension).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Numero de palabras clave a extraer por PDF (por defecto: 25).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help='Directorio para archivos de salida (por defecto: "outputs/").',
    )
    parser.add_argument(
        "--jaccard-weight",
        type=float,
        default=0.4,
        help="Peso para la puntuacion Jaccard en la metrica combinada (por defecto: 0.4).",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.6,
        help="Peso para la puntuacion semantica en la metrica combinada (por defecto: 0.6).",
    )

    return parser


def _print_summary(
    result: SimilarityResult,
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Imprimir un resumen formateado del analisis en la consola.

    Args:
        result: El :class:`SimilarityResult` calculado.
        label_a: Etiqueta para mostrar del documento A.
        label_b: Etiqueta para mostrar del documento B.
        output_dir: Directorio donde se guardaron los archivos de salida.
    """
    sep = "=" * 60
    dash = "-" * 60

    shared_preview: str = ", ".join(result.shared_keywords[:8])
    if len(result.shared_keywords) > 8:
        shared_preview += ", …"

    print()
    print(sep)
    print("  REPORTE DE SIMILITUD DE PDFs")
    print(sep)
    print(f"  PDF A : {label_a}")
    print(f"  PDF B : {label_b}")
    print(dash)
    print(f"  Similitud Jaccard   (palabras clave) : {result.jaccard_pct}")
    print(f"  Similitud Semantica (significado)     : {result.semantic_pct}")
    print(f"  Puntuacion combinada                  : {result.combined_pct}")
    print(dash)
    print(f"  Palabras clave compartidas : {shared_preview if shared_preview else '(ninguna)'}")
    print(f"  Salidas guardadas en       : {output_dir.resolve()}")
    print(sep)
    print()


def main() -> None:
    """Ejecutar el pipeline completo de analisis de similitud de PDFs."""
    # -- Logging ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # -- 1. Parsear argumentos CLI ──────────────────────────────────
    parser = _build_parser()
    args = parser.parse_args()

    pdf_a_path: Path = Path(args.pdf_a)
    pdf_b_path: Path = Path(args.pdf_b)
    output_dir: Path = Path(args.output_dir)

    label_a: str = args.label_a if args.label_a else pdf_a_path.stem
    label_b: str = args.label_b if args.label_b else pdf_b_path.stem

    top_n: int = args.top_n
    jaccard_w: float = args.jaccard_weight
    semantic_w: float = args.semantic_weight

    try:
        # -- 2. Extraer texto ───────────────────────────────────────
        logger.info("Extrayendo texto de '%s' …", pdf_a_path.name)
        text_a: str = extract_text(str(pdf_a_path))
        logger.info("  Vista previa: %s", get_text_preview(text_a, max_chars=120))

        logger.info("Extrayendo texto de '%s' …", pdf_b_path.name)
        text_b: str = extract_text(str(pdf_b_path))
        logger.info("  Vista previa: %s", get_text_preview(text_b, max_chars=120))

        # -- 3. Extraer palabras clave ──────────────────────────────
        logger.info("Extrayendo palabras clave de '%s' (top_n=%d) …", label_a, top_n)
        keywords_a: list[str] = extract_keywords(text_a, top_n=top_n)
        keywords_a = filter_keywords(keywords_a)

        logger.info("Extrayendo palabras clave de '%s' (top_n=%d) …", label_b, top_n)
        keywords_b: list[str] = extract_keywords(text_b, top_n=top_n)
        keywords_b = filter_keywords(keywords_b)

        # -- 4. Calcular similitud ──────────────────────────────────
        logger.info("Calculando puntuaciones de similitud …")
        result: SimilarityResult = compute_similarity(
            keywords_a=keywords_a,
            keywords_b=keywords_b,
            text_a=text_a,
            text_b=text_b,
            jaccard_weight=jaccard_w,
            semantic_weight=semantic_w,
        )

        # -- 5. Generar diagrama de Venn ────────────────────────────
        venn_path: str = plot_venn_diagram(
            result,
            label_a=label_a,
            label_b=label_b,
            output_path=str(output_dir / "venn.png"),
        )
        logger.info("Diagrama de Venn guardado en '%s'.", venn_path)

        # -- 6. Generar grafico de barras ───────────────────────────
        bars_path: str = plot_score_bars(
            result,
            label_a=label_a,
            label_b=label_b,
            output_path=str(output_dir / "scores.png"),
        )
        logger.info("Grafico de barras guardado en '%s'.", bars_path)

        # -- 7. Generar reporte de texto ────────────────────────────
        report_path: str = generate_report(
            result,
            label_a=label_a,
            label_b=label_b,
            output_path=str(output_dir / "report.txt"),
        )
        logger.info("Reporte de texto guardado en '%s'.", report_path)

        # -- 8. Imprimir resumen en consola ─────────────────────────
        _print_summary(result, label_a, label_b, output_dir)

    except FileNotFoundError as exc:
        print(f"\n[ERROR] Archivo no encontrado: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] Entrada invalida: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
