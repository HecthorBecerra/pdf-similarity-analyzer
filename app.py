"""
app.py — Interfaz web Streamlit para el Analizador de Similitud de PDFs.

Proporciona una interfaz de usuario basada en navegador donde el usuario
sube dos archivos PDF, ejecuta el pipeline completo de analisis y ve
los resultados mostrados en linea.

Toda la logica de negocio se importa desde ``src/`` — en este archivo
no reside ningun calculo de extraccion, palabras clave, similitud ni
visualizacion.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# -- Configuracion de pagina (debe ser el primer comando de Streamlit) --
st.set_page_config(
    page_title="Analizador de Similitud de PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Cargadores de modelos con cache ────────────────────────────────
# Envolver la instanciacion de modelos con st.cache_resource asegura que
# los modelos pesados SentenceTransformer / KeyBERT se carguen exactamente
# una vez en todas las re-ejecuciones y sesiones de Streamlit — critico
# para el rendimiento.

@st.cache_resource(show_spinner="Cargando modelo KeyBERT …")
def _load_keybert_model():
    """Cargar y cachear el modelo KeyBERT (respaldado por sentence-transformers)."""
    from keybert import KeyBERT
    return KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner="Cargando modelo SentenceTransformer …")
def _load_sentence_model():
    """Cargar y cachear el modelo SentenceTransformer."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def _warm_models() -> None:
    """Cargar ambos modelos de forma anticipada para que los imports posteriores los encuentren en cache."""
    _load_keybert_model()
    _load_sentence_model()


# Cargar modelos antes de importar los modulos de src que los cargan a nivel de modulo.
_warm_models()

# -- Imports desde src/ (los modelos ya estan en memoria) ───────────
from src.extractor import extract_text  # noqa: E402
from src.keywords import extract_keywords, extract_keywords_with_scores, filter_keywords  # noqa: E402
from src.similarity import SimilarityResult, compute_similarity  # noqa: E402
from src.visualization import generate_report, plot_score_bars, plot_venn_diagram  # noqa: E402


# -- Funcion auxiliar: guardar archivo subido en disco ──────────────

def _save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    """Escribir un UploadedFile de Streamlit en *dest_dir* y retornar su ruta.

    Args:
        uploaded_file: Un valor retornado por ``st.file_uploader``.
        dest_dir: Directorio donde escribir el archivo.

    Returns:
        La ``pathlib.Path`` del archivo guardado en disco.
    """
    dest_path = dest_dir / uploaded_file.name
    dest_path.write_bytes(uploaded_file.getbuffer())
    return dest_path


# -- Barra lateral ──────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Analizador de Similitud de PDFs")
    st.caption("Compara dos documentos PDF usando solapamiento de palabras clave y embeddings semanticos.")

    st.divider()

    file_a = st.file_uploader("Subir primer PDF", type=["pdf"], key="pdf_a")
    file_b = st.file_uploader("Subir segundo PDF", type=["pdf"], key="pdf_b")

    st.divider()

    label_a: str = st.text_input("Etiqueta para PDF A", value="PDF A")
    label_b: str = st.text_input("Etiqueta para PDF B", value="PDF B")

    top_n: int = st.number_input(
        "Top-N palabras clave",
        min_value=10,
        max_value=50,
        value=25,
        step=1,
    )

    jaccard_weight: float = st.slider(
        "Peso Jaccard",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.1,
    )

    semantic_weight: float = round(1.0 - jaccard_weight, 2)
    st.info(f"Peso semantico: **{semantic_weight}**")

    st.divider()

    run_button: bool = st.button("🔍 Analizar similitud", type="primary", use_container_width=True)


# -- Area principal ─────────────────────────────────────────────────

if run_button:
    # -- Validar archivos subidos ───────────────────────────────────
    if file_a is None or file_b is None:
        st.warning("Por favor sube ambos PDFs antes de ejecutar el analisis.")
        st.stop()

    # -- Crear espacio de trabajo temporal ──────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_sim_"))

    try:
        # -- Guardar archivos subidos ───────────────────────────────
        path_a: Path = _save_uploaded_file(file_a, tmp_dir)
        path_b: Path = _save_uploaded_file(file_b, tmp_dir)

        # -- Paso 1: Extraer texto ──────────────────────────────────
        with st.spinner("Extrayendo texto …"):
            text_a: str = extract_text(str(path_a))
            text_b: str = extract_text(str(path_b))

        # -- Paso 2: Extraer palabras clave ─────────────────────────
        with st.spinner("Extrayendo palabras clave …"):
            keywords_a: list[str] = extract_keywords(text_a, top_n=top_n)
            keywords_a = filter_keywords(keywords_a)

            keywords_b: list[str] = extract_keywords(text_b, top_n=top_n)
            keywords_b = filter_keywords(keywords_b)

            # Tambien obtener versiones con puntuacion para la pestana de detalle
            kw_scored_a: list[tuple[str, float]] = extract_keywords_with_scores(text_a, top_n=top_n)
            kw_scored_b: list[tuple[str, float]] = extract_keywords_with_scores(text_b, top_n=top_n)

        # -- Paso 3: Calcular similitud ─────────────────────────────
        with st.spinner("Calculando similitud …"):
            result: SimilarityResult = compute_similarity(
                keywords_a=keywords_a,
                keywords_b=keywords_b,
                text_a=text_a,
                text_b=text_b,
                jaccard_weight=jaccard_weight,
                semantic_weight=semantic_weight,
            )

        # -- Paso 4: Generar visualizaciones ────────────────────────
        with st.spinner("Generando visualizaciones …"):
            venn_path: str = plot_venn_diagram(
                result,
                label_a=label_a,
                label_b=label_b,
                output_path=str(tmp_dir / "venn.png"),
            )

            bars_path: str = plot_score_bars(
                result,
                label_a=label_a,
                label_b=label_b,
                output_path=str(tmp_dir / "scores.png"),
            )

            report_path: str = generate_report(
                result,
                label_a=label_a,
                label_b=label_b,
                output_path=str(tmp_dir / "report.txt"),
            )

        # -- Banner de exito ────────────────────────────────────────
        st.success("Analisis completado!")

        # -- Tarjetas de metricas ───────────────────────────────────
        col_j, col_s, col_c = st.columns(3)
        col_j.metric("Similitud Jaccard", result.jaccard_pct)
        col_s.metric("Similitud Semantica", result.semantic_pct)
        col_c.metric("Puntuacion Combinada", result.combined_pct)

        # -- Pestanas ───────────────────────────────────────────────
        tab_viz, tab_kw = st.tabs(["📊 Visualizaciones", "🔑 Detalle de palabras clave"])

        with tab_viz:
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.subheader("Diagrama de Venn")
                st.image(venn_path, use_container_width=True)
            with viz_col2:
                st.subheader("Grafico de barras de puntuaciones")
                st.image(bars_path, use_container_width=True)

        with tab_kw:
            kw_col1, kw_col2 = st.columns(2)

            with kw_col1:
                st.subheader(f"Palabras clave — {label_a}")
                df_a = pd.DataFrame(kw_scored_a, columns=["Palabra clave", "Puntuacion"])
                df_a["Puntuacion"] = df_a["Puntuacion"].round(4)
                st.dataframe(df_a, use_container_width=True, hide_index=True)

            with kw_col2:
                st.subheader(f"Palabras clave — {label_b}")
                df_b = pd.DataFrame(kw_scored_b, columns=["Palabra clave", "Puntuacion"])
                df_b["Puntuacion"] = df_b["Puntuacion"].round(4)
                st.dataframe(df_b, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Palabras clave compartidas")
            if result.shared_keywords:
                df_shared = pd.DataFrame(
                    {"Palabra clave": result.shared_keywords}
                )
                st.dataframe(df_shared, use_container_width=True, hide_index=True)
            else:
                st.info("No se encontraron palabras clave compartidas entre los dos documentos.")

        # -- Descargar reporte ──────────────────────────────────────
        st.divider()
        report_text: str = Path(report_path).read_text(encoding="utf-8")
        st.download_button(
            label="📥 Descargar reporte de texto",
            data=report_text,
            file_name="reporte_similitud.txt",
            mime="text/plain",
        )

    except Exception as exc:
        st.error(f"Ocurrio un error durante el analisis: {exc}")
