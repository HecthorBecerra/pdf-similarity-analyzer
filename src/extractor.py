"""
extractor.py — Modulo de extraccion de texto de PDFs.

Responsable de leer archivos PDF desde el sistema de archivos y extraer
su contenido de texto crudo usando PyMuPDF (fitz). Cada funcion publica
opera sobre una o mas rutas de PDF y retorna cadenas de texto plano
normalizadas. No se realiza extraccion de palabras clave, logica de
similitud ni filtrado de idioma aqui.
"""

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> str:
    """Extraer y retornar el contenido de texto completo de un archivo PDF.

    Abre el PDF ubicado en *pdf_path*, itera sobre cada pagina y
    concatena el texto extraido. Los espacios en blanco y saltos de
    linea excesivos se colapsan en espacios simples para que el
    llamador reciba una cadena limpia y continua.

    Args:
        pdf_path: Ruta absoluta o relativa al archivo PDF.

    Returns:
        Una cadena unica conteniendo el texto normalizado de todas las paginas.

    Raises:
        FileNotFoundError: Si *pdf_path* no apunta a un archivo existente.
        ValueError: Si el PDF existe pero no produce texto legible.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Archivo PDF no encontrado: '{path.resolve()}'. "
            "Por favor verifica la ruta e intenta de nuevo."
        )

    doc: fitz.Document = fitz.open(str(path))
    try:
        raw_pages: list[str] = [page.get_text() for page in doc]
    finally:
        doc.close()

    raw_text = " ".join(raw_pages)

    # Normalizar espacios en blanco: colapsar secuencias de espacios / saltos de linea / tabulaciones
    # en un solo espacio y eliminar espacios al inicio y final.
    clean_text: str = re.sub(r"\s+", " ", raw_text).strip()

    if not clean_text:
        raise ValueError(
            f"El PDF en '{path.resolve()}' no contiene texto extraible. "
            "El archivo puede estar basado en imagenes o estar corrupto."
        )

    return clean_text


def extract_texts_from_folder(folder_path: str) -> dict[str, str]:
    """Extraer texto de cada PDF encontrado en *folder_path*.

    Itera sobre todos los archivos ``*.pdf`` en el directorio dado.
    Los archivos que no se pueden procesar (corruptos, solo imagenes,
    etc.) se omiten con una advertencia registrada — la funcion nunca
    falla por errores en archivos individuales.

    Args:
        folder_path: Ruta a un directorio que contiene uno o mas archivos PDF.

    Returns:
        Un diccionario mapeando cada nombre de archivo PDF (ej. ``"reporte.pdf"``)
        a su texto plano extraido. Los archivos que fallaron en la extraccion
        se omiten del resultado.

    Raises:
        FileNotFoundError: Si *folder_path* no existe o no es un directorio.
    """
    folder = Path(folder_path)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Directorio no encontrado: '{folder.resolve()}'. "
            "Por favor proporciona una ruta de carpeta valida."
        )

    pdf_files: list[Path] = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No se encontraron archivos PDF en '%s'.", folder.resolve())
        return {}

    texts: dict[str, str] = {}

    for pdf_file in pdf_files:
        try:
            text = extract_text(str(pdf_file))
            texts[pdf_file.name] = text
            logger.info(
                "Texto extraido exitosamente de '%s' (%d caracteres).",
                pdf_file.name,
                len(text),
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            logger.warning(
                "Omitiendo '%s': %s",
                pdf_file.name,
                exc,
            )

    return texts


def get_text_preview(text: str, max_chars: int = 300) -> str:
    """Retornar una vista previa truncada de *text* para logging y depuracion.

    Si el texto es mas largo que *max_chars*, se corta y se agrega
    ``"..."`` para indicar truncamiento. De lo contrario se retorna
    el texto completo sin cambios.

    Args:
        text: El texto fuente del cual generar la vista previa.
        max_chars: Numero maximo de caracteres a conservar antes de
            truncar. Por defecto ``300``.

    Returns:
        La cadena de vista previa (posiblemente truncada).
    """
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


# -- Prueba rapida ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # 1. Demostrar manejo de FileNotFoundError
    dummy_path = "archivo_inexistente.pdf"
    print(f"\n--- Prueba 1: extract_text('{dummy_path}') ---")
    try:
        extract_text(dummy_path)
    except FileNotFoundError as e:
        print(f"FileNotFoundError capturado: {e}")

    # 2. Demostrar extraccion de carpeta con directorio inexistente
    dummy_folder = "carpeta_inexistente"
    print(f"\n--- Prueba 2: extract_texts_from_folder('{dummy_folder}') ---")
    try:
        extract_texts_from_folder(dummy_folder)
    except FileNotFoundError as e:
        print(f"FileNotFoundError capturado: {e}")

    # 3. Demostrar get_text_preview
    sample = "Lorem ipsum " * 50
    preview = get_text_preview(sample, max_chars=80)
    print(f"\n--- Prueba 3: get_text_preview (max_chars=80) ---")
    print(f"Vista previa: {preview}")

    # 4. Intentar extraccion desde la carpeta real pdfs/ (puede estar vacia)
    pdfs_dir = "pdfs"
    print(f"\n--- Prueba 4: extract_texts_from_folder('{pdfs_dir}') ---")
    try:
        results = extract_texts_from_folder(pdfs_dir)
        if results:
            for name, content in results.items():
                print(f"  {name}: {get_text_preview(content, 120)}")
        else:
            print("  (no se encontraron PDFs o todos fallaron)")
    except FileNotFoundError as e:
        print(f"FileNotFoundError capturado: {e}")
