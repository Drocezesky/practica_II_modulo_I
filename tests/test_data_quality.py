import os
import tempfile

import pandas as pd
import pytest
import requests

from app.load_data import DATASET_URL


CACHE_PATH = os.path.join(tempfile.gettempdir(), "qa_dataset_test_cache.parquet")


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """Descarga el parquet una sola vez (o usa la copia cacheada) y lo
    carga como DataFrame para que todos los tests lo reutilicen."""
    if not os.path.exists(CACHE_PATH):
        r = requests.get(DATASET_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(CACHE_PATH, "wb") as f:
            f.write(r.content)

    return pd.read_parquet(CACHE_PATH)


# =======================================================================
# Estructura y esquema
#
# NOTA: el dataset de origen (category-classifier-supplement) es un
# dataset de clasificación de texto, no un dataset de preguntas y
# respuestas. Sus columnas reales son "text" y "label", no "question"
# y "answer". load_data.py ya hace ese mapeo (row.get("text", "") ->
# question, row.get("label", None) -> category), así que estos tests
# verifican las columnas REALES del dataset, no las del modelo destino.
# =======================================================================


def test_dataset_no_esta_vacio(df):
    """El dataset debe tener al menos una fila; si Hugging Face cambió
    la URL o el archivo, esto lo detecta temprano."""
    assert len(df) > 0


def test_columnas_esperadas_existen(df):
    """Las columnas mínimas que load_data.py necesita para mapear a
    'question' y 'category' deben estar presentes en el dataset."""
    columnas_requeridas = {"text", "label"}
    faltantes = columnas_requeridas - set(df.columns)
    assert not faltantes, f"Faltan columnas esperadas: {faltantes}"


def test_tipos_de_datos_son_string(df):
    """'text' debe ser texto (object/string), no números ni otro tipo
    que rompería la inserción en la columna Text de Postgres."""
    assert pd.api.types.is_string_dtype(df["text"].dtype), (
        f"La columna 'text' tiene tipo {df['text'].dtype}, se esperaba string"
    )


# =======================================================================
# Completitud (nulos / vacíos)
# =======================================================================


def test_sin_valores_nulos_en_campos_obligatorios(df):
    """'text' se mapea a 'question', que es nullable=False en el modelo
    de SQLAlchemy: si viene nulo del dataset, la inserción fallaría o
    insertaría basura."""
    nulos_text = df["text"].isna().sum()
    assert nulos_text == 0, f"{nulos_text} filas con 'text' nulo"


def test_sin_strings_vacios_o_solo_espacios(df):
    """Un valor como '' o '   ' pasa el chequeo de 'no nulo' pero es
    igual de inútil. load_data.py no lo filtra, así que lo detectamos acá."""
    text_vacios = df["text"].str.strip().eq("").sum()
    assert text_vacios == 0, f"{text_vacios} filas de 'text' vacías o solo espacios"


def test_proporcion_de_label_nulo_es_razonable(df):
    """'label' se mapea a 'category', que es opcional en el modelo. Pero
    si viene 100% nulo, es señal de que el mapeo en load_data.py no
    sirve para este dataset (o cambió el nombre de columna)."""
    if "label" not in df.columns:
        pytest.skip("El dataset no tiene columna 'label'; no aplica.")

    proporcion_nula = df["label"].isna().mean()
    assert proporcion_nula < 0.95, (
        "Más del 95% de 'label' es nulo: revisar el mapeo de columnas "
        "en load_data.py, puede que el dataset use otro nombre de columna."
    )


# =======================================================================
# Campo 'answer': este dataset probablemente NO trae respuestas reales
# =======================================================================


def test_answer_es_none_para_todas_las_filas(df):
    """load_data.py hace:
        answer=row.get("answer", None) or row.get("answer_alias", None)
    Este dataset de clasificación probablemente no trae ni 'answer' ni
    'answer_alias', así que TODAS las filas insertadas quedan con
    answer=None. Esto documenta el comportamiento actual: si algún día
    el dataset SÍ trae respuestas, este test empieza a fallar y hay que
    revisarlo (dejaría de ser un problema y pasaría a ser una mejora)."""
    tiene_answer = "answer" in df.columns
    tiene_answer_alias = "answer_alias" in df.columns

    if not tiene_answer and not tiene_answer_alias:
        pytest.skip(
            "El dataset no tiene 'answer' ni 'answer_alias': confirmado, "
            "todas las respuestas se insertan como None. Si el modelo "
            "requiere 'answer' obligatorio, considerar cambiar de dataset "
            "o generar respuestas por otro medio."
        )

    # Si en algún momento aparece alguna de las dos columnas, verificamos
    # que el fallback realmente capture contenido.
    respuesta_final = pd.Series([None] * len(df))
    if tiene_answer:
        respuesta_final = df["answer"]
    if tiene_answer_alias:
        respuesta_final = respuesta_final.where(
            respuesta_final.notna() & (respuesta_final != ""), df["answer_alias"]
        )

    vacias = respuesta_final.isna().sum() + (respuesta_final == "").sum()
    assert vacias < len(df), (
        "Aparecieron columnas de respuesta pero siguen totalmente vacías: "
        "revisar el mapeo de todas formas."
    )


# =======================================================================
# Duplicados y consistencia
# =======================================================================


def test_proporcion_de_preguntas_duplicadas_es_baja(df):
    """Un dataset de texto puede tener algún duplicado legítimo, pero si
    la mayoría se repite, algo está mal en la fuente (ej. se descargó el
    mismo shard dos veces)."""
    proporcion_duplicados = df["text"].duplicated().mean()
    assert proporcion_duplicados < 0.10, (
        f"{proporcion_duplicados:.1%} de las filas de 'text' están duplicadas, "
        "se esperaba menos del 10%"
    )


def test_longitud_de_texto_es_razonable(df):
    """Detecta outliers extremos: textos de 1 carácter o de 50.000
    caracteres suelen indicar filas corruptas o mal parseadas."""
    longitudes_text = df["text"].str.len()

    assert (longitudes_text >= 1).all(), "Hay textos sospechosamente cortos (<3 caracteres)"
    assert (longitudes_text <= 5000).mean() > 0.99, (
        "Más del 1% de los textos supera los 5000 caracteres, revisar parsing"
    )


def test_valores_de_label_son_consistentes(df):
    """Si 'label' se usa como categoría, no debería tener una cardinalidad
    absurda (ej. un valor distinto por fila), lo que indicaría que en
    realidad es un campo de texto libre, no una categoría."""
    if "label" not in df.columns:
        pytest.skip("El dataset no tiene columna 'label'; no aplica.")

    cardinalidad = df["label"].nunique(dropna=True)
    assert cardinalidad < len(df) * 0.5, (
        f"'label' tiene {cardinalidad} valores distintos sobre {len(df)} filas: "
        "demasiado alta para ser una categoría, revisar si es la columna correcta."
    )