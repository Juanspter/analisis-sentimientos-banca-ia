"""
Proyecto: Análisis de sentimientos bancarios con Hugging Face
Archivo: src/evaluar_modelos.py

Calcula métricas de clasificación a partir de las
predicciones obtenidas en Google Colab.

Modelos:
- RoBERTuito
- XLM-RoBERTa
- BERT estrellas

Ejecutar desde la raíz del repositorio:
    python src/evaluar_modelos.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent

ARCHIVO_ENTRADA = RAIZ / "results" / "predicciones_modelos.csv"
ARCHIVO_SALIDA = RAIZ / "results" / "metricas_recalculadas.csv"

CLASES = ["NEG", "NEU", "POS"]

# Columnas verificadas en el CSV original
COL_MODELO = "modelo"
COL_REAL = "real"
COL_PREDICHO = "predicho"


# --------------------------------------------------
# 2. CARGAR Y VALIDAR DATOS
# --------------------------------------------------

def cargar_datos():

    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ARCHIVO_ENTRADA}"
        )

    df = pd.read_csv(ARCHIVO_ENTRADA)

    columnas = [
        COL_MODELO,
        COL_REAL,
        COL_PREDICHO
    ]

    faltantes = [
        col for col in columnas
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}"
        )

    if df[columnas].isnull().any().any():
        raise ValueError(
            "Existen valores vacíos en columnas obligatorias."
        )

    for columna in [COL_REAL, COL_PREDICHO]:
        df[columna] = (
            df[columna]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    etiquetas = set(df[COL_REAL]) | set(df[COL_PREDICHO])
    invalidas = etiquetas - set(CLASES)

    if invalidas:
        raise ValueError(
            f"Etiquetas no reconocidas: {invalidas}"
        )

    print("\nDATOS CARGADOS CORRECTAMENTE")
    print(f"Total de predicciones: {len(df)}")
    print(f"Modelos encontrados: {df[COL_MODELO].nunique()}")

    return df


# --------------------------------------------------
# 3. EVALUAR MODELOS
# --------------------------------------------------

def evaluar_modelos(df):

    resultados = []

    for modelo, grupo in df.groupby(COL_MODELO):

        reales = grupo[COL_REAL]
        predicciones = grupo[COL_PREDICHO]

        accuracy = accuracy_score(
            reales, predicciones
        )

        precision = precision_score(
            reales,
            predicciones,
            labels=CLASES,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            reales,
            predicciones,
            labels=CLASES,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            reales,
            predicciones,
            labels=CLASES,
            average="macro",
            zero_division=0
        )

        resultados.append({
            "modelo": modelo,
            "n": len(grupo),
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        print("\n" + "=" * 55)
        print(f"MODELO: {modelo}")
        print("=" * 55)

        print(
            classification_report(
                reales,
                predicciones,
                labels=CLASES,
                zero_division=0
            )
        )

        matriz = confusion_matrix(
            reales,
            predicciones,
            labels=CLASES
        )

        print("Matriz de confusión")
        print("Orden de clases:", CLASES)
        print(matriz)

    tabla = pd.DataFrame(resultados)

    tabla = tabla.sort_values(
        by=["f1_macro", "accuracy"],
        ascending=False
    ).reset_index(drop=True)

    return tabla


# --------------------------------------------------
# 4. GUARDAR RESULTADOS
# --------------------------------------------------

def guardar_resultados(tabla):

    ARCHIVO_SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tabla.to_csv(
        ARCHIVO_SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nRESULTADOS COMPARATIVOS")
    print("-" * 55)

    print(
        tabla.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        f"\nArchivo generado: {ARCHIVO_SALIDA}"
    )


# --------------------------------------------------
# 5. CONCLUSIONES
# --------------------------------------------------

def mostrar_conclusion(tabla):

    if tabla.empty:
        print("No hay resultados para comparar.")
        return

    mejor = tabla.iloc[0]

    print("\nCONCLUSIÓN AUTOMÁTICA")
    print("-" * 55)

    print(
        f"Modelo con mayor F1 macro: {mejor['modelo']}"
    )

    print(
        f"Accuracy: {mejor['accuracy']:.2%}"
    )

    print(
        f"Precision macro: {mejor['precision_macro']:.2%}"
    )

    print(
        f"Recall macro: {mejor['recall_macro']:.2%}"
    )

    print(
        f"F1 macro: {mejor['f1_macro']:.2%}"
    )

    print(
        "\nNota: la recomendación final también debe "
        "considerar latencia, recursos, licencia y riesgos."
    )


# --------------------------------------------------
# 6. PROGRAMA PRINCIPAL
# --------------------------------------------------

def main():

    print("=" * 55)
    print("EVALUACIÓN DE MODELOS HUGGING FACE")
    print("ANÁLISIS DE SENTIMIENTOS BANCARIOS")
    print("=" * 55)

    df = cargar_datos()

    tabla = evaluar_modelos(df)

    guardar_resultados(tabla)

    mostrar_conclusion(tabla)

    print("\nEvaluación finalizada correctamente.")


if __name__ == "__main__":
    main()

