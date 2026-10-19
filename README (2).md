# Análisis de sentimientos bancarios con Hugging Face

Proyecto académico de procesamiento de lenguaje natural (NLP) para comparar **tres modelos preentrenados** en la clasificación de comentarios bancarios en español: **NEG**, **NEU** y **POS**.

> **Datos sintéticos:** este proyecto no utiliza comentarios ni información confidencial de clientes reales. Los resultados son experimentales y no deben emplearse para tomar decisiones bancarias automatizadas.

## Objetivo
Evaluar accuracy, precision macro, recall macro, F1 macro y latencia en Google Colab (GPU) para seleccionar un modelo considerando calidad y recursos.

## Modelos evaluados
1. [`pysentimiento/robertuito-sentiment-analysis`](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis)
2. [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
3. [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) (1–2 estrellas=NEG; 3=NEU; 4–5=POS)

## Resultados del experimento

| Modelo | Accuracy | F1 macro | Latencia media |
|---|---:|---:|---:|
| RoBERTuito | 91.11 % | 90.95 % | 3.52 ms |
| XLM-RoBERTa | 88.89 % | 88.75 % | 5.86 ms |
| BERT estrellas | 66.67 % | 60.12 % | 2.53 ms |

**Recomendación experimental:** RoBERTuito. Estos valores provienen del reporte técnico de la ejecución compartida; para validarlos o actualizarlos, ejecutar el notebook y consultar sus resultados. Las latencias dependen del hardware y de las condiciones de ejecución.

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── comentarios_bancarios.csv
├── notebooks/
│   └── 20261017_actividad4_comparacion_sentimientos_banca.ipynb
├── results/
│   └── predicciones_modelos.csv
├── docs/
│   └── analisis_comparativo_actualizado.docx
└── src/
    └── README.md
```

## Entorno y ejecución
1. Abrir [Google Colab](https://colab.research.google.com/) y cargar `notebooks/20261017_actividad4_comparacion_sentimientos_banca.ipynb`.
2. Activar **Entorno de ejecución → Cambiar tipo de entorno → GPU T4** (si está disponible).
3. Ejecutar las celdas en orden. Instalar dependencias desde el notebook o `requirements.txt`.
4. Para autenticar en Hugging Face, usar **Secrets** de Colab con `HF_TOKEN` o inicio de sesión interactivo; **nunca publicar tokens**.
5. Mantener el CSV `data/comentarios_bancarios.csv` disponible en la ruta indicada en el notebook. Verificar rutas de entrada y salida antes de ejecutar.
6. Comparar los resultados generados con `results/predicciones_modelos.csv` y el reporte Word.

## Dataset y trazabilidad
`data/comentarios_bancarios.csv` contiene **45 textos sintéticos**, etiquetados para clasificación ternaria. `results/predicciones_modelos.csv` contiene **135 predicciones** (45 textos × 3 modelos). Consultar los encabezados de ambos CSV para el esquema exacto. La generación de datos sintéticos se describe en el reporte y el notebook; no constituye un muestreo representativo de clientes reales.

## Hugging Face
- **Models:** modelos preentrenados utilizados en la comparación.
- **Datasets:** ecosistema para datasets; en este experimento los datos se versionan como CSV.
- **Spaces:** opción de despliegue futuro, no implementada.

## Riesgos y limitaciones
Dataset pequeño y sintético; posibles sesgos de dominio y clase; diferencias entre etiquetas originales; variabilidad de latencia; licencias de cada modelo pendientes de comprobar para cualquier uso distinto al académico. No hay evidencia de generalización a datos reales ni validación de producción.

## Conclusión
En esta evaluación RoBERTuito ofrece el mejor equilibrio entre F1 macro y latencia. Antes de usarlo en un sistema real, ampliar el conjunto de evaluación, revisar errores por clase, comprobar licencias y realizar pruebas con datos autorizados y anonimizados.

## Documentación
El análisis técnico detallado se encuentra en `docs/analisis_comparativo_actualizado.docx`.
