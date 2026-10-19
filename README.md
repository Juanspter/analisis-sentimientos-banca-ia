# Análisis de sentimientos bancarios con Hugging Face

**Portafolio de Inteligencia Artificial | Procesamiento de Lenguaje Natural (NLP)**

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow) ![Google%20Colab](https://img.shields.io/badge/Google%20Colab-GPU-orange) ![Estado](https://img.shields.io/badge/Estado-Prototipo%20acad%C3%A9mico-lightgrey)

## Descripción del proyecto

Prototipo académico para **clasificar comentarios sintéticos de clientes bancarios** en sentimientos **positivo (POS), neutral (NEU) y negativo (NEG)**. Se comparan tres modelos preentrenados disponibles en Hugging Face mediante inferencia en Google Colab con GPU. El objetivo es seleccionar el modelo más conveniente considerando **calidad predictiva, latencia y recursos computacionales**.

> **Alcance:** investigación y demostración académica. No es un sistema bancario productivo, no procesa datos reales de clientes y no debe utilizarse para decisiones automatizadas sobre personas.

## Objetivo

Identificar e implementar el modelo preentrenado más adecuado para el análisis de sentimientos de comentarios bancarios en español, documentando resultados y condiciones para facilitar su reproducción.

## Ecosistema y metodología

- **Hugging Face Models:** descarga de modelos preentrenados y ejecución mediante `transformers.pipeline`.
- **Hugging Face Datasets:** componente del ecosistema para gestionar conjuntos de datos; en este experimento se utiliza un **CSV sintético local**.
- **Hugging Face Spaces:** alternativa para desplegar una futura demostración interactiva; **no se desplegó un Space**.
- **Google Colab:** ejecución del notebook con GPU (en la prueba reportada, dispositivo GPU), instalación de dependencias y autenticación **opcional y segura** mediante el secreto `HF_TOKEN`.
- **Evaluación:** mismas 45 observaciones etiquetadas para los tres modelos; normalización de etiquetas a POS/NEU/NEG; cálculo de accuracy, precision macro, recall macro y F1 macro; medición de latencia de inferencia y tiempo de carga.

### Modelos evaluados

| Modelo | Identificador de Hugging Face | Tarea | Parámetros aproximados |
|---|---|---|---:|
| RoBERTuito | [`pysentimiento/robertuito-sentiment-analysis`](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis) | Clasificación de sentimiento | 108.8 M |
| XLM-RoBERTa | [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | Clasificación de sentimiento | 278.0 M |
| BERT estrellas | [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) | Clasificación por estrellas, convertida a sentimiento | 167.4 M |

**Licencias:** consultar las tarjetas oficiales enlazadas de cada modelo antes de redistribuirlo o utilizarlo comercialmente. El presente repositorio no presupone que todos compartan una misma licencia.

## Dataset

Archivo: [`data/comentarios_bancarios.csv`](data/comentarios_bancarios.csv).

- **Origen:** ejemplos **sintéticos** creados para la actividad; no son conversaciones ni datos personales de clientes reales.
- **Volumen:** 45 comentarios únicos; cada uno es evaluado por los tres modelos (**135 predicciones** en total).
- **Campos del dataset:** `id` (identificador), `texto` (comentario), `sentimiento` (etiqueta de referencia POS/NEU/NEG) y `origen` (procedencia declarada en el conjunto de datos).
- **Limitación:** la muestra pequeña y artificial no permite extrapolar los resultados al conjunto de clientes de un banco.

## Resultados experimentales

Resultados tomados de [`results/metricas_modelos.csv`](results/metricas_modelos.csv), generados en la ejecución de Google Colab; **45 ejemplos por modelo**. Los valores se muestran redondeados.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Latencia media por texto | Tiempo de carga |
|---|---:|---:|---:|---:|---:|---:|
| **RoBERTuito** | **91.11 %** | **92.98 %** | **91.11 %** | **90.95 %** | 3.52 ms | 8.93 s |
| XLM-RoBERTa | 88.89 % | 90.51 % | 88.89 % | 88.75 % | 5.86 ms | 33.48 s |
| BERT estrellas | 66.67 % | 63.56 % | 66.67 % | 60.12 % | **2.53 ms** | 28.37 s |

**Archivos de evidencia:**

- [`results/predicciones_modelos.csv`](results/predicciones_modelos.csv): una fila por texto y modelo; columnas `id`, `modelo`, `real`, `predicho`, `confianza_top1`, `texto`.
- [`results/metricas_modelos.csv`](results/metricas_modelos.csv): métricas agregadas, latencias, parámetros, dispositivo y número de ejemplos.
- [`docs/`](docs/): reporte técnico comparativo en Word, incluyendo metodología, decisiones, riesgos y conclusiones.

**Interpretación:** RoBERTuito obtuvo el mayor accuracy y F1 macro en este conjunto. BERT estrellas fue el más rápido en latencia media, pero obtuvo un F1 macro notablemente inferior. Las latencias corresponden a una ejecución específica y pueden variar según GPU, versiones, caché y carga del entorno. El valor `confianza_top1` es la puntuación del modelo, **no** una medida de accuracy ni una probabilidad necesariamente calibrada.

## Modelo recomendado y trade-offs

**RoBERTuito** es la recomendación **para este prototipo y este dataset**, por presentar el mejor desempeño de clasificación, una latencia de 3.52 ms por comentario y menor cantidad de parámetros que las otras dos alternativas. La decisión no implica superioridad universal ni validación en producción.

Se utilizaron **ajustes mínimos**, sin reentrenamiento: preparación y validación de datos, normalización de etiquetas, selección de GPU y configuración de inferencia. Antes de plantear *fine-tuning* conviene ampliar el conjunto de evaluación, revisar errores por clase y medir estabilidad y sesgos.

## Cómo reproducir el experimento

1. Abre [`notebooks/`](notebooks/) y selecciona el notebook `.ipynb` de la actividad; utiliza la opción **Open in Colab** si está disponible o descárgalo y ábrelo en [Google Colab](https://colab.research.google.com/).
2. En Colab selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**, si está disponible. Comprueba la GPU con `!nvidia-smi`.
3. Ejecuta las celdas en orden. La primera instala las dependencias necesarias; reinicia el entorno si Colab lo solicita.
4. Los modelos son públicos. La autenticación con Hugging Face es **opcional**; si la utilizas, crea un secreto `HF_TOKEN` en Colab. **No publiques tokens**.
5. Cuando el notebook solicite el dataset, sube el archivo [`data/comentarios_bancarios.csv`](data/comentarios_bancarios.csv) desde tu computadora. El notebook espera el nombre `comentarios_bancarios.csv` en el directorio de trabajo de Colab.
6. Ejecuta la inferencia de los tres modelos y la evaluación. Revisa los resultados generados y compáralos con los CSV de [`results/`](results/).
7. Para el entorno local, consulta [`requirements.txt`](requirements.txt). La disponibilidad y versión de PyTorch con GPU dependen de Colab y del controlador CUDA.

### Recalcular métricas a partir de predicciones

Los datos publicados permiten recalcular las métricas sin volver a descargar modelos. El CSV utiliza las columnas **`modelo`, `real` y `predicho`**. Para reproducir el resumen con `scikit-learn`, desde la raíz del repositorio:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv('results/predicciones_modelos.csv')
for nombre, grupo in df.groupby('modelo'):
    p, r, f1, _ = precision_recall_fscore_support(
        grupo['real'], grupo['predicho'], average='macro', zero_division=0
    )
    print(nombre, 'accuracy=', round(accuracy_score(grupo['real'], grupo['predicho']), 4),
          'precision=', round(p, 4), 'recall=', round(r, 4), 'f1=', round(f1, 4))
```

**Atención:** si `src/evaluar_modelos.py` todavía utiliza nombres de columnas de plantilla (`etiqueta_real`, `prediccion`), deben sustituirse por `real` y `predicho` antes de ejecutarlo. Esta comprobación es necesaria para que el script sea funcional.

## Estructura del repositorio

```text
analisis-sentimientos-banca-ia/
├── README.md
├── requirements.txt
├── data/
│   └── comentarios_bancarios.csv
├── notebooks/
│   └── 20261017_actividad4_comparacion_sentimientos_banca.ipynb
├── docs/
│   └── [reporte comparativo en Word]
├── results/
│   ├── predicciones_modelos.csv
│   └── metricas_modelos.csv
└── src/
    └── evaluar_modelos.py
```

> El nombre del documento Word en `docs/` puede variar según la versión subida. Comprueba que sea el informe actualizado y que sus tablas coincidan con los CSV.

## Umbrales, riesgos y siguientes pasos

- **Umbral de referencia académico:** accuracy ≥ 80 %, complementado con F1 macro y revisión del desempeño de cada clase. No es un criterio suficiente para aprobar un sistema productivo.
- **Riesgos:** muestra reducida, datos sintéticos, posibles sesgos de lenguaje y dominio, ambigüedad de mensajes neutrales, etiquetas convertidas desde estrellas y variación de tiempos entre ejecuciones.
- **Privacidad y seguridad:** no incorporar mensajes bancarios reales, identificadores personales, credenciales ni tokens al repositorio público.
- **Próximas mejoras:** dataset real autorizado y anonimizado, evaluación por clase, validación externa, repetición de mediciones de latencia, revisión de licencias y análisis de necesidad de ajuste fino.

## Conclusión

La comparación práctica demuestra que seleccionar un modelo preentrenado requiere evaluar **calidad y eficiencia en el contexto del problema**, no solo popularidad. En esta prueba controlada, RoBERTuito ofreció el mejor equilibrio para clasificar comentarios bancarios sintéticos en español. El notebook, los datos, las predicciones y las métricas permiten auditar la decisión y continuar mejorando el prototipo.

---

**Proyecto académico de portafolio de IA.** Los resultados son experimentales y no constituyen una validación para uso bancario en producción.
