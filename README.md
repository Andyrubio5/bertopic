# BERTopic: noticias en español

Código reproducible para identificar tópicos en un subcorpus de noticias de **CESS-ESP**, distribuido por NLTK.

Se seleccionaron **330 archivos**, correspondientes a **216 familias de noticias**, y se obtuvieron **1370 segmentos**. Se compararon tres configuraciones de BERTopic usando los mismos embeddings.

## Ejecutar

Entorno comprobado: **Linux, Python 3.12.3 y CPU**.

```bash
git clone https://github.com/Andyrubio5/bertopic.git
cd bertopic
bash instalar.sh
bash ejecutar.sh
```

La instalación requiere internet. La primera ejecución descarga el corpus, las stopwords y el encoder multilingüe (aproximadamente 470 MB). No se necesita una clave de API.

También se puede ejecutar directamente:

```bash
python3 -I -m venv .venv
.venv/bin/python -I -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -I -m pip check
.venv/bin/python -I pipeline.py
```

El modo `-I` evita mezclar las bibliotecas del entorno virtual con paquetes externos expuestos mediante `PYTHONPATH`.

Si el encoder ya está descargado, puede indicarse su carpeta. Debe corresponder a la revisión fijada en `pipeline.py`:

```bash
bash ejecutar.sh --model-path /ruta/al/modelo
```

## Qué hace el código

1. Descarga y verifica por SHA-256 el corpus y las stopwords.
2. Selecciona los archivos con el patrón `ID_YYYYMMDD[_parte].tbf`.
3. Recupera las palabras de los árboles sintácticos y agrupa oraciones consecutivas hasta **128 tokens**, incluidos los especiales. No hay solapamiento.
4. Calcula embeddings con `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
5. Reduce los embeddings de 384 a 5 dimensiones con UMAP y agrupa con HDBSCAN.
6. Describe los tópicos con c-TF-IDF y `CountVectorizer`, usando **313 stopwords en español** y unigramas/bigramas.
7. Compara tres configuraciones y repite cada ajuste para comprobar reproducibilidad.
8. Exporta asignaciones, tres representantes por tópico, métricas, modelos ligeros y visualizaciones.

Las stopwords se eliminan solo en el vectorizador; el encoder conserva el contexto. Se fija `random_state=42` en UMAP. Los nombres de las figuras son interpretaciones de los tópicos de este corpus; deben revisarse si se cambian datos o parámetros.

## Resultados

| Configuración | Selección HDBSCAN | Tamaño mínimo | Tópicos | Outliers | Silueta |
|---|---|---:|---:|---:|---:|
| A | eom | 30 | 7 | 15.1% | 0.087 |
| B | leaf | 30 | 11 | 53.1% | 0.103 |
| C | leaf | 60 | 5 | 28.4% | 0.082 |

En las tres se mantienen `min_samples=10`, `ngram_range=(1,2)` y `nr_topics=None`. La silueta se calcula con distancia coseno en los embeddings originales de 384 dimensiones, excluyendo el ruido. No es una medida de exactitud de clasificación.

Se eligió **A** por su cobertura de **84.9%** y sus grupos interpretables. Los siete nombres propuestos son deportes; economía, empresas y cooperación internacional; gobierno y partidos; asilo y autoridades; Unión Europea; elecciones estadounidenses de 2000; y minas y conflictos.

Los **207 outliers** se conservan como `-1` en el resultado principal. El ensayo de reasignación con similitud mínima 0.65 se guarda por separado. En A solo reasigna cinco segmentos.

Hay errores que requieren revisión: por ejemplo, el grupo de minas mezcla desminado con un segmento sobre minas de carbón. La semilla fija permite repetir la ejecución, pero no demuestra estabilidad ante otros corpus o semillas.

## Archivos y salidas

- `pipeline.py`: implementación completa.
- `instalar.sh` y `ejecutar.sh`: comandos de instalación y ejecución.
- `requirements.txt` y `environment.lock.txt`: versiones utilizadas.
- `results/metrics.json`: métricas del experimento ejecutado.
- `results/corpus_stats.json`: estadísticas observadas de la segmentación.
- `results/environment.json`: entorno y revisión del encoder.
- `images/`: dos figuras del experimento.

Al ejecutar se generan `data/`, tablas completas en `results/`, modelos ligeros, seis gráficas HTML en `interactive/` y las dos figuras de `images/`. El archivo `results/run_complete.json` aparece solo cuando termina correctamente.

El entorno virtual, los pesos del encoder, los textos completos del corpus y los archivos grandes generados no se publican en este repositorio. El script recupera las fuentes y reconstruye los resultados. Los modelos ligeros guardan términos y representaciones; para repetir UMAP y HDBSCAN hay que ejecutar el pipeline.

## Fuente y bibliografía

CESS-ESP pertenece a CLiC-UB, UA, UPC y EHU-UPV, según su README. Se utiliza para un ejercicio académico y se descarga de [NLTK data](https://github.com/nltk/nltk_data). Su disponibilidad en NLTK no convierte los textos periodísticos en dominio público.

- Martí, M. A., Taulé, M., Márquez, L. y Bertran, M. (2007). *CESS-ECE: A Multilingual and Multilevel Annotated Corpus*. Referencia solicitada en el README del corpus.
- Grootendorst, M. (2022). [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794).
- [Tarjeta del encoder multilingüe](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
- [Documentación de BERTopic](https://maartengr.github.io/BERTopic/).

## Uso de IA

Se utilizó Codex (OpenAI) para búsqueda de documentación, implementación, depuración, nombres de tópicos y redacción. Las métricas y asignaciones proceden de ejecuciones reales. El estudiante debe revisar las decisiones y la interpretación. No se ejecutó una comparación con una API generativa; el experimento no requiere una clave ni gasto de API.
