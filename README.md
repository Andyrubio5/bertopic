# BERTopic

Análisis de noticias en español con BERTopic para la actividad de modelado de tópicos.

## Ejecutar

Con Python 3.12 y CPU:

```bash
git clone https://github.com/Andyrubio5/bertopic.git
cd bertopic
python3 -m venv .venv
.venv/bin/python -I -m pip install -r requirements.txt
.venv/bin/python -I pipeline.py
```

La primera ejecución necesita internet para descargar CESS-ESP, las stopwords y el encoder multilingüe. No necesita una clave de API.

El código compara tres configuraciones con los mismos embeddings y `random_state=42`. El resultado principal tiene **7 tópicos y 15.1% de outliers**, sobre **1370 segmentos**.

Al ejecutar se guardan en `resultados/` la comparación, las tablas de tópicos, los segmentos y tres ejemplos por tópico de A. Las dos gráficas se guardan en `images/`. No se generan JSON ni se guardan modelos adicionales.

Fuente: [CESS-ESP distribuido por NLTK](https://github.com/nltk/nltk_data), con atribución a Martí, Taulé, Márquez y Bertran (2007). Los textos se recuperan desde su fuente para este ejercicio académico.
