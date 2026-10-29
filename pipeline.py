"""BERTopic en español. Ejecutar: python pipeline.py"""
import argparse
import hashlib
import os
from pathlib import Path
import random
import re
import tempfile
import zipfile

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bertopic-mpl"))

import numpy as np
import pandas as pd
import requests
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

ROOT = Path(__file__).resolve().parent
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
SEED = 42
SOURCES = {
    "cess_esp": "ae5b12898039e51911ae16d25c4822cb92adcfc034a2e12b57676d21d3c94884",
    "stopwords": "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
}
CONFIGS = {
    "A": {"min_cluster_size": 30, "min_samples": 10, "selection": "eom", "ngram_range": (1, 2), "nr_topics": None},
    "B": {"min_cluster_size": 30, "min_samples": 10, "selection": "leaf", "ngram_range": (1, 2), "nr_topics": None},
    "C": {"min_cluster_size": 60, "min_samples": 10, "selection": "leaf", "ngram_range": (1, 2), "nr_topics": None},
}
DETOK = TreebankWordDetokenizer()
PUNCT = {"-Fpa-": "(", "-Fpt-": ")", "-Fca-": "[", "-Fct-": "]", "-Fe-": '"'}


def sha256(path):
    """Calcula el hash SHA-256 del contenido de un archivo."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def acquire():
    """Descarga, verifica y extrae el corpus y las palabras vacías en español."""
    for name, digest in SOURCES.items():
        path = ROOT / "data" / f"{name}.zip"
        if not path.exists():
            url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{name}.zip"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        if sha256(path) != digest:
            raise ValueError(f"La fuente {name} cambió. Revisar antes de actualizar el hash.")
    with zipfile.ZipFile(ROOT / "data/stopwords.zip") as z:
        (ROOT / "data/stopwords_es.txt").write_bytes(z.read("stopwords/spanish"))
    with zipfile.ZipFile(ROOT / "data/cess_esp.zip") as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Ruta insegura en el ZIP")
        z.extractall(ROOT / "data/raw")
        (ROOT / "data/README_CESS.txt").write_bytes(z.read("cess_esp/README"))

def clean_sentence(words):
    """Limpia marcadores sintácticos y recompone una oración a partir de sus tokens."""
    clean = []
    for word in words:
        if re.fullmatch(r"\*.*\*", word):
            continue
        clean.extend(PUNCT.get(word, word).replace("_", " ").split())
    return DETOK.detokenize(clean)

def prepare(tokenizer, max_length):
    """Agrupa oraciones sin superar el límite de tokens del modelo."""
    reader = BracketParseCorpusReader(str(ROOT / "data/raw/cess_esp"), r".*\.tbf", encoding="latin-1")
    all_files = sorted(reader.fileids())
    files = [f for f in all_files if re.fullmatch(r"\d+_\d{8}(?:_\d+)?\.tbf", f)]
    records, n_sentences, n_split, n_short, n_duplicate = [], 0, 0, 0, 0
    seen = set()

    def n_tokens(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])

    for file_id in files:
        current, sentence_ids, segment_index = "", [], 0

        def emit():
            """Guarda el segmento acumulado si cumple los mínimos de longitud y unicidad."""
            nonlocal current, sentence_ids, segment_index, n_short, n_duplicate
            if not current:
                return
            n_words = len(re.findall(r"\b\w+\b", current))
            if n_words < 8:
                n_short += 1
            elif current in seen:
                n_duplicate += 1
            else:
                seen.add(current)
                article_id = "_".join(Path(file_id).stem.split("_")[:2])
                records.append({"segment_id": f"{file_id}:{segment_index:03}", "file_id": file_id, "article_id": article_id,
                                "sentence_ids": ",".join(map(str, sorted(set(sentence_ids)))),
                                "text": current, "words": n_words, "model_tokens": n_tokens(current)})
            segment_index += 1
            current, sentence_ids = "", []

        for sid, sentence in enumerate(reader.sents(file_id)):
            n_sentences += 1
            text = clean_sentence(sentence)
            pieces = []
            if n_tokens(text) > max_length:
                n_split += 1
                chunk = ""
                for word in text.split():
                    candidate = (chunk + " " + word).strip()
                    if chunk and n_tokens(candidate) > max_length:
                        pieces.append(chunk)
                        chunk = word
                    else:
                        chunk = candidate
                if chunk:
                    pieces.append(chunk)
            else:
                pieces = [text]
            for piece in pieces:
                candidate = (current + " " + piece).strip()
                if current and n_tokens(candidate) > max_length:
                    emit()
                current = (current + " " + piece).strip()
                sentence_ids.append(sid)
        emit()

    frame = pd.DataFrame(records)
    assert frame.model_tokens.max() <= max_length
    assert not frame.text.duplicated().any()
    print(f"Corpus: {len(files)} archivos, {n_sentences} oraciones y {len(frame)} segmentos.")
    print(f"Palabras por segmento: media {frame.words.mean():.1f}; mínimo {frame.words.min()}; máximo {frame.words.max()}.")
    print(f"Oraciones divididas: {n_split}. Segmentos cortos: {n_short}. Duplicados: {n_duplicate}.")
    return frame


def make_model(config, stopwords):
    """Construye BERTopic con los parámetros indicados y componentes reproducibles."""
    return BERTopic(
        language="multilingual",
        embedding_model=None,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=SEED, transform_seed=SEED, n_jobs=1),
        hdbscan_model=HDBSCAN(min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"],
                              metric="euclidean", cluster_selection_method=config["selection"], prediction_data=True,
                              core_dist_n_jobs=1),
        vectorizer_model=CountVectorizer(stop_words=stopwords, ngram_range=config["ngram_range"],
                                         min_df=1, strip_accents=None,
                                         token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b"),
        nr_topics=config["nr_topics"], calculate_probabilities=False, verbose=True,
    )

def export_representatives(frame, labels, embeddings):
    """Guarda tres segmentos representativos de cada tópico de A."""
    representatives = []
    used_files = set()
    for topic in sorted(set(labels)-{-1}):
        indices = np.flatnonzero(labels == topic)
        centroid = embeddings[indices].mean(axis=0, keepdims=True)
        scores = cosine_similarity(embeddings[indices], centroid).ravel()
        ranked = indices[np.argsort(-scores)]
        chosen = []
        for idx in ranked:
            f = frame.iloc[idx].article_id
            if f in used_files:
                continue
            chosen.append(int(idx)); used_files.add(f)
            if len(chosen)==3:
                break
        if len(chosen)<3:
            for idx in ranked:
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                if len(chosen)==3:
                    break
        for rank, idx in enumerate(chosen,1):
            row = frame.iloc[idx].to_dict()
            row.update(topic=int(topic), rank=rank,
                       cosine_to_centroid=float(cosine_similarity(embeddings[idx:idx+1],centroid)[0,0]))
            representatives.append(row)
    pd.DataFrame(representatives).to_csv(ROOT / "resultados/ejemplos.csv",index=False)

def export_static_figures(frame, labels, embeddings):
    """Regenera las dos imágenes citadas en main.tex sin duplicarlas en otra carpeta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {0: "Deportes", 1: "Economía e internacional", 2: "Gobierno y partidos",
             3: "Asilo y autoridades", 4: "Unión Europea", 5: "Elecciones de EE. UU.",
             6: "Minas y conflictos"}
    topic_ids = sorted(set(labels) - {-1})
    palette = ["#147D92", "#E39A32", "#557BC1", "#A86298", "#77A66A", "#BD615F", "#86674D"]
    colors = [palette[i % len(palette)] for i in range(len(topic_ids))]
    counts = [int((labels == t).sum()) for t in topic_ids] + [int((labels == -1).sum())]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.barh(range(len(counts)), counts, color=colors + ["#A1A7AB"])
    ax.set_yticks(range(len(counts)), [f"T{t} {names.get(int(t), str(t))}" for t in topic_ids] + ["-1 Sin tópico / ruido"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.3)
    for y, count in enumerate(counts):
        ax.text(count + 6, y, f"{count} ({count / len(frame) * 100:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel(f"Número de segmentos (denominador: {len(frame)})")
    ax.set_title("Cobertura de la configuración A")
    fig.savefig(ROOT / "images/distribution.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not topic_ids:
        raise ValueError("No hay tópicos para la figura de similitud.")
    centroids = np.vstack([embeddings[labels == t].mean(axis=0) for t in topic_ids])
    similarity = cosine_similarity(centroids)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(similarity, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_yticks(range(len(topic_ids)), [f"T{t} {names.get(int(t), str(t))}" for t in topic_ids], fontsize=9)
    ax.set_xticks(range(len(topic_ids)), [f"T{t}" for t in topic_ids])
    for i in range(len(topic_ids)):
        for j in range(len(topic_ids)):
            ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center",
                    color="white" if similarity[i, j] > 0.65 else "#132D40", fontsize=9)
    ax.set_title("Similitud coseno entre centroides (384D)")
    fig.colorbar(im, ax=ax, label="Coseno (no es probabilidad)")
    fig.savefig(ROOT / "images/similarity.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main(model_path=None):
    for directory in ["data", "resultados", "images"]:
        (ROOT / directory).mkdir(exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    # 1. Descargar, limpiar y segmentar las noticias.
    acquire()
    encoder = SentenceTransformer(model_path or MODEL,
        **({} if model_path else {"revision": REVISION}), device="cpu")
    frame = prepare(encoder.tokenizer, encoder.max_seq_length)
    documents = frame.text.tolist()
    stopwords = sorted(set((ROOT / "data/stopwords_es.txt").read_text(encoding="utf-8").split()))

    # 2. Calcular los mismos embeddings para todas las pruebas.
    embeddings = encoder.encode(documents, batch_size=32,
        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)

    # 3. Comparar las configuraciones y guardar sus tablas de tópicos.
    comparison = []
    for name, config in CONFIGS.items():
        model = make_model(config, stopwords)
        topics, _ = model.fit_transform(documents, embeddings=embeddings)
        labels = np.asarray(topics)
        model.get_topic_info().to_csv(ROOT / f"resultados/topicos_{name}.csv", index=False)
        mask = labels != -1
        silhouette = silhouette_score(embeddings[mask], labels[mask], metric="cosine")
        reassigned = model.reduce_outliers(documents, topics, strategy="embeddings",
            embeddings=embeddings, threshold=0.65) if (~mask).any() else topics
        repeat, _ = make_model(config, stopwords).fit_transform(documents, embeddings=embeddings)
        comparison.append({
            "Configuracion": name, "Seleccion": config["selection"],
            "Tamano_minimo": config["min_cluster_size"], "Topicos": len(set(topics) - {-1}),
            "Outliers": int((~mask).sum()), "Porcentaje_outliers": float((~mask).mean() * 100),
            "Silueta": float(silhouette), "Reasignados_065": int(np.sum(labels != reassigned)),
            "Repeticion_identica": bool(np.array_equal(labels, repeat)),
        })

        # 4. Conservar los segmentos y ejemplos de la configuración principal A.
        if name == "A":
            frame.assign(topic=labels).to_csv(ROOT / "resultados/segmentos.csv", index=False)
            export_representatives(frame, labels, embeddings)
            export_static_figures(frame, labels, embeddings)

    # 5. Mostrar y guardar la comparación, sin JSON ni modelos adicionales.
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(ROOT / "resultados/comparacion.csv", index=False)
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de noticias en español con BERTopic")
    parser.add_argument("--model-path", default=None, help="Carpeta opcional del encoder descargado")
    main(parser.parse_args().model_path)
