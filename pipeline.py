
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import re
import time
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
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

# Directorio del proyecto; todos los archivos de entrada y salida se resuelven desde aquí.
ROOT = Path(__file__).resolve().parent
# Modelo multilingüe usado para generar las representaciones vectoriales de los textos.
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Revisión fija del modelo para que las ejecuciones puedan reproducirse.
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
# Semilla compartida por los componentes aleatorios del proceso.
SEED = 42
# Hashes esperados para detectar cambios inesperados en las fuentes descargadas.
SOURCES = {
    "cess_esp": "ae5b12898039e51911ae16d25c4822cb92adcfc034a2e12b57676d21d3c94884",
    "stopwords": "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
}
# Variantes de parámetros que se comparan en la ejecución.
CONFIGS = {
    "A": {"min_cluster_size": 30, "min_samples": 10, "selection": "eom", "ngram_range": (1, 2), "nr_topics": None},
    "B": {"min_cluster_size": 30, "min_samples": 10, "selection": "leaf", "ngram_range": (1, 2), "nr_topics": None},
    "C": {"min_cluster_size": 60, "min_samples": 10, "selection": "leaf", "ngram_range": (1, 2), "nr_topics": None},
}


def save_json(path, value):
    """Guarda un valor como JSON legible y conserva caracteres Unicode."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    """Calcula el hash SHA-256 del contenido de un archivo."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire():
    """Descarga, verifica y extrae el corpus y las palabras vacías en español."""
    for name, digest in SOURCES.items():
        path = ROOT / "data" / f"{name}.zip"
        # Solo descarga el archivo cuando todavía no existe localmente.
        if not path.exists():
            url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{name}.zip"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        # No se usan datos cuya integridad no coincida con el hash fijado en el proyecto.
        if sha256(path) != digest:
            raise ValueError(f"La fuente {name} cambió. Revisar antes de actualizar el hash.")
    # Extrae la lista de palabras vacías directamente desde el archivo verificado.
    with zipfile.ZipFile(ROOT / "data/stopwords.zip") as z:
        (ROOT / "data/stopwords_es.txt").write_bytes(z.read("stopwords/spanish"))
    with zipfile.ZipFile(ROOT / "data/cess_esp.zip") as z:
        # Rechaza rutas absolutas o con segmentos ".." antes de extraer el contenido.
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Ruta insegura en el ZIP")
        z.extractall(ROOT / "data/raw")
        # Copia el README del corpus para conservar su documentación junto a los datos.
        (ROOT / "data/README_CESS.txt").write_bytes(z.read("cess_esp/README"))


# Conversor de tokens del corpus a texto legible y traducción de sus marcadores de puntuación.
DETOK = TreebankWordDetokenizer()
PUNCT = {"-Fpa-": "(", "-Fpt-": ")", "-Fca-": "[", "-Fct-": "]", "-Fe-": '"'}


def clean_sentence(words):
    """Limpia marcadores sintácticos y recompone una oración a partir de sus tokens."""
    clean = []
    for word in words:
        if re.fullmatch(r"\*.*\*", word):
            # Los elementos entre asteriscos son anotaciones, no palabras del texto.
            continue
        # Traduce marcadores especiales, sustituye guiones bajos y separa subcomponentes.
        clean.extend(PUNCT.get(word, word).replace("_", " ").split())
    return DETOK.detokenize(clean)


def prepare(tokenizer, max_length):
    """Convierte los archivos del corpus en segmentos listos para modelar y registra estadísticas."""
    reader = BracketParseCorpusReader(str(ROOT / "data/raw/cess_esp"), r".*\.tbf", encoding="latin-1")
    all_files = sorted(reader.fileids())
    # Acepta únicamente archivos de artículos con el formato esperado y excluye suplementos.
    files = [f for f in all_files if re.fullmatch(r"\d+_\d{8}(?:_\d+)?\.tbf", f)]
    # Contadores usados para describir qué se incluyó y qué se descartó del corpus.
    records, n_sentences, n_split, n_short, n_duplicate = [], 0, 0, 0, 0
    seen = set()

    def n_tokens(text):
        # Cuenta tokens del modelo sin truncar para poder controlar el límite explícitamente.
        return len(tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])

    for file_id in files:
        # Cada archivo se procesa por separado; los segmentos no cruzan límites de artículo.
        current, sentence_ids, segment_index = "", [], 0

        def emit():
            """Guarda el segmento acumulado si cumple los mínimos de longitud y unicidad."""
            nonlocal current, sentence_ids, segment_index, n_short, n_duplicate
            if not current:
                return
            n_words = len(re.findall(r"\b\w+\b", current))
            # Descarta segmentos demasiado breves y duplicados para evitar ruido en el análisis.
            if n_words < 8:
                n_short += 1
            elif current in seen:
                n_duplicate += 1
            else:
                seen.add(current)
                # El identificador de artículo agrupa variantes del mismo documento fuente.
                article_id = "_".join(Path(file_id).stem.split("_")[:2])
                records.append({"segment_id": f"{file_id}:{segment_index:03}", "file_id": file_id, "article_id": article_id,
                                "sentence_ids": ",".join(map(str, sorted(set(sentence_ids)))),
                                "text": current, "words": n_words, "model_tokens": n_tokens(current)})
            # Avanza el identificador aunque el segmento se descarte y limpia el acumulador.
            segment_index += 1
            current, sentence_ids = "", []

        for sid, sentence in enumerate(reader.sents(file_id)):
            n_sentences += 1
            text = clean_sentence(sentence)
            pieces = []
            if n_tokens(text) > max_length:
                n_split += 1
                # Divide oraciones largas solo entre palabras, sin perder ni repetir contenido.
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
                # Cierra el segmento antes de agregar texto que excedería el máximo del modelo.
                if current and n_tokens(candidate) > max_length:
                    emit()
                current = (current + " " + piece).strip()
                sentence_ids.append(sid)
        # Emite el último segmento pendiente del archivo.
        emit()

    # Convierte los registros en tabla y comprueba los límites y la unicidad establecidos.
    frame = pd.DataFrame(records)
    assert frame.model_tokens.max() <= max_length
    assert not frame.text.duplicated().any()
    frame.to_csv(ROOT / "data/segments.csv", index=False)
    # Resume el tamaño y las decisiones de filtrado para facilitar auditorías de cada ejecución.
    stats = {"all_tbf_files": len(all_files), "selected_files": len(files),
             "excluded_files": len(all_files)-len(files), "article_families": int(frame.article_id.nunique()),
             "source_sentences": n_sentences,
             "long_sentences_split": n_split, "short_segments_removed": n_short,
             "duplicate_segments_removed": n_duplicate, "segments": len(frame),
             "total_words": int(frame.words.sum()), "words": frame.words.describe().to_dict(),
             "model_tokens": frame.model_tokens.describe().to_dict(), "max_length": max_length,
             "source_file_ids": files, "segments_sha256": sha256(ROOT / "data/segments.csv")}
    save_json(ROOT / "results/corpus_stats.json", stats)
    return frame, stats


def make_model(config, stopwords):
    """Construye BERTopic con los parámetros indicados y componentes reproducibles."""
    return BERTopic(
        # El modo multilingüe conserva tildes y ñ durante el preprocesamiento de términos.
        language="multilingual",
        embedding_model=None,
        # UMAP reduce las dimensiones de los embeddings antes del agrupamiento.
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=SEED, transform_seed=SEED, n_jobs=1),
        # HDBSCAN identifica grupos y permite recuperar sus probabilidades de pertenencia.
        hdbscan_model=HDBSCAN(min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"],
                              metric="euclidean", cluster_selection_method=config["selection"], prediction_data=True,
                              core_dist_n_jobs=1),
        # El vectorizador determina los términos que describen cada tema.
        vectorizer_model=CountVectorizer(stop_words=stopwords, ngram_range=config["ngram_range"],
                                         min_df=1, strip_accents=None,
                                         token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b"),
        nr_topics=config["nr_topics"], calculate_probabilities=False, verbose=True,
    )


def evaluate(model, labels, embeddings):
    """Calcula métricas de temas, documentos atípicos, cobertura y diversidad."""
    labels = np.asarray(labels)
    # BERTopic asigna -1 a documentos que no pertenecen a ningún tema.
    topic_ids = sorted(set(labels)-{-1})
    mask = labels != -1
    # Obtiene hasta diez términos por tema para medir la diversidad del vocabulario.
    terms = [[w for w, _ in model.get_topic(int(t))[:10] if w] for t in topic_ids]
    diversity = len(set(w for topic in terms for w in topic))/sum(map(len, terms))
    # Silhouette solo tiene sentido si hay al menos dos temas y se excluyen los atípicos.
    score = silhouette_score(embeddings[mask], labels[mask], metric="cosine") if len(topic_ids)>1 else None
    return {"topics": len(topic_ids), "outliers": int((~mask).sum()),
            "outlier_pct": float((~mask).mean()*100), "coverage_pct": float(mask.mean()*100),
            "silhouette_cosine_384d_without_outliers": float(score) if score is not None else None,
            "topic_diversity_top10": diversity,
            "topic_sizes": {str(t): int((labels==t).sum()) for t in topic_ids}}


def export_representatives(model, frame, labels, embeddings, name):
    """Exporta documentos representativos de cada tema y guarda la selección nativa de BERTopic."""
    # Conserva la selección nativa de BERTopic como referencia para auditorías.
    save_json(ROOT / f"results/representatives_native_{name}.json", model.get_representative_docs())
    representatives = []
    # Intenta usar artículos distintos como representantes mientras haya candidatos disponibles.
    used_files = set()
    for topic in sorted(set(labels)-{-1}):
        # Ordena los documentos del tema por similitud con el centroide de sus embeddings.
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
        # Si no hay tres artículos distintos, completa con los mejores documentos restantes.
        if len(chosen)<3:
            for idx in ranked:
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                if len(chosen)==3:
                    break
        # Registra el orden de selección y la similitud con el centroide.
        for rank, idx in enumerate(chosen,1):
            row = frame.iloc[idx].to_dict()
            row.update(topic=int(topic), rank=rank,
                       cosine_to_centroid=float(cosine_similarity(embeddings[idx:idx+1],centroid)[0,0]))
            representatives.append(row)
    pd.DataFrame(representatives).to_csv(ROOT / f"results/representatives_{name}.csv",index=False)


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
    """Ejecuta la preparación, modelado, evaluación y exportación de resultados."""
    # Crea las carpetas necesarias sin borrar resultados previos.
    for d in ["data", "results", "images", "interactive"]:
        (ROOT/d).mkdir(exist_ok=True)
    # Una ejecución interrumpida no debe conservar una marca de éxito anterior.
    (ROOT / "results/run_complete.json").unlink(missing_ok=True)
    # Fija las fuentes principales de aleatoriedad para facilitar la repetición del experimento.
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    acquire()
    # Carga el codificador en CPU y usa una revisión concreta salvo que se indique una ruta local.
    encoder = SentenceTransformer(model_path or MODEL, **({} if model_path else {"revision": REVISION}), device="cpu")
    frame, stats = prepare(encoder.tokenizer, encoder.max_seq_length)
    docs = frame.text.tolist()
    # La huella permite reutilizar embeddings solo si coinciden el modelo y los segmentos.
    fingerprint = {"model": MODEL, "revision": REVISION, "segments_sha256": stats["segments_sha256"],
                   "normalize_embeddings": True}
    ep = ROOT / "data/embeddings.npy"
    fp = ROOT / "data/embedding_manifest.json"
    if ep.exists() and fp.exists() and json.loads(fp.read_text())==fingerprint:
        # Reutiliza el cálculo previo cuando su manifiesto coincide exactamente.
        embeddings = np.load(ep)
    else:
        # Calcula y almacena embeddings normalizados para los segmentos actuales.
        embeddings = encoder.encode(docs, batch_size=32, show_progress_bar=True,
                                    normalize_embeddings=True, convert_to_numpy=True)
        np.save(ep, embeddings); save_json(fp, fingerprint)
    assert embeddings.shape==(len(frame),384)
    # Prepara palabras vacías y contenedores para métricas, etiquetas y modelos.
    stopwords = sorted(set((ROOT / "data/stopwords_es.txt").read_text().split()))
    results, all_labels, models = {}, {}, {}
    for name, config in CONFIGS.items():
        # Ajusta cada variante, evalúa sus temas y guarda los resultados asociados.
        start = time.perf_counter()
        model = make_model(config, stopwords)
        topics, _ = model.fit_transform(docs, embeddings=embeddings)
        labels = np.asarray(topics)
        result = evaluate(model, labels, embeddings)
        result.update(config=config, fit_and_evaluation_seconds=time.perf_counter()-start)
        results[name], all_labels[name], models[name] = result, labels, model
        model.get_topic_info().to_csv(ROOT / f"results/topic_info_{name}.csv",index=False)
        save_json(ROOT / f"results/terms_{name}.json", model.get_topics())
        export_representatives(model,frame,labels,embeddings,name)
        frame.assign(topic=labels).to_csv(ROOT / f"results/assignments_{name}.csv",index=False)
        model.save(str(ROOT / f"results/model_{name}"), serialization="safetensors",
                   save_ctfidf=True, save_embedding_model=MODEL)
        print(name, json.dumps(result,ensure_ascii=False), flush=True)
    # Compara la consistencia de las asignaciones entre pares de configuraciones.
    results["comparison"] = {}
    for first,second in [("A","B"),("B","C"),("A","C")]:
        common = (all_labels[first]!=-1)&(all_labels[second]!=-1)
        results["comparison"][f"{first}_{second}"] = {
            "ari_all_including_noise": float(adjusted_rand_score(all_labels[first],all_labels[second])),
            "ari_common_non_outliers": float(adjusted_rand_score(all_labels[first][common],all_labels[second][common])),
            "common_non_outliers": int(common.sum()),
        }
    # Mantiene las etiquetas -1 originales y mide por separado el efecto de reasignar atípicos.
    for name, model in models.items():
        reassigned = np.asarray(model.reduce_outliers(docs, all_labels[name].tolist(),
                                  strategy="embeddings", embeddings=embeddings, threshold=0.65)) if -1 in all_labels[name] else all_labels[name].copy()
        pd.DataFrame({"segment_id": frame.segment_id, "original": all_labels[name],
                      "reassigned_threshold_065": reassigned}).to_csv(ROOT / f"results/outlier_sensitivity_{name}.csv",index=False)
        results[name]["sensitivity_outliers_after_065"] = int((reassigned==-1).sum())
        results[name]["sensitivity_reassigned"] = int((reassigned!=all_labels[name]).sum())
    # Vuelve a ajustar cada configuración con la misma semilla para comprobar reproducibilidad.
    results["reproducibility"] = {"seed":SEED}
    for name in CONFIGS:
        repeat_model = make_model(CONFIGS[name],stopwords)
        repeat_topics,_ = repeat_model.fit_transform(docs,embeddings=embeddings)
        results["reproducibility"][name] = {"same_assignments": bool(np.array_equal(all_labels[name],repeat_topics)),
            "ari":float(adjusted_rand_score(all_labels[name],repeat_topics))}
    # Guarda métricas, entorno de software y tablas de comparación entre temas.
    save_json(ROOT / "results/metrics.json",results)
    save_json(ROOT / "results/environment.json", {"python":os.sys.version,
              "packages": {p:metadata.version(p) for p in ["bertopic","sentence-transformers","transformers","torch","umap-learn","hdbscan","numpy","pandas","scikit-learn","nltk","matplotlib","reportlab","numba","scipy"]},
              "cpu_threads":4,"stopwords_count":len(stopwords),"model":MODEL,"revision":REVISION})
    pd.crosstab(pd.Series(all_labels["A"],name="A"),pd.Series(all_labels["B"],name="B")).to_csv(ROOT/"results/cross_configuration.csv")
    pd.crosstab(pd.Series(all_labels["B"],name="B"),pd.Series(all_labels["C"],name="C")).to_csv(ROOT/"results/cross_B_C.csv")
    # Genera visualizaciones interactivas de términos e interrelaciones entre temas.
    for name, model in models.items():
        model.visualize_barchart(top_n_topics=12,n_words=6).write_html(str(ROOT/f"interactive/barchart_{name}.html"),include_plotlyjs=True)
        model.visualize_hierarchy().write_html(str(ROOT/f"interactive/hierarchy_{name}.html"),include_plotlyjs=True)
    export_static_figures(frame, all_labels["A"], embeddings)
    # Marca la ejecución como completa solo después de terminar todas las exportaciones.
    save_json(ROOT/"results/run_complete.json",{"complete":True,"segments":len(frame),"seed":SEED})
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
    return results


if __name__=="__main__":
    # Permite ejecutar el pipeline desde la terminal con una ruta opcional al modelo local.
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-path",default=None)
    args=parser.parse_args()
    main(args.model_path)
