import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import openai
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from dotenv import load_dotenv
from google import genai
from google.genai import types
from matplotlib import pyplot as plt
from sentence_transformers import CrossEncoder, SentenceTransformer

# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import dot_score, pairwise_cos_sim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tabulate import tabulate
from tqdm import tqdm

load_dotenv()

BATCH_SIZE = 5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, "classifier_analysis_results", "embeddings_v1")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=False, default=None)
    parser.add_argument("--sim-method", type=str, required=True)
    return parser.parse_args()


def hashlib_hash(args):
    return hashlib.sha256(str(args).encode()).hexdigest()


def compute_scores(N, sim_callback, batch_sim_callback, data, df):
    N = N or len(data)
    if batch_sim_callback is not None:
        truncated_data = data[:N]
        batch = []
        for i, item in enumerate(tqdm(truncated_data)):
            if len(batch) < BATCH_SIZE:
                batch.append(item)
            if len(batch) == BATCH_SIZE or i == len(truncated_data) - 1:
                orig_responses = [
                    item["original_response"]["choices"][0]["message"]["content"]
                    for item in batch
                ]
                repl_responses = [
                    item["replication_response"]["choices"][0]["message"]["content"]
                    for item in batch
                ]
                similarities = batch_sim_callback(orig_responses, repl_responses)
                for j, similarity in enumerate(similarities):
                    df.append(
                        {
                            "prompt": batch[j]["prompt"],
                            "original_response": orig_responses[j],
                            "replication_response": repl_responses[j],
                            "similarity": similarity,
                            "genuine": batch[j]["original_request"]["model"]
                            == batch[j]["replication_request"]["model"],
                            "label": (
                                "Genuine"
                                if batch[j]["original_request"]["model"]
                                == batch[j]["replication_request"]["model"]
                                else "Spoof"
                            ),
                            "inference_machine": batch[j]["inference_machine"],
                            "replication_machine": batch[j]["replication_machine"],
                            "original_model": batch[j]["original_request"]["model"],
                            "replication_model": batch[j]["replication_request"][
                                "model"
                            ],
                        }
                    )
                batch = []

    else:
        for item in tqdm(data[:N]):

            orig_model = item["original_request"]["model"]
            repl_model = item["replication_request"]["model"]

            orig_response = item["original_response"]["choices"][0]["message"][
                "content"
            ]

            repl_response = item["replication_response"]["choices"][0]["message"][
                "content"
            ]

            similarity = sim_callback(orig_response, repl_response)

            df.append(
                {
                    "prompt": item["prompt"],
                    "original_response": orig_response,
                    "replication_response": repl_response,
                    "similarity": similarity,
                    "genuine": orig_model == repl_model,
                    "label": "Genuine" if orig_model == repl_model else "Spoof",
                    "inference_machine": item["inference_machine"],
                    "replication_machine": item["replication_machine"],
                    "original_model": orig_model,
                    "replication_model": repl_model,
                }
            )


def cosine_similarity_callback(model):
    def callback(orig_response, repl_response):
        orig_embedding = model.encode([orig_response])  # -> [1, D]
        repl_embedding = model.encode([repl_response])  # -> [1, D]
        similarity = pairwise_cos_sim(
            orig_embedding,
            repl_embedding,
        )
        similarity = similarity.numpy()
        assert np.prod(similarity.shape) == 1
        return float(similarity[0])

    return callback


def cross_encoder_similarity_callback(model):
    def callback(orig_response, repl_response):
        similarity = model.predict([orig_response, repl_response])
        assert np.prod(similarity.shape) == 1
        return float(similarity)

    return callback


def dot_similarity_callback(model):
    def callback(orig_response, repl_response):
        orig_embedding = model.encode([orig_response])  # -> [1, D]
        repl_embedding = model.encode([repl_response])  # -> [1, D]
        similarity = dot_score(orig_embedding, repl_embedding)
        similarity = similarity.numpy()
        assert np.prod(similarity.shape) == 1
        return float(similarity[0])

    return callback


def nomic_cosine_similarity_callback(model, kind1, kind2):
    def callback(orig_response, repl_response):
        orig_embedding = model.encode([orig_response], prompt_name=kind1)
        repl_embedding = model.encode([repl_response], prompt_name=kind2)
        similarity = model.similarity(orig_embedding[0], repl_embedding[0])
        similarity = similarity.numpy()
        return float(similarity[0][0])

    return callback


CLIENTS = {}
CACHE = {}


def dump_cache():
    with open("cache.json", "w") as f:
        json.dump(CACHE, f)


def load_cache():
    global CACHE
    if os.path.exists("cache.json"):
        with open("cache.json", "r") as f:
            CACHE = json.load(f)


def gemini_batch_similarity_callback(model):
    def callback(orig_responses, repl_responses):
        if "gemini" not in CLIENTS:
            CLIENTS["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
        client = CLIENTS["gemini"]

        cached_orig_responses, cached_repl_responses = [], []
        non_cached_orig_responses, non_cached_repl_responses = [], []

        for orig_response, repl_response in zip(orig_responses, repl_responses):
            key = hashlib_hash(("gemini", model, orig_response, repl_response))
            if key not in CACHE:
                non_cached_orig_responses.append(orig_response)
                non_cached_repl_responses.append(repl_response)
            else:
                cached_orig_responses.append(orig_response)
                cached_repl_responses.append(repl_response)

        if len(non_cached_orig_responses) == 0:
            embeddings = []
        else:
            print("Invoking API with batch of ", len(non_cached_orig_responses))
            embeddings = client.models.embed_content(
                model=model,
                contents=non_cached_orig_responses + non_cached_repl_responses,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            ).embeddings
            embeddings = [r.values for r in embeddings]

        print("len(embeddings)", len(embeddings))

        sims = []
        for i in range(len(non_cached_orig_responses)):
            sim = pairwise_cos_sim(
                [embeddings[i]], [embeddings[i + len(non_cached_orig_responses)]]
            )
            sim = float(sim[0])
            sims.append(sim)
            key = hashlib_hash(
                (
                    "gemini",
                    model,
                    non_cached_orig_responses[i],
                    non_cached_repl_responses[i],
                )
            )

            CACHE[key] = sim
        for i in range(len(cached_orig_responses)):
            key = hashlib_hash(
                ("gemini", model, cached_orig_responses[i], cached_repl_responses[i])
            )
            sims.append(CACHE[key])

        if len(non_cached_orig_responses) > 0:
            time.sleep(20)

        dump_cache()

        return sims

    return callback


def gemini_similarity_callback(model):
    def callback(orig_response, repl_response):
        if "gemini" not in CLIENTS:
            CLIENTS["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
        client = CLIENTS["gemini"]

        embeddings = client.models.embed_content(
            model=model,
            contents=[orig_response, repl_response],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        ).embeddings

        emb1, emb2 = ([r.values] for r in embeddings)  # each r is types.Embedding

        key = hashlib_hash(("gemini", model, orig_response, repl_response))

        if key not in CACHE:
            CACHE[key] = [emb1, emb2]
        else:
            emb1, emb2 = CACHE[key]

        emb1, emb2 = np.asarray(emb1, dtype=float), np.asarray(emb2, dtype=float)

        print("emb1", emb1)
        print("emb2", emb2)
        similarity = pairwise_cos_sim(emb1, emb2)
        similarity = similarity.numpy()
        assert np.prod(similarity.shape) == 1
        return float(similarity[0])

    return callback


def openai_similarity_callback(model):
    if "openai" not in CLIENTS:
        CLIENTS["openai"] = openai.Client(api_key=OPENAI_API_KEY)

    def callback(orig_response, repl_response):
        client = CLIENTS["openai"]
        key = hashlib_hash(("openai", model, orig_response, repl_response))
        if key in CACHE:
            return CACHE[key]
        embeddings = client.embeddings.create(
            input=[orig_response, repl_response], model=model
        )
        emb1_data, emb2_data = embeddings.data
        emb1 = np.asarray([emb1_data.embedding], dtype=float)
        emb2 = np.asarray([emb2_data.embedding], dtype=float)
        similarity = pairwise_cos_sim(emb1, emb2)
        similarity = similarity.numpy()
        assert np.prod(similarity.shape) == 1
        similarity_val = float(similarity[0])
        if key not in CACHE:
            CACHE[key] = similarity_val
        return similarity_val

    return callback


def make_batch_sim_callback(sim_method: str):
    family, model = sim_method.split("/")
    if family == "gemini":
        return gemini_batch_similarity_callback(model)
    else:
        return None


def make_sim_callback(sim_method: str):
    family, model = sim_method.split("/")
    if family == "cosine":
        model = SentenceTransformer(f"sentence-transformers/{model}")
        return cosine_similarity_callback(model)
    elif family == "cross-encoder":
        model = CrossEncoder(f"cross-encoder/{model}")
        return cross_encoder_similarity_callback(model)
    elif family == "dot":
        model = SentenceTransformer(f"sentence-transformers/{model}")
        return dot_similarity_callback(model)
    elif family == "nomic-ai":
        kind1, kind2, model_name = model.split(";")
        model = SentenceTransformer(f"nomic-ai/{model_name}", trust_remote_code=True)
        return nomic_cosine_similarity_callback(model, kind1, kind2)
    elif family == "gemini":
        return gemini_similarity_callback(model)
    elif family == "openai":
        return openai_similarity_callback(model)
    else:
        raise ValueError(f"Invalid sim_method: {sim_method}")


def analyze_results(args, name, df, score_column, write_to_disk=True):
    """Main analysis function that calls individual analysis components."""
    out_dir = os.path.join(OUTPUT_DIR, args.sim_method.replace("/", "_"), name)
    os.makedirs(out_dir, exist_ok=True)

    # Plot and save individual visualizations
    roc_display = plot_roc_curve(df, score_column)
    plt.savefig(os.path.join(out_dir, f"roc_curve.png"))
    roc_display.plot()

    plot_score_histogram(df, score_column, name)
    plt.savefig(os.path.join(out_dir, f"score_hist.png"))
    plt.show()

    # Generate and save summary
    summary = generate_summary(df, score_column)
    print(name)
    print(summary)

    make_summary_page(out_dir, summary)


def write_summary(df, score_column):
    y_true = df["genuine"]
    y_score = df[score_column]
    roc_data = calculate_roc_metrics(y_true, y_score)
    best_threshold = find_best_threshold(roc_data)
    summary = generate_summary(df, score_column)
    return summary


def calculate_roc_metrics(y_true, y_score):
    """Calculate ROC curve metrics."""
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, threshes = roc_curve(y_true, y_score)

    return {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr, "threshes": threshes}


def plot_roc_curve(df, score_column):

    # y_true = df["genuine"]
    y_true = ~df["genuine"]
    y_score = 1 - df[score_column]

    roc_data = calculate_roc_metrics(y_true, y_score)
    raw_best_threshold = find_best_threshold(roc_data)
    best_threshold = 1 - raw_best_threshold

    best_idx = np.argmin(np.abs(roc_data["threshes"] - raw_best_threshold))
    thresh_x, thresh_y = roc_data["fpr"][best_idx], roc_data["tpr"][best_idx]

    fig, ax = plt.subplots()
    ax.plot(roc_data["fpr"], roc_data["tpr"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC = {roc_data['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], color="r", linestyle="--")
    ax.plot(thresh_x, thresh_y, "o", color="r", markersize=10)

    ax.legend()


def plot_score_histogram(df, score_column, name, ax=None, selected_threshold=None):
    """Plot and save the histogram of similarity scores."""

    if ax is None:
        _, ax = plt.subplots()
    else:
        pass

    hplot = sns.histplot(
        df, x=score_column, hue="label", ax=ax, element="step", stat="count"
    )

    roc_data = calculate_roc_metrics(df["genuine"], df[score_column])
    thresh = selected_threshold or find_best_threshold(roc_data)

    hplot.axvline(x=thresh, color="r", linestyle="--", label="Threshold")
    if score_column == "similarity":
        title = f"Cosine Similarity Scores ({name.replace('_', ' ').title()})"
    else:
        title = f"{name.replace('_', ' ').title()} {score_column}"
    hplot.set_title(title)


def find_best_threshold(roc_data):
    """Find the best threshold based on ROC curve data."""
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    threshes = roc_data["threshes"]
    f1_scores = 2 * (tpr * (1 - fpr)) / ((tpr + (1 - fpr)) + 1e-10)

    best_thresh_idx = np.argmax(f1_scores)
    thresh = threshes[best_thresh_idx]

    return thresh


def generate_summary(df, score_column):
    """Generate a text summary of the classification results."""

    y_true = ~df["genuine"]
    y_score = 1 - df[score_column]

    roc_data = calculate_roc_metrics(y_true, y_score)
    best_threshold = find_best_threshold(roc_data)

    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    threshes = roc_data["threshes"]

    best_thresh_idx = np.argmin(np.abs(best_threshold - threshes))
    thresh = threshes[best_thresh_idx]

    selected_fpr = fpr[best_thresh_idx]
    selected_fnr = 1 - tpr[best_thresh_idx]
    roc_auc = roc_data["roc_auc"]

    cm = confusion_matrix(
        (df["label"]),
        [
            "Spoof" if score > (1 - best_threshold) else "Genuine"
            for score in df[score_column]
        ],
    )

    return f"""
False Positive Rate (0% is best): {100*selected_fpr:.2f}%
False Negative Rate (0% is best): {100*selected_fnr:.2f}%
Best threshold: {1-best_threshold:.4f}
ROC AUC (0.5 is no better than random): {roc_auc:.4f}
Confusion Matrix:
{tabulate(cm, tablefmt="plain")}
"""


def make_summary_page(out_dir, summary):
    with open(os.path.join(out_dir, f"summary.md"), "w") as f:
        markdown_content = f"""# Classification Results Summary

## Summary Statistics
```
{summary}
```

## ROC Curve
![ROC Curve](roc_curve.png)

## Score Distribution
![Score Histogram](score_hist.png)

## Confusion Matrix
![Confusion Matrix](confusion_matrix.png)
"""
        f.write(markdown_content)


def get_filenames():
    return os.listdir(os.path.join(ROOT_DIR, "replications"))


def create_data_subsets(df):

    _3_1_8b_quantizations = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "context-labs/neuralmagic-llama-3.1-8b-instruct-FP8",
    ]
    is_3_1_8b = df["original_model"].isin(_3_1_8b_quantizations) & df[
        "replication_model"
    ].isin(_3_1_8b_quantizations)

    dfs = {
        "same_machine_quantization_test": df[
            (df["inference_machine"] == df["replication_machine"]) & is_3_1_8b
        ],
        "different_machine_quantization_test": df[
            (df["inference_machine"] != df["replication_machine"]) & is_3_1_8b
        ],
        "same_machine_non_quantization_test": df[
            (df["inference_machine"] == df["replication_machine"])
        ],
        "different_machine_non_quantization_test": df[
            (df["inference_machine"] != df["replication_machine"])
        ],
        "all_data": df,
    }

    return dfs


def main():
    args = parse_args()
    sim_callback = make_sim_callback(args.sim_method)
    batch_sim_callback = make_batch_sim_callback(args.sim_method)

    df = []
    filenames = get_filenames()
    for filename in filenames:
        filepath = os.path.join(ROOT_DIR, "replications", filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        compute_scores(args, sim_callback, batch_sim_callback, data, df)
    df = pd.DataFrame(df)
    data_subsets = create_data_subsets(df)
    for name, df in data_subsets.items():
        analyze_results(args, name, df, "similarity")


if __name__ == "__main__":
    main()
