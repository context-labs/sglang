import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
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


def compute_scores(N, sim_callback, data, df):
    N = N or len(data)
    for item in tqdm(data[:N]):

        orig_model = item["original_request"]["model"]
        repl_model = item["replication_request"]["model"]

        orig_response = item["original_response"]["choices"][0]["message"]["content"]

        repl_response = item["replication_response"]["choices"][0]["message"]["content"]

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

    df = []
    filenames = get_filenames()
    for filename in filenames:
        filepath = os.path.join(ROOT_DIR, "replications", filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        compute_scores(args, sim_callback, data, df)
    df = pd.DataFrame(df)
    data_subsets = create_data_subsets(df)
    for name, df in data_subsets.items():
        analyze_results(args, name, df, "similarity")


if __name__ == "__main__":
    main()
