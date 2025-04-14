import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=False, default=None)
    return parser.parse_args()


def measure_cosine_similarities(args, filename, df):
    filepath = os.path.join(SCRIPT_DIR, "replications", filename)
    with open(filepath, "r") as f:
        data = json.load(f)

    N = args.N if args.N is not None else len(data)
    for item in tqdm(data[:N]):

        orig_model = item["original_request"]["model"]
        repl_model = item["replication_request"]["model"]

        orig_response = item["original_response"]["choices"][0]["message"]["content"]
        orig_embedding = get_np_embedding(orig_response)  # -> [M, D]

        repl_response = item["replication_response"]["choices"][0]["message"]["content"]
        repl_embedding = get_np_embedding(repl_response)  # -> [M, D]

        similarity = cosine_similarity(
            orig_embedding.mean(axis=0).reshape(1, -1),
            repl_embedding.mean(axis=0).reshape(1, -1),
        )
        print(f"Similarity: {similarity}")
        df.append(
            {
                "prompt": item["prompt"],
                "original_response": orig_response,
                "replication_response": repl_response,
                "similarity": similarity,
                "label": orig_model == repl_model,
            }
        )


def get_np_embedding(text: str):
    embeddings = model.encode([text])
    return embeddings


def analyze_results(df):
    df = pd.DataFrame(df)

    # Calculate metrics for different thresholds
    thresholds = np.linspace(0, 1, 100)
    results = []
    beta = 2  # F-beta score: beta > 1 gives more weight to recall

    for threshold in thresholds:
        predictions = df["similarity"] >= threshold
        true_positives = sum(predictions & df["label"])
        false_positives = sum(predictions & ~df["label"])
        false_negatives = sum(~predictions & df["label"])

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f_beta = (
            (1 + beta**2)
            * (precision * recall)
            / max((beta**2 * precision + recall), 1e-10)
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f_beta": f_beta,
            }
        )

    results_df = pd.DataFrame(results)
    best_idx = results_df["f_beta"].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]

    # Plot ROC curve and metrics
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(df["label"], df["similarity"])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    # ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    # F-beta and threshold
    plt.subplot(1, 2, 2)
    plt.plot(results_df["threshold"], results_df["precision"], label="Precision")
    plt.plot(results_df["threshold"], results_df["recall"], label="Recall")
    plt.plot(results_df["threshold"], results_df["f_beta"], label=f"F-{beta} Score")
    plt.axvline(
        x=best_threshold,
        color="r",
        linestyle="--",
        label=f"Best Threshold = {best_threshold:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Metrics vs Threshold (Best F-{beta} Score)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "embedding_similarity_analysis.png"))
    plt.show()

    print(
        f"Best threshold: {best_threshold:.4f} (F-{beta} Score: {results_df.loc[best_idx, 'f_beta']:.4f})"
    )


def get_filenames():
    return os.listdir(os.path.join(SCRIPT_DIR, "replications"))


def main():
    args = parse_args()
    df = []
    filenames = get_filenames()
    for filename in filenames:
        measure_cosine_similarities(args, filename, df)
    analyze_results(df)


if __name__ == "__main__":
    main()
