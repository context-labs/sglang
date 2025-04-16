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
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm

TOPLOC_DIR = os.path.dirname(os.path.join("../", os.path.abspath(__file__)))
OUTPUT_DIR = os.path.abspath(
    os.path.join(TOPLOC_DIR, "classifier_analysis_results", "embeddings_v1")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=False, default=None)
    return parser.parse_args()


def compute_scores(args, sim_callback, filename, df):
    filepath = os.path.join(TOPLOC_DIR, "replications", filename)
    with open(filepath, "r") as f:
        data = json.load(f)

    N = args.N if args.N is not None else len(data)
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
                "similarity": similarity[0][0],
                "genuine": orig_model == repl_model,
                "label": "Genuine" if orig_model == repl_model else "Spoof",
                "inference_machine": item["inference_machine"],
                "replication_machine": item["replication_machine"],
            }
        )


def get_np_embedding(text: str):
    embeddings = model.encode([text])
    return embeddings


def cosine_similarity_callback(model):
    def callback(orig_response, repl_response):
        orig_embedding = get_np_embedding(orig_response)  # -> [M, D]
        repl_embedding = get_np_embedding(repl_response)  # -> [M, D]

        similarity = cosine_similarity(
            orig_embedding.mean(axis=0).reshape(1, -1),
            repl_embedding.mean(axis=0).reshape(1, -1),
        )
        return similarity

    return callback


def cross_encoder_similarity_callback(model):
    def callback(orig_response, repl_response):
        similarity = model.predict([orig_response, repl_response])
        return similarity[0]

    return callback


def make_sim_callback(args):
    if args.sim_method == "cosine/all-MiniLM-L6-v2":
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return cosine_similarity_callback(model)
    elif args.sim_method == "cross-encoder/stsb-distilroberta-base":
        model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
        return cross_encoder_similarity_callback(model)


def analyze_results(name, df, score_column):

    y_true = df["genuine"]
    y_score = df[score_column]

    roc_auc = roc_auc_score(y_true, y_score)
    print(f"{name} {score_column}: ROC AUC: {roc_auc}")

    fpr, tpr, threshes = roc_curve(y_true, y_score)
    display = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{name} {score_column}"
    )
    display.plot()
    plt.savefig(os.path.join(TOPLOC_DIR, f"{name}_{score_column}_roc_curve.png"))
    plt.show()

    fig, ax = plt.subplots()
    hplot = sns.histplot(
        df, x=score_column, hue="label", ax=ax, element="step", stat="count"
    )
    if score_column == "similarity":
        title = f"Cosine Similarity Scores ({name.replace("_", " ").title()})"
    else:
        title = f"{name.replace("_", " ").title()} {score_column}"
    hplot.set_title(title)
    plt.savefig(os.path.join(TOPLOC_DIR, f"{name}_{score_column}_hist.png"))
    plt.show()

    best_thresh_idx = np.argmin(np.abs(1 - tpr - fpr))
    thresh = threshes[best_thresh_idx]

    y_pred_label = ["Genuine" if pred else "Spoof" for pred in (y_score >= thresh)]
    cm = confusion_matrix(df["label"], y_pred_label)
    fig, conf_mat_ax = plt.subplots()
    conf_mat_display = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Spoof"])
    conf_mat_display.plot(cmap="gray", colorbar=False, ax=conf_mat_ax)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_{score_column}_confusion_matrix.png"))
    plt.show()

    print(f"\n=== {name} {score_column} ===")
    print(f"Best threshold: {thresh:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"TPR: {tpr[best_thresh_idx]:.4f}")
    print(f"FPR: {fpr[best_thresh_idx]:.4f}")
    print(f"FNR: {1 - tpr[best_thresh_idx]:.4f}")


def get_filenames():
    return os.listdir(os.path.join(TOPLOC_DIR, "replications"))


def analyze_data_subsets(df):

    dfs = {
        "same_machine": df[df["inference_machine"] == df["replication_machine"]],
        "different_machine": df[df["inference_machine"] != df["replication_machine"]],
        "all_data": df,
    }

    for name, df in dfs.items():
        analyze_results(name, df, "similarity")


def main():
    args = parse_args()
    df = []
    filenames = get_filenames()
    for filename in filenames:
        compute_scores(args, filename, df)

    df = pd.DataFrame(df)
    analyze_data_subsets(df)


if __name__ == "__main__":
    main()
