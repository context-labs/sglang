#!/usr/bin/env python3
"""
Compare TopLoc fingerprint verification results from original and spoofed models.
This script processes the JSON files produced by test_ultrachat.py and test_spoof_ultrachat.py
and creates a consolidated CSV file with comparison statistics.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json_data(json_path):
    """Load the JSON data from the given path."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_data(original_json, spoof_json, output_csv):
    """
    Process both JSON files and create a consolidated CSV.

    Args:
        original_json: Path to the original verification results JSON
        spoof_json: Path to the spoofed verification results JSON
        output_csv: Path to the output CSV file
    """
    print(f"Loading data from {original_json}...")
    original_data = load_json_data(original_json)

    print(f"Loading data from {spoof_json}...")
    spoof_data = load_json_data(spoof_json)

    # Convert the lists to pandas DataFrames
    df_original = pd.DataFrame(original_data)
    df_spoof = pd.DataFrame(spoof_data)

    # Set MultiIndex for joining
    df_original.set_index(["ultrachat_filepath", "ultrachat_id"], inplace=True)
    df_spoof.set_index(["ultrachat_filepath", "ultrachat_id"], inplace=True)

    # Rename columns to differentiate between original and spoofed data
    df_original = df_original.rename(
        columns={
            "exp_mismatches": "original_exp_mismatches",
            "mant_err_mean": "original_mant_err_mean",
            "mant_err_median": "original_mant_err_median",
            "verified": "original_verified",
        }
    )

    df_spoof = df_spoof.rename(
        columns={
            "exp_mismatches": "spoof_exp_mismatches",
            "mant_err_mean": "spoof_mant_err_mean",
            "mant_err_median": "spoof_mant_err_median",
            "verified": "spoof_verified",
        }
    )

    # Join the DataFrames on their indices
    print("Joining datasets...")
    df_combined = df_original.join(
        df_spoof, how="outer", lsuffix="_orig", rsuffix="_spoof"
    )

    # Reset index to make ultrachat_filepath and ultrachat_id regular columns again
    df_combined.reset_index(inplace=True)

    # Select only the relevant columns for the final CSV
    columns_to_keep = [
        "ultrachat_filepath",
        "ultrachat_id",
        "original_exp_mismatches",
        "original_mant_err_mean",
        "original_mant_err_median",
        "original_verified",
        "spoof_exp_mismatches",
        "spoof_mant_err_mean",
        "spoof_mant_err_median",
        "spoof_verified",
    ]

    df_final = df_combined[columns_to_keep]

    # Add a column indicating if the verification result changed
    df_final["verification_changed"] = (
        df_final["original_verified"] != df_final["spoof_verified"]
    )

    # Calculate verification success rates
    total_samples = len(df_final)
    original_verified_count = df_final["original_verified"].sum()
    spoof_verified_count = df_final["spoof_verified"].sum()
    verification_changed_count = df_final["verification_changed"].sum()

    print(f"\nResults Summary:")
    print(f"Total samples: {total_samples}")
    print(
        f"Original model verified: {original_verified_count} ({original_verified_count/total_samples*100:.2f}%)"
    )
    print(
        f"Spoofed model verified: {spoof_verified_count} ({spoof_verified_count/total_samples*100:.2f}%)"
    )
    print(
        f"Verification result changed: {verification_changed_count} ({verification_changed_count/total_samples*100:.2f}%)"
    )

    # Save to CSV
    print(f"\nSaving results to {output_csv}...")
    df_final.to_csv(output_csv, index=False)
    print(f"Results saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TopLoc fingerprint verification results"
    )
    parser.add_argument(
        "--original-json",
        type=str,
        default="ultrachat_verification_results.json",
        help="JSON file with original verification results",
    )
    parser.add_argument(
        "--spoof-json",
        type=str,
        default="ultrachat_spoof_verification_results.json",
        help="JSON file with spoof verification results",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="verification_comparison.csv",
        help="Output CSV file with consolidated results",
    )
    args = parser.parse_args()

    # Get the current directory
    script_dir = Path(__file__).parent.absolute()

    # Process paths
    original_json = script_dir / args.original_json
    spoof_json = script_dir / args.spoof_json
    output_csv = script_dir / args.output_csv

    # Make sure both input files exist
    if not original_json.exists():
        print(f"Error: Original JSON file {original_json} not found.")
        return

    if not spoof_json.exists():
        print(f"Error: Spoof JSON file {spoof_json} not found.")
        return

    # Process the data
    process_data(original_json, spoof_json, output_csv)


if __name__ == "__main__":
    main()
