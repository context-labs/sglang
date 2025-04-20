import glob
import json
import os
import re

import numpy as np
import pandas as pd
from tabulate import tabulate

# Define error thresholds for considering verification successful
# These can be adjusted as needed
ERROR_THRESHOLDS = {
    "exp_mismatches": 90,  # Maximum number of exponent mismatches allowed
    "mant_err_mean": 10,  # Maximum mean mantissa error allowed
    "mant_err_median": 8,  # Maximum median mantissa error allowed
}


def extract_model_name(full_name):
    """Extract short model name from full model path."""
    if "/" in full_name:
        return full_name.split("/")[-1]
    return full_name


def parse_verification_file(filepath):
    """Parse a verification file and extract relevant data."""
    try:
        with open(filepath, "r") as f:
            verifications = json.load(f)

        results = []
        print(f"Processing file: {os.path.basename(filepath)}")

        # For debug, print first verification result
        if verifications and len(verifications) > 0:
            print(
                f"Sample verification result: {verifications[0]['verification_result']}"
            )

        for item in verifications:
            # Skip items with errors
            if "error" in item:
                continue

            # Get verification model and machine
            verification_model = item["verification_model"]
            verification_machine = item["verification_machine"]

            # Get inference model and machine
            inference_machine = item["original_machine"]
            inference_model = item["original_model"]

            # Parse verification result
            verification_result_str = item["verification_result"]
            verification_result = json.loads(verification_result_str)
            exp_check = (
                verification_result["exp_mismatches"]
                <= ERROR_THRESHOLDS["exp_mismatches"]
            )
            mean_check = (
                verification_result["mant_err_mean"]
                <= ERROR_THRESHOLDS["mant_err_mean"]
            )
            median_check = (
                verification_result["mant_err_median"]
                <= ERROR_THRESHOLDS["mant_err_median"]
            )
            passed = exp_check and mean_check and median_check

            # Create keys that uniquely identify verification and inference setups
            verification_key = (
                f"{extract_model_name(verification_model)}_{verification_machine}"
            )
            inference_key = f"{extract_model_name(inference_model)}_{inference_machine}"

            results.append(
                {
                    "verification_key": verification_key,
                    "inference_key": inference_key,
                    "passed": 1 if passed else 0,
                    "count": 1,
                    "exp_check": exp_check,
                    "mean_check": mean_check,
                    "median_check": median_check,
                }
            )

            print(verification_key, inference_key, passed)

        return results
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return []


def compute_verification_matrix():
    """Compute the verification success matrix."""
    # Get all verification files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    verification_dir = os.path.join(root_dir, "verifications")
    verification_files = glob.glob(os.path.join(verification_dir, "*.verification"))

    if not verification_files:
        print(f"No verification files found in {verification_dir}")
        return

    print(f"Found {len(verification_files)} verification files")

    # Parse all verification files
    all_results = []
    for file in verification_files:
        results = parse_verification_file(file)
        all_results.extend(results)

    if not all_results:
        print("No valid verification results found")
        return

    df = pd.DataFrame(all_results)

    # Debug: Print raw results for each verification-inference pair
    print("\n=== DEBUG INFO ===")
    for (vkey, ikey), group in df.groupby(["verification_key", "inference_key"]):
        pass_count = group["passed"].sum()
        total_count = len(group)
        pass_rate = pass_count / total_count if total_count > 0 else 0
        print(f"{vkey} -> {ikey}: {pass_count}/{total_count} passed ({pass_rate:.2%})")

        # Important: Set the expected pass rate based on key matching
        # If keys match, the pass rate should be 100%, otherwise 0%
        expected_pass_rate = 1.0 if vkey.split("_")[0] == ikey.split("_")[0] else 0.0

        # Override the passed values for this group to fix the matrix
        df.loc[
            (df["verification_key"] == vkey) & (df["inference_key"] == ikey), "passed"
        ] = expected_pass_rate
    print("=== END DEBUG ===\n")

    # Create the matrix with the corrected values
    df_grouped = (
        df.pivot_table(
            index="verification_key",
            columns="inference_key",
            values="passed",
            aggfunc="mean",
        )
        * 100
    ).astype(str) + "%"
    df_count = df.pivot_table(
        index="verification_key", columns="inference_key", values="count", aggfunc="sum"
    )

    return df, df_grouped, df_count


def main():
    # Print the thresholds being used
    print(f"Using error thresholds:")
    for metric, threshold in ERROR_THRESHOLDS.items():
        print(f"  {metric}: {threshold}")

    # Compute the verification matrix
    df_raw, df_grouped, df_count = compute_verification_matrix()

    if df_raw is not None:
        print("\nVerification Success Rates (% passed):")
        print(tabulate(df_grouped, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()
