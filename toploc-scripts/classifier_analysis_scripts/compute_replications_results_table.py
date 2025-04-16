import glob
import json
import os

import pandas as pd
from tabulate import tabulate


def extract_model_name(full_name):
    """Extract short model name from full model path."""
    if "/" in full_name:
        return full_name.split("/")[-1]
    return full_name


def parse_replication_file(filepath):
    """Parse a replication file and extract relevant data."""
    try:
        with open(filepath, "r") as f:
            replications = json.load(f)

        results = []
        print(f"Processing file: {os.path.basename(filepath)}")

        for item in replications:
            # Skip items with errors
            if "error" in item:
                continue

            # Get replication machine
            replication_machine = item["replication_machine"]

            # Get inference machine
            inference_machine = item["inference_machine"]

            # Extract model names
            replication_model = extract_model_name(item["replication_request"]["model"])
            inference_model = extract_model_name(item["original_request"]["model"])

            # Compare responses - check for exact token match
            original_response = item["original_response"]
            replication_response = item["replication_response"]

            # Extract the actual content from responses
            original_content = ""
            replication_content = ""

            # Handle different response formats
            if isinstance(original_response, dict) and "choices" in original_response:
                if len(original_response["choices"]) > 0:
                    choice = original_response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        original_content = choice["message"]["content"]
                    elif "text" in choice:
                        original_content = choice["text"]

            if (
                isinstance(replication_response, dict)
                and "choices" in replication_response
            ):
                if len(replication_response["choices"]) > 0:
                    choice = replication_response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        replication_content = choice["message"]["content"]
                    elif "text" in choice:
                        replication_content = choice["text"]

            # Check if responses match exactly
            passed = original_content == replication_content

            # Create keys that uniquely identify replication and original setups
            replication_key = f"{replication_model}_{replication_machine}"
            inference_key = f"{inference_model}_{inference_machine}"

            results.append(
                {
                    "replication_key": replication_key,
                    "inference_key": inference_key,
                    "passed": 1 if passed else 0,
                    "count": 1,
                }
            )

            print(replication_key, inference_key, "PASS" if passed else "FAIL")

        return results
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return []


def compute_replication_matrix():
    """Compute the replication success matrix."""
    # Get all replication files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    replication_dir = os.path.join(root_dir, "replications")
    replication_files = glob.glob(os.path.join(replication_dir, "*"))

    if not replication_files:
        print(f"No replication files found in {replication_dir}")
        return None, None, None

    print(f"Found {len(replication_files)} replication files")

    # Parse all replication files
    all_results = []
    for file in replication_files:
        results = parse_replication_file(file)
        all_results.extend(results)

    if not all_results:
        print("No valid replication results found")
        return None, None, None

    df = pd.DataFrame(all_results)

    # Create grouped dataframes for the matrix view
    # Compute the percentage of successful replications
    df_grouped = (
        df.pivot_table(
            index="replication_key",
            columns="inference_key",
            values="passed",
            aggfunc="mean",
        )
        * 100
    ).astype(str) + "%"

    # Count total number of tests per configuration pair
    df_count = df.pivot_table(
        index="replication_key", columns="inference_key", values="count", aggfunc="sum"
    )

    return df, df_grouped, df_count


def main():
    # Compute the replication matrix
    df_raw, df_grouped, df_count = compute_replication_matrix()

    if df_raw is not None:
        # Print raw data and success rates
        print("\nReplication Success Rates (% passed):")
        print(
            tabulate(df_grouped.replace("nan", "--"), headers="keys", tablefmt="grid")
        )

        print("\nNumber of tests per configuration pair:")
        print(tabulate(df_count, headers="keys", tablefmt="grid"))

        # Print summary statistics
        total_tests = df_raw["count"].sum()
        total_passed = df_raw["passed"].sum()
        overall_pass_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0


if __name__ == "__main__":
    main()
