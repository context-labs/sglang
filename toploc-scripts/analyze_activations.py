#!/usr/bin/env python3
"""
Script to load and analyze activation files saved by SGLang's ModelRunner.
Usage: python analyze_activations.py <activation_filename>
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from toploc import build_proofs_base64, build_proofs_bytes


def load_activations(filepath: str) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """Load activation data from a saved file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Activation file not found: {filepath}")

    print(f"Loading activations from {filepath}")
    data = torch.load(filepath)
    return data


def analyze_activations(data: Dict[str, List[Dict[str, torch.Tensor]]]) -> None:
    """Analyze and print information about the activations"""
    if "activations" not in data:
        print("Warning: No 'activations' key found in the data")
        return

    activations = data["activations"]
    print(f"Number of activation records: {len(activations)}")

    for i, act_dict in enumerate(activations):
        print(f"\nRecord {i+1}:")
        for key, tensor in act_dict.items():
            print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

            # Print a small sample of the tensor for inspection
            flat_tensor = tensor.flatten()
            sample_size = min(5, len(flat_tensor))
            print(f"  - Sample values: {flat_tensor[:sample_size].tolist()}")

            # Print basic statistics
            if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                print(
                    f"  - Stats: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
                    f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
                )

            # Print data by row
            print(f"  - Data by row:")
            for i in range(tensor.shape[0]):
                row = tensor[i]
                print(
                    f" Row {i+1} - Stats: min={row.min().item():.6f}, max={row.max().item():.6f}, "
                    f"mean={row.mean().item():.6f}, std={row.std().item():.6f}"
                )

            # Print data by column
            """
            print(f"  - Data by column:")
            for i in range(tensor.shape[1]):
                col = tensor[:, i]
                print(f" Column {i+1} - Stats: min={col.min().item():.6f}, max={col.max().item():.6f}, "
                      f"mean={col.mean().item():.6f}, std={col.std().item():.6f}")
            """

            proofs = build_proofs_base64(
                [tensor], decode_batching_size=3, topk=4, skip_prefill=False
            )
            print(f"Proof: {proofs}")


def main() -> None:
    """Main function to handle command line usage"""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <activation_filename>")
        sys.exit(1)

    # Get filename from command line
    filename = sys.argv[1]

    # If just a filename is provided, look in the default location
    if not os.path.dirname(filename):
        activations_path = os.path.join(
            os.path.dirname(os.path.dirname(filename)),
            "meta-llama",
            "activations",
            filename,
        )
    else:
        activations_path = filename

    try:
        data = load_activations(activations_path)
        analyze_activations(data)
    except Exception as e:
        print(f"Error analyzing activations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
