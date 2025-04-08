import dataclasses
import json
import logging
from typing import List, Optional

import torch
from toploc import build_proofs_base64, verify_proofs_base64

from sglang.srt.managers.schedule_batch import global_server_args_dict

logger = logging.getLogger(__name__)


def verify_toploc_fingerprint(
    verification_hidden_state: torch.Tensor, verification_fingerprint: str
) -> Optional[str]:
    """
    Verify a TopLoc fingerprint fingerprint against the provided hidden state.

    Args:
        verification_hidden_state: Hidden state tensor to verify against
        verification_fingerprint: Base64 encoded verification fingerprint string

    Returns:
        JSON string containing the verification result
    """
    try:

        topk = global_server_args_dict.get("toploc_verification_topk", 128)

        results = verify_proofs_base64(
            [verification_hidden_state],
            [verification_fingerprint],
            decode_batching_size=1,
            topk=topk,
            skip_prefill=False,
        )

        if not results or len(results) == 0:
            raise Exception(
                "No verification results returned from verify_fingerprints_base64"
            )

        validation_result = results[0]

        return json.dumps(
            {
                "exp_mismatches": validation_result.exp_mismatches,
                "mant_err_mean": validation_result.mant_err_mean,
                "mant_err_median": validation_result.mant_err_median,
            }
        )
    except Exception as e:
        error_msg = f"Error verifying TopLoc fingerprint: {str(e)}"
        logger.error(error_msg)
        return None


def create_toploc_fingerprint(
    verification_hidden_state: Optional[torch.Tensor],
) -> Optional[str]:
    """
    Move verification_hidden_state to CPU for additional processing when they are not None.

    Args:
        verification_hidden_state: Hidden state tensor from the verification process or None

    Returns:
        The hidden states tensor moved to CPU or None if input was None
    """

    # Will return N fingerprints
    try:

        if verification_hidden_state is None:
            raise Exception(
                "Attempted to create TopLoc fingerprints with None verification_hidden_state"
            )

        topk = global_server_args_dict.get("toploc_verification_topk", 128)

        fingerprint = build_proofs_base64(
            [verification_hidden_state],
            decode_batching_size=1,
            topk=topk,
            skip_prefill=False,
        )[0]

        return fingerprint
    except Exception as e:
        logger.error(f"Error generating TopLoc fingerprints: {str(e)}")
        return None
