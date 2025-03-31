import dataclasses
import json
import logging
from typing import List, Optional

import torch
from toploc import build_proofs_base64, verify_proofs_base64

from sglang.srt.managers.schedule_batch import global_server_args_dict

logger = logging.getLogger(__name__)


def verify_toploc_proof(
    verification_hidden_state: torch.Tensor, verification_proof: str
) -> str:
    """
    Verify a TopLoc fingerprint proof against the provided hidden state.

    Args:
        verification_hidden_state: Hidden state tensor to verify against
        verification_proof: Base64 encoded verification proof string

    Returns:
        JSON string containing the verification result
    """
    try:
        if verification_hidden_state is None:
            logger.error("Cannot verify proof: verification_hidden_state is None")
            return json.dumps(
                {"error": "verification_hidden_state is None", "verified": False}
            )

        topk = global_server_args_dict.get("toploc_verification_topk", 128)
        logger.debug(f"Verifying TopLoc proof with topk={topk}")

        results = verify_proofs_base64(
            [verification_hidden_state],
            [verification_proof],
            decode_batching_size=1,
            topk=topk,
            skip_prefill=False,
        )

        if not results or len(results) == 0:
            logger.error("No verification results returned from verify_proofs_base64")
            return json.dumps({"error": "No verification results", "verified": False})

        logger.debug(f"Verification result: {results[0]}")
        return json.dumps(dataclasses.asdict(results[0]))
    except Exception as e:
        error_msg = f"Error verifying TopLoc proof: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "verified": False})


def create_toploc_proofs(
    verification_hidden_states: Optional[torch.Tensor],
) -> Optional[List]:
    """
    Move verification_hidden_states to CPU for additional processing when they are not None.

    Args:
        verification_hidden_states: Hidden states tensor from the verification process or None

    Returns:
        The hidden states tensor moved to CPU or None if input was None
    """
    if verification_hidden_states is None:
        logger.warning(
            "Attempted to create TopLoc proofs with None verification_hidden_states"
        )
        return None

    logger.debug(
        f"Creating TopLoc proofs from tensor with shape {verification_hidden_states.shape}"
    )

    # Move to CPU . Will have size [N,hidden] - each one should represent a "last token"
    verification_hidden_states = verification_hidden_states.detach().cpu()

    topk = global_server_args_dict["toploc_verification_topk"]
    logger.debug(
        f"Using TopLoc verification topk={topk}, tensor: {verification_hidden_states.shape}"
    )

    # Will return N proofs
    try:
        proofs = build_proofs_base64(
            [verification_hidden_states],
            decode_batching_size=1,
            topk=topk,
            skip_prefill=False,
        )
        logger.debug(
            f"Successfully generated {len(proofs) if proofs else 0} TopLoc proofs"
        )

        return proofs
    except Exception as e:
        logger.error(f"Error generating TopLoc proofs: {str(e)}")
        return None
