from typing import Optional

import torch
from toploc import build_proofs_base64

from sglang.srt.managers.schedule_batch import global_server_args_dict


def create_toploc_proofs(
    verification_hidden_states: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Move verification_hidden_states to CPU for additional processing when they are not None.

    Args:
        verification_hidden_states: Hidden states tensor from the verification process or None

    Returns:
        The hidden states tensor moved to CPU or None if input was None
    """

    # Move to CPU . Will have size [N,hidden] - each one should represent a "last token"
    verification_hidden_states = verification_hidden_states.detach().cpu()

    topk = global_server_args_dict["toploc_verification_topk"]

    # Will return N proofs
    return build_proofs_base64(
        verification_hidden_states,
        decode_batching_size=3,
        topk=topk,
        skip_prefill=False,
    )
