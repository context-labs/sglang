from toploc import build_proofs_base64


def generate_toploc_proof(tensor):
    return build_proofs_base64(
        [tensor], decode_batching_size=3, topk=4, skip_prefill=False
    )
