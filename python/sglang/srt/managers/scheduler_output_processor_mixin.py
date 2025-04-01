from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import BatchEmbeddingOut, BatchTokenIDOut
from sglang.srt.managers.schedule_batch import BaseFinishReason, Req, ScheduleBatch
from sglang.srt.verification.verification_utils import (
    create_toploc_proof,
    verify_toploc_proof,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
    )

logger = logging.getLogger(__name__)


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None

        if self.is_generation:
            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
                bid,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
                result.bid,
            )

            if self.enable_overlap:
                logits_output, next_token_ids = self.tp_worker.resolve_batch_result(bid)
            else:
                # Move next_token_ids and logprobs to cpu
                next_token_ids = next_token_ids.tolist()
                if batch.return_logprob:
                    if logits_output.next_token_logprobs is not None:
                        logits_output.next_token_logprobs = (
                            logits_output.next_token_logprobs.tolist()
                        )
                    if logits_output.input_token_logprobs is not None:
                        logits_output.input_token_logprobs = tuple(
                            logits_output.input_token_logprobs.tolist()
                        )

            hidden_state_offset = 0
            verification_hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    continue

                if self.is_mixed_chunk and self.enable_overlap and req.finished():
                    # Free the one delayed token for the mixed decode batch
                    j = len(batch.out_cache_loc) - len(batch.reqs) + i
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j : j + 1])
                    continue

                if req.is_chunked <= 0:
                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    if req.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if logits_output.verification_hidden_states is not None:
                        # each item in verification_hidden_states can contain a mixture of sequences for prefill
                        # this fetches the last token in the sequence
                        logger.debug(
                            f"prefill {logits_output.verification_hidden_states.shape}, {logits_output.hidden_states.shape if logits_output.hidden_states is not None else None}, {len(req.origin_input_ids)}"
                        )
                        verification_hidden_state = (
                            logits_output.verification_hidden_states[i].cpu().clone()
                        )
                        verification_hidden_state_offset += len(req.origin_input_ids)
                        logger.debug(
                            f"prefill Verification hidden state in decode step: {verification_hidden_state.shape}"
                        )
                        req.verification_hidden_states.append(verification_hidden_state)
                        req.verification_proofs.append(
                            create_toploc_proof(verification_hidden_state)
                        )
                        logger.debug(
                            f"prefill Added verification proof #{len(req.verification_proofs)} to req (decode)"
                        )
                        if req.verification_proof_to_validate is not None:
                            req.verification_proof_validation_result = (
                                verify_toploc_proof(
                                    verification_hidden_state,
                                    req.verification_proof_to_validate,
                                )
                            )
                            logger.debug(
                                f"prefill Verified proof #{len(req.verification_proofs)} in req (decode)"
                            )
                    else:
                        logger.debug("No verification hidden states to process")
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if req.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = (
                                extend_input_len - extend_logprob_start_len
                            )
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                            logprob_pt += num_input_logprobs

            if batch.next_batch_sampling_info:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
                batch.next_batch_sampling_info.sampling_info_done.set()

        else:  # embedding or reward model
            embeddings, bid = result.embeddings, result.bid
            embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        logits_output, next_token_ids, bid = (
            result.logits_output,
            result.next_token_ids,
            result.bid,
        )
        self.num_generated_tokens += len(batch.reqs)

        if self.enable_overlap:
            logits_output, next_token_ids = self.tp_worker.resolve_batch_result(bid)
            next_token_logprobs = logits_output.next_token_logprobs
        elif batch.spec_algorithm.is_none():
            # spec decoding handles output logprobs inside verify process.
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()

        self.token_to_kv_pool_allocator.free_group_begin()

        # Check finish condition
        # NOTE: the length of reqs and next_token_ids don't match if it is spec decoding.
        # We should ignore using next_token_ids for spec decoding cases.
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished():
                # Free the one extra delayed token
                if self.page_size == 1:
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[i : i + 1])
                else:
                    # Only free when the extra token is in a new page
                    if (
                        len(req.origin_input_ids) + len(req.output_ids) - 1
                    ) % self.page_size == 0:
                        self.token_to_kv_pool_allocator.free(
                            batch.out_cache_loc[i : i + 1]
                        )
                continue

            if batch.spec_algorithm.is_none():
                # speculative worker will solve the output_ids in speculative decoding
                req.output_ids.append(next_token_id)

            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if logits_output.verification_hidden_states is not None:
                # each item in verification_hidden_states is [N,D] when N is number of tokens
                if req.finished():
                    verification_hidden_state = (
                        logits_output.verification_hidden_states[i].cpu().clone()
                    )
                    logger.debug(
                        f"Verification hidden state in prefill step: {verification_hidden_state.shape}"
                    )
                    req.verification_hidden_states.append(verification_hidden_state)
                    req.verification_proofs.append(
                        create_toploc_proof(verification_hidden_state)
                    )
                    logger.debug(
                        f"Added verification proof #{len(req.verification_proofs)} to req (prefill)"
                    )
                    if req.verification_proof_to_validate is not None:
                        req.verification_proof_validation_result = verify_toploc_proof(
                            verification_hidden_state[-1, ...],  # last token
                            req.verification_proof_to_validate,
                        )
                        logger.debug(
                            f"Verified proof #{len(req.verification_proofs)} in req (prefill)"
                        )
                else:
                    # No need to generate a proof until the last decode step of the sequence
                    req.verification_hidden_states.append(None)
                    req.verification_proofs.append(None)
            else:
                logger.debug("No verification hidden states to process")

            if req.grammar is not None and batch.spec_algorithm.is_none():
                req.grammar.accept_token(next_token_id)
                req.grammar.finished = req.finished()

        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

        self.stream_output(batch.reqs, batch.return_logprob)

        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.attn_tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats()

    def add_input_logprob_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Compute input_token_logprobs_val
            # Always pad the first one with None.
            req.input_token_logprobs_val = [None]
            req.input_token_logprobs_val.extend(input_token_logprobs)
            # The last input logprob is for sampling, so just pop it out.
            req.input_token_logprobs_val.pop()

            # Compute input_token_logprobs_idx
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]
            # Clip the padded hash values from image tokens.
            # Otherwise, it will lead to detokenization errors.
            input_token_logprobs_idx = [
                x if x < self.model_config.vocab_size - 1 else 0
                for x in input_token_logprobs_idx
            ]
            req.input_token_logprobs_idx = input_token_logprobs_idx

            if req.top_logprobs_num > 0:
                req.input_top_logprobs_val = [None]
                req.input_top_logprobs_idx = [None]
                assert len(req.temp_input_token_ids_logprobs_val) == len(
                    req.temp_input_token_ids_logprobs_idx
                )
                for val, idx in zip(
                    req.temp_input_top_logprobs_val,
                    req.temp_input_top_logprobs_idx,
                    strict=True,
                ):
                    req.input_top_logprobs_val.extend(val)
                    req.input_top_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_top_logprobs_val.pop()
                req.input_top_logprobs_idx.pop()
                req.temp_input_top_logprobs_idx = None
                req.temp_input_top_logprobs_val = None

            if req.token_ids_logprob is not None:
                req.input_token_ids_logprobs_val = [None]
                req.input_token_ids_logprobs_idx = [None]

                for val, idx in zip(
                    req.temp_input_token_ids_logprobs_val,
                    req.temp_input_token_ids_logprobs_idx,
                    strict=True,
                ):
                    req.input_token_ids_logprobs_val.extend(val)
                    req.input_token_ids_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_token_ids_logprobs_val.pop()
                req.input_token_ids_logprobs_idx.pop()
                req.temp_input_token_ids_logprobs_idx = None
                req.temp_input_token_ids_logprobs_val = None

            if req.return_logprob:
                relevant_tokens_len = len(req.origin_input_ids) - req.logprob_start_len
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs_val.append(output.next_token_logprobs[i])
        req.output_token_logprobs_idx.append(next_token_ids[i])

        self.add_input_logprob_return_values(
            i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
        )

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.output_token_ids_logprobs_val.append(
                output.next_token_token_ids_logprobs_val[i]
            )
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def stream_output(
        self, reqs: List[Req], return_logprob: bool, skip_req: Optional[Req] = None
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

    def stream_output_generation(
        self, reqs: List[Req], return_logprob: bool, skip_req: Optional[Req] = None
    ):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        output_hidden_states = None
        verification_proofs = None
        verification_proof_validation_results = None
        origin_input_ids = []
        output_token_ids = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_abort:
                continue

            if (
                req.finished()
                # If stream, follow the given stream_interval
                or (req.stream and len(req.output_ids) % self.stream_interval == 0)
                # If not stream, we still want to output some tokens to get the benefit of incremental decoding.
                # TODO(lianmin): this is wrong for speculative decoding because len(req.output_ids) does not
                # always increase one-by-one.
                or (
                    not req.stream
                    and len(req.output_ids) % 50 == 0
                    and not self.model_config.is_multimodal_gen
                )
            ):
                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()
                decode_ids_list.append(decode_ids)
                read_offsets.append(read_offset)
                if self.skip_tokenizer_init:
                    output_ids.append(req.output_ids)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(req.output_ids))
                cached_tokens.append(req.cached_tokens)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)

                if return_logprob:
                    input_token_logprobs_val.append(req.input_token_logprobs_val)
                    input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                    output_token_logprobs_val.append(req.output_token_logprobs_val)
                    output_token_logprobs_idx.append(req.output_token_logprobs_idx)
                    input_top_logprobs_val.append(req.input_top_logprobs_val)
                    input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                    output_top_logprobs_val.append(req.output_top_logprobs_val)
                    output_top_logprobs_idx.append(req.output_top_logprobs_idx)
                    input_token_ids_logprobs_val.append(
                        req.input_token_ids_logprobs_val
                    )
                    input_token_ids_logprobs_idx.append(
                        req.input_token_ids_logprobs_idx
                    )
                    output_token_ids_logprobs_val.append(
                        req.output_token_ids_logprobs_val
                    )
                    output_token_ids_logprobs_idx.append(
                        req.output_token_ids_logprobs_idx
                    )

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)

                if verification_proofs is None:
                    verification_proofs = []
                if (
                    hasattr(req, "verification_proofs")
                    and req.verification_proofs is not None
                ):
                    verification_proofs.append(req.verification_proofs)

                # Collect verification proof validation results
                if verification_proof_validation_results is None:
                    verification_proof_validation_results = []
                verification_proof_validation_results.append(
                    req.verification_proof_validation_result
                )

                if (
                    hasattr(req, "origin_input_ids")
                    and req.origin_input_ids is not None
                ):
                    origin_input_ids.append(list(req.origin_input_ids))
                else:
                    origin_input_ids.append([])

                if hasattr(req, "output_ids") and req.output_ids is not None:
                    output_token_ids.append(list(req.output_ids))
                else:
                    output_token_ids.append([])

        logger.debug(
            f" just prior to batch_out creation origin_input_ids: {origin_input_ids}"
        )
        logger.debug(
            f" just prior to batch_out creation output_token_ids: {output_token_ids}"
        )

        # Send to detokenizer
        if rids:
            if self.model_config.is_multimodal_gen:
                return
            # Create the BatchTokenIDOut object and log its contents
            batch_out = BatchTokenIDOut(
                rids,
                finished_reasons,
                decoded_texts,
                decode_ids_list,
                read_offsets,
                output_ids,
                skip_special_tokens,
                spaces_between_special_tokens,
                no_stop_trim,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                spec_verify_ct,
                input_token_logprobs_val,
                input_token_logprobs_idx,
                output_token_logprobs_val,
                output_token_logprobs_idx,
                input_top_logprobs_val,
                input_top_logprobs_idx,
                output_top_logprobs_val,
                output_top_logprobs_idx,
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
                output_token_ids_logprobs_val,
                output_token_ids_logprobs_idx,
                origin_input_ids,
                output_token_ids,
                output_hidden_states,
                verification_proofs,
                verification_proof_validation_results,
            )

            # Log what verification_proofs looks like in the batch object
            logger.debug(
                f"BatchTokenIDOut verification_proofs before send ({rids}): {getattr(batch_out, 'verification_proofs', None)}"
            )
            logger.debug(
                f"BatchTokenIDOut verification_proof_validation_results before send ({rids}): {getattr(batch_out, 'verification_proof_validation_results', None)}"
            )

            # Send to detokenizer
            self.send_to_detokenizer.send_pyobj(batch_out)

    def stream_output_embedding(self, reqs: List[Req]):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)
        self.send_to_detokenizer.send_pyobj(
            BatchEmbeddingOut(
                rids, finished_reasons, embeddings, prompt_tokens, cached_tokens
            )
        )
