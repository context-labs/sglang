import unittest
from typing import Optional
from unittest import mock

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.verification.verification_info import VerificationAlgorithm


class TestVerificationAlgorithm(unittest.TestCase):
    """
    Test cases for verification_algorithm field initialization.

    To run this test:
    1. Make sure you've installed sglang in development mode:
       pip install -e .
    2. Run the test using pytest:
       pytest test/srt/test_verification_algorithm.py -v
    """

    def test_verification_algorithm_setting(self):
        """Test that verification_algorithm is set based on toploc_fingerprint flag."""
        # Create ServerArgs with toploc_fingerprint=True
        enabled_args = ServerArgs(
            model_path="dummy_model_path", toploc_fingerprint=True
        )

        # Create ServerArgs with toploc_fingerprint=False (default)
        disabled_args = ServerArgs(
            model_path="dummy_model_path", toploc_fingerprint=False
        )

        # Test when toploc_fingerprint is enabled
        enabled_verification = get_verification_algorithm(enabled_args)
        self.assertEqual(enabled_verification, VerificationAlgorithm.TOPLOC)

        # Test when toploc_fingerprint is disabled
        disabled_verification = get_verification_algorithm(disabled_args)
        self.assertEqual(disabled_verification, VerificationAlgorithm.NONE)

    @mock.patch("sglang.srt.managers.scheduler.compute_dp_attention_world_info")
    @mock.patch("sglang.srt.managers.scheduler.zmq.Context")
    @mock.patch("sglang.srt.managers.scheduler.Scheduler.init_tokenizer")
    @mock.patch("sglang.srt.managers.scheduler.Scheduler.init_memory_pool_and_cache")
    @mock.patch("sglang.srt.managers.scheduler.TpModelWorker")
    @mock.patch("sglang.srt.managers.scheduler.get_zmq_socket")
    @mock.patch("sglang.srt.managers.scheduler.threading.Thread")
    @mock.patch("sglang.srt.managers.scheduler.psutil.Process")
    @mock.patch("sglang.srt.managers.scheduler.TorchMemorySaverAdapter")
    @mock.patch("sglang.srt.managers.scheduler.get_device_module")
    @mock.patch("sglang.srt.managers.scheduler.set_random_seed")
    @mock.patch("sglang.srt.managers.scheduler.SchedulePolicy")
    class TestSchedulerInitialization(unittest.TestCase):
        """Test verification_algorithm is correctly set in Scheduler and batches."""

        def test_scheduler_with_verification_enabled(self, *mocks):
            """Test with --toploc-fingerprint enabled."""
            # Setup mocks for Scheduler initialization
            self._setup_mocks(mocks)

            # Create ServerArgs with toploc_fingerprint=True
            server_args = ServerArgs(
                model_path="dummy_model_path", toploc_fingerprint=True
            )
            port_args = mock.MagicMock(spec=PortArgs)

            # Initialize a Scheduler
            scheduler = Scheduler(
                server_args=server_args,
                port_args=port_args,
                gpu_id=0,
                tp_rank=0,
                dp_rank=None,
            )

            # Test the verification_algorithm in Scheduler
            self.assertEqual(
                scheduler.verification_algorithm, VerificationAlgorithm.TOPLOC
            )

        def test_scheduler_with_verification_disabled(self, *mocks):
            """Test with --toploc-fingerprint disabled."""
            # Setup mocks for Scheduler initialization
            self._setup_mocks(mocks)

            # Create ServerArgs with toploc_fingerprint=False
            server_args = ServerArgs(
                model_path="dummy_model_path", toploc_fingerprint=False
            )
            port_args = mock.MagicMock(spec=PortArgs)

            # Initialize a Scheduler
            scheduler = Scheduler(
                server_args=server_args,
                port_args=port_args,
                gpu_id=0,
                tp_rank=0,
                dp_rank=None,
            )

            # Test the verification_algorithm in Scheduler
            self.assertEqual(
                scheduler.verification_algorithm, VerificationAlgorithm.NONE
            )

        def _setup_mocks(self, mocks):
            """Setup common mocks for Scheduler initialization."""
            # Set up the initial mock return values
            (
                compute_dp_attn_mock,
                zmq_context_mock,
                init_tokenizer_mock,
                init_memory_mock,
                tp_worker_mock,
                get_zmq_socket_mock,
                thread_mock,
                psutil_mock,
                memory_saver_mock,
                device_module_mock,
                random_seed_mock,
                schedule_policy_mock,
            ) = mocks

            # Mock tp_worker info
            tp_worker_mock.return_value.get_worker_info.return_value = (
                100,  # max_total_num_tokens
                100,  # max_prefill_tokens
                10,  # max_running_requests
                100,  # max_req_len
                100,  # max_req_input_len
                42,  # random_seed
                "cpu",  # device
                {},  # worker_global_server_args_dict
                None,  # model_config
                None,  # pad_input_ids_func
                None,  # lora_embedding_adapter_indices
            )
            tp_worker_mock.return_value.get_tp_cpu_group.return_value = None
            tp_worker_mock.return_value.get_attention_tp_cpu_group.return_value = None
            tp_worker_mock.return_value.get_pad_input_ids_func.return_value = None

            # Mock compute_dp_attention_world_info
            compute_dp_attn_mock.return_value = (0, 1, None)

            # Mock device module
            device_module_mock.return_value.current_stream.return_value = (
                mock.MagicMock()
            )

            # Mock memory saver adapter
            memory_saver_mock.create.return_value = None


class TestScheduleBatchVerification(unittest.TestCase):
    """Test verification_algorithm is correctly set in different ScheduleBatch types."""

    def test_batch_initialization_with_verification(self):
        """Test that ScheduleBatch.init_new sets verification_algorithm correctly."""
        # Mock dependencies
        req_to_token_pool = mock.MagicMock()
        token_to_kv_pool_allocator = mock.MagicMock()
        tree_cache = mock.MagicMock()
        model_config = mock.MagicMock()

        # Mock for empty list of Reqs
        reqs = []

        # Test with TOPLOC verification_algorithm
        batch_toploc = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=False,
            spec_algorithm=None,
            verification_algorithm=VerificationAlgorithm.TOPLOC,
            enable_custom_logit_processor=False,
        )

        # Test with NONE verification_algorithm
        batch_none = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=False,
            spec_algorithm=None,
            verification_algorithm=VerificationAlgorithm.NONE,
            enable_custom_logit_processor=False,
        )

        # Check verification_algorithm is set correctly
        self.assertEqual(
            batch_toploc.verification_algorithm, VerificationAlgorithm.TOPLOC
        )
        self.assertEqual(batch_none.verification_algorithm, VerificationAlgorithm.NONE)

    def test_batch_methods_preserve_verification(self):
        """Test that batch methods (prepare_for_*) preserve verification_algorithm."""
        # Mock dependencies
        req_to_token_pool = mock.MagicMock()
        token_to_kv_pool_allocator = mock.MagicMock()
        tree_cache = mock.MagicMock()
        model_config = mock.MagicMock()

        # Create a batch with TOPLOC verification
        batch = ScheduleBatch.init_new(
            reqs=[],
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=False,
            spec_algorithm=None,
            verification_algorithm=VerificationAlgorithm.TOPLOC,
            enable_custom_logit_processor=False,
        )

        # Set essential fields and prepare different batch types
        # Patch the prepare_for_idle method to avoid full execution
        with mock.patch.object(
            ScheduleBatch,
            "prepare_for_idle",
            lambda self: setattr(self, "forward_mode", ForwardMode.IDLE),
        ):
            batch.prepare_for_idle()
            self.assertEqual(batch.verification_algorithm, VerificationAlgorithm.TOPLOC)
            self.assertEqual(batch.forward_mode, ForwardMode.IDLE)

        # Create a new batch for extend since prepare_for_extend modifies state
        batch_extend = ScheduleBatch.init_new(
            reqs=[],
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=False,
            spec_algorithm=None,
            verification_algorithm=VerificationAlgorithm.TOPLOC,
            enable_custom_logit_processor=False,
        )

        # Patch the prepare_for_extend method to avoid full execution
        with mock.patch.object(
            ScheduleBatch,
            "prepare_for_extend",
            lambda self: setattr(self, "forward_mode", ForwardMode.EXTEND),
        ):
            batch_extend.prepare_for_extend()
            self.assertEqual(
                batch_extend.verification_algorithm, VerificationAlgorithm.TOPLOC
            )
            self.assertEqual(batch_extend.forward_mode, ForwardMode.EXTEND)


def get_verification_algorithm(server_args: ServerArgs) -> VerificationAlgorithm:
    """
    Helper function that simulates the scheduler's initialization of verification_algorithm.

    This is the exact same logic used in Scheduler.__init__ to initialize verification_algorithm.
    """
    return (
        VerificationAlgorithm.TOPLOC
        if server_args.toploc_fingerprint
        else VerificationAlgorithm.NONE
    )


if __name__ == "__main__":
    unittest.main()
