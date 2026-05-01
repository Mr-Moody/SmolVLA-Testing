"""Qwen3-VL offline batch annotator via vLLM.

Model is NOT loaded at import time. It is loaded lazily when QwenAnnotator is
first used or when the CLI entrypoint is called.
"""
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AWQ_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"
_BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class QwenAnnotator:
    """Wraps vLLM's LLM engine for offline Qwen3-VL batch annotation.

    Args:
        model_id: HuggingFace model ID. If None, tries AWQ first, falls back to base.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
    ):
        self._model_id = model_id
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._llm = None  # lazy load

    def _load(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "vllm is required for QwenAnnotator. "
                "Install with: pip install vllm>=0.7"
            ) from e

        from vllm import LLM

        model_id = self._model_id or self._resolve_model()
        logger.info("Loading Qwen3-VL model: %s", model_id)
        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._max_model_len,
            limit_mm_per_prompt={"video": 1, "image": 8},
        )
        logger.info("Qwen3-VL loaded.")

    def _resolve_model(self) -> str:
        """Try AWQ quantized model; fall back to base if not available."""
        try:
            from huggingface_hub import model_info
            model_info(_AWQ_MODEL)
            logger.info("Using AWQ model: %s", _AWQ_MODEL)
            return _AWQ_MODEL
        except Exception:
            logger.warning(
                "AWQ model not available or huggingface_hub not installed. "
                "Falling back to base model: %s",
                _BASE_MODEL,
            )
            return _BASE_MODEL

    def annotate_episode(
        self,
        video_path: str,
        prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Run inference on a single episode video.

        Args:
            video_path: Path to the episode video file.
            prompt: Full prompt string (system + user combined or user message).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Raw text output from the model.
        """
        self._load()
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 2.0},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Build vLLM prompt using the Qwen chat template
        tokenizer = self._llm.get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            repetition_penalty=1.05,
        )

        outputs = self._llm.generate(
            [{"prompt": text, "multi_modal_data": {"video": video_inputs}}],
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text
