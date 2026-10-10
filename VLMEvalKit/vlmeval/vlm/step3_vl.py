import os
import warnings
from typing import Optional, Tuple

from PIL import Image

from .base import BaseModel, resolve_vllm_batch_size


class Step3VLChat(BaseModel):
    """VLMEvalKit wrapper for local Step3-VL-10B weights.

    Backends
    --------
    use_vllm=True (recommended):
        Uses vLLM offline ``LLM.chat`` directly on local weights. vLLM owns the
        model-specific chat-template and multimodal preprocessing path, matching
        the same backend used by the official ``vllm serve`` recipe.

    use_vllm=False:
        Uses the official Hugging Face Transformers inference path:
        ``AutoProcessor.apply_chat_template(..., tokenize=True, ...)`` followed by
        ``AutoModelForCausalLM.generate``.

    Reasoning parsing
    -----------------
    The official Step3-VL-10B vLLM serve command uses
    ``--reasoning-parser deepseek_r1``. vLLM exposes that parser only at the
    online serving API layer, not through offline ``LLM.chat``. For offline
    inference we therefore mirror the non-streaming DeepSeek-R1 semantics:

      - text before ``</think>`` -> reasoning
      - text after ``</think>``  -> final answer/content
      - if reasoning never closes, final answer is empty

    ``generate_inner`` returns ONLY the final answer to VLMEvalKit, while the
    latest raw output/reasoning are retained in ``last_raw_output`` and
    ``last_reasoning`` for debugging/auditing.
    """

    INTERLEAVE = True
    INSTALL_REQ = False

    def __init__(
        self,
        model_path: str = "/T_A800_USER_1/Step3-VL-10B",
        use_vllm: bool = True,
        max_new_tokens: int = 16384,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        reasoning_parser: str = "deepseek_r1",
        # vLLM engine options
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_images: int = 24,
        trust_remote_code: bool = True,
        # Transformers fallback options
        device_map: str = "auto",
        **kwargs,
    ):
        super().__init__()

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Step3-VL model path does not exist: {model_path}"
            )

        self.model_path = model_path
        self.use_vllm = bool(use_vllm)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.presence_penalty = float(presence_penalty)
        self.seed = seed
        self.reasoning_parser = reasoning_parser
        self.max_images = int(max_images)
        self.trust_remote_code = bool(trust_remote_code)

        # Audit/debug state. generate_inner() still returns final answer only.
        self.last_raw_output = None
        self.last_reasoning = None
        self.last_answer = None
        self.last_finish_reason = None
        self.last_stop_reason = None
        self.last_batch_raw_outputs = []
        self.last_batch_reasoning = []
        self.last_batch_answers = []
        self.last_batch_finish_reasons = []
        self.last_batch_stop_reasons = []

        if self.reasoning_parser not in (None, "none", "deepseek_r1"):
            raise ValueError(
                "Step3VLChat currently supports reasoning_parser='deepseek_r1' "
                f"or None, got {self.reasoning_parser!r}."
            )

        if self.use_vllm:
            self._init_vllm(
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                extra_engine_kwargs=kwargs,
            )
        else:
            self._init_transformers(
                device_map=device_map,
                extra_generate_kwargs=kwargs,
            )

    # ---------------------------------------------------------------------
    # Shared helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _is_remote_image(value: str) -> bool:
        value = str(value)
        return value.startswith(("http://", "https://", "data:"))

    @staticmethod
    def _load_local_image(path: str) -> Image.Image:
        # .copy() detaches from the file handle before leaving the context.
        with Image.open(path) as img:
            return img.convert("RGB").copy()

    def _validate_message(self, message):
        n_images = sum(1 for x in message if x.get("type") == "image")
        if n_images > self.max_images:
            raise ValueError(
                f"Step3VLChat received {n_images} images, but max_images="
                f"{self.max_images}. Increase max_images / limit_mm_per_prompt."
            )

        for item in message:
            if item.get("type") not in ("text", "image"):
                raise ValueError(
                    "Step3VLChat supports VLMEvalKit text/image messages only, "
                    f"got type={item.get('type')!r}."
                )

    @staticmethod
    def _deepseek_r1_parse(raw_text: str) -> Tuple[str, str]:
        """Mirror the final-content behavior of vLLM's deepseek_r1 parser.

        Step3's assistant generation prompt starts reasoning mode before the
        generated continuation. Therefore an output may not contain a generated
        ``<think>`` token, but ``</think>`` still marks the reasoning end.
        """
        text = (raw_text or "").strip()
        start = "<think>"
        end = "</think>"

        if end in text:
            reasoning, answer = text.split(end, 1)
            if start in reasoning:
                reasoning = reasoning.split(start, 1)[1]
            return reasoning.strip(), answer.strip()

        if start in text:
            # Reasoning started but never closed: API-style content is empty.
            return text.split(start, 1)[1].strip(), ""

        # For Step3-VL the generation prompt itself enters <think> mode. If the
        # generated continuation never emits </think>, treat all generated text
        # as reasoning rather than leaking CoT into benchmark scoring.
        return text, ""

    def _parse_output(self, raw_text: str) -> str:
        if self.reasoning_parser in (None, "none"):
            reasoning, answer = "", (raw_text or "").strip()
        else:
            reasoning, answer = self._deepseek_r1_parse(raw_text)

        self.last_raw_output = (raw_text or "").strip()
        self.last_reasoning = reasoning
        self.last_answer = answer

        if self.reasoning_parser == "deepseek_r1" and not answer:
            warnings.warn(
                "Step3-VL produced no final content after reasoning parsing. "
                f"finish_reason={self.last_finish_reason!r}, "
                f"stop_reason={self.last_stop_reason!r}. This usually means "
                "the generation was truncated before </think>; VLMEvalKit will "
                "receive an empty answer instead of the CoT."
            )

        return answer

    # ---------------------------------------------------------------------
    # vLLM offline backend (recommended)
    # ---------------------------------------------------------------------
    def _init_vllm(
        self,
        tensor_parallel_size,
        gpu_memory_utilization,
        max_model_len,
        max_num_seqs,
        extra_engine_kwargs,
    ):
        from vllm import LLM, SamplingParams

        tp = int(
            tensor_parallel_size
            if tensor_parallel_size is not None
            else os.environ.get("VLMEVAL_TP_SIZE", 1)
        )
        gpu_mem = float(
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else os.environ.get("VLMEVAL_GPU_MEMORY_UTILIZATION", 0.90)
        )
        model_len = int(
            max_model_len
            if max_model_len is not None
            else os.environ.get("VLMEVAL_MAX_MODEL_LEN", 65536)
        )
        configured_num_seqs = (
            max_num_seqs
            if max_num_seqs is not None
            else os.environ.get("VLMEVAL_MAX_NUM_SEQS", 4)
        )
        num_seqs = resolve_vllm_batch_size(configured_num_seqs)
        self.vllm_batch_size = num_seqs

        engine_kwargs = dict(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            max_model_len=model_len,
            max_num_seqs=num_seqs,
            limit_mm_per_prompt={"image": self.max_images},
        )
        engine_kwargs.update(extra_engine_kwargs)

        self.model = LLM(**engine_kwargs)

        sampling_kwargs = dict(
            max_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
        )

        if self.do_sample:
            sampling_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            # vLLM uses temperature=0 for greedy decoding.
            sampling_kwargs["temperature"] = 0.0

        if self.seed is not None:
            sampling_kwargs["seed"] = int(self.seed)

        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.processor = None
        self.input_device = None

        warnings.warn(
            "Step3VLChat initialized with LOCAL vLLM offline inference: "
            f"model={self.model_path}, tp={tp}, max_model_len={model_len}, "
            f"max_new_tokens={self.max_new_tokens}, max_images={self.max_images}, "
            f"reasoning_parser={self.reasoning_parser}."
        )

    def _build_vllm_messages(self, message):
        self._validate_message(message)
        content = []

        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            else:
                value = str(item["value"])
                if self._is_remote_image(value):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": value},
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "image_pil",
                            "image_pil": self._load_local_image(value),
                        }
                    )

        return [{"role": "user", "content": content}]

    def _generate_batch_vllm(self, messages):
        if not messages:
            self.last_batch_raw_outputs = []
            self.last_batch_reasoning = []
            self.last_batch_answers = []
            self.last_batch_finish_reasons = []
            self.last_batch_stop_reasons = []
            return []

        conversations = [self._build_vllm_messages(message) for message in messages]

        # LLM.chat applies the model's chat template and multimodal processor
        # to every conversation while vLLM schedules the batch jointly.
        outputs = self.model.chat(
            conversations,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(messages):
            raise RuntimeError(
                "Step3-VL vLLM batch returned "
                f"{len(outputs)} outputs for {len(messages)} inputs."
            )

        answers = []
        batch_raw_outputs = []
        batch_reasoning = []
        batch_finish_reasons = []
        batch_stop_reasons = []
        for output in outputs:
            completion = output.outputs[0]
            self.last_finish_reason = getattr(completion, "finish_reason", None)
            self.last_stop_reason = getattr(completion, "stop_reason", None)
            answer = self._parse_output(completion.text or "")

            answers.append(answer)
            batch_raw_outputs.append(self.last_raw_output)
            batch_reasoning.append(self.last_reasoning)
            batch_finish_reasons.append(self.last_finish_reason)
            batch_stop_reasons.append(self.last_stop_reason)

        self.last_batch_raw_outputs = batch_raw_outputs
        self.last_batch_reasoning = batch_reasoning
        self.last_batch_answers = list(answers)
        self.last_batch_finish_reasons = batch_finish_reasons
        self.last_batch_stop_reasons = batch_stop_reasons
        return answers

    def _generate_vllm(self, message):
        return self._generate_batch_vllm([message])[0]

    # ---------------------------------------------------------------------
    # Official Hugging Face Transformers fallback
    # ---------------------------------------------------------------------
    def _init_transformers(self, device_map, extra_generate_kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        # Official Step3-VL-10B weight-name remapping from the model card.
        key_mapping = {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            device_map=device_map,
            torch_dtype="auto",
            key_mapping=key_mapping,
        ).eval()

        self.input_device = self._infer_input_device()

        generation_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            use_cache=True,
            repetition_penalty=self.repetition_penalty,
        )
        if self.do_sample:
            generation_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        generation_kwargs.update(extra_generate_kwargs)
        self.generation_kwargs = generation_kwargs

        warnings.warn(
            "Step3VLChat initialized with official Hugging Face inference path: "
            f"model={self.model_path}, max_new_tokens={self.max_new_tokens}, "
            f"reasoning_parser={self.reasoning_parser}."
        )

    def _infer_input_device(self):
        candidate_paths = [
            ("model", "language_model", "embed_tokens"),
            ("model", "language_model", "model", "embed_tokens"),
            ("model", "embed_tokens"),
        ]
        for path in candidate_paths:
            obj = self.model
            ok = True
            for name in path:
                if not hasattr(obj, name):
                    ok = False
                    break
                obj = getattr(obj, name)
            if ok and hasattr(obj, "weight") and obj.weight.device.type != "meta":
                return obj.weight.device

        try:
            device = self.model.device
            if device.type != "meta":
                return device
        except Exception:
            pass

        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device

        raise RuntimeError("Could not infer a real device for Step3-VL inputs.")

    def _build_hf_messages(self, message):
        self._validate_message(message)
        content = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            else:
                # This is the exact message schema used by the official model card.
                content.append({"type": "image", "url": item["value"]})
        return [{"role": "user", "content": content}]

    def _prepare_hf_inputs(self, message):
        messages = self._build_hf_messages(message)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.input_device)

    def _generate_transformers(self, message):
        inputs = self._prepare_hf_inputs(message)
        input_len = inputs["input_ids"].shape[-1]

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs,
            )

        new_tokens = output_ids[0, input_len:]
        self.last_finish_reason = None
        self.last_stop_reason = None
        return self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    # ---------------------------------------------------------------------
    # VLMEvalKit interface
    # ---------------------------------------------------------------------
    def generate_batch(self, messages, dataset=None):
        """Generate one answer per message, using native vLLM batching."""
        if not messages:
            self.last_batch_raw_outputs = []
            self.last_batch_reasoning = []
            self.last_batch_answers = []
            self.last_batch_finish_reasons = []
            self.last_batch_stop_reasons = []
            return []

        if not self.use_vllm:
            answers = []
            batch_raw_outputs = []
            batch_reasoning = []
            batch_finish_reasons = []
            batch_stop_reasons = []
            for message in messages:
                answers.append(self.generate(message, dataset=dataset))
                batch_raw_outputs.append(self.last_raw_output)
                batch_reasoning.append(self.last_reasoning)
                batch_finish_reasons.append(self.last_finish_reason)
                batch_stop_reasons.append(self.last_stop_reason)
            self.last_batch_raw_outputs = batch_raw_outputs
            self.last_batch_reasoning = batch_reasoning
            self.last_batch_answers = list(answers)
            self.last_batch_finish_reasons = batch_finish_reasons
            self.last_batch_stop_reasons = batch_stop_reasons
            return answers

        prepared = []
        for message in messages:
            assert self.check_content(message) in [
                'str', 'dict', 'liststr', 'listdict'
            ], message
            message = self.preproc_content(message)
            assert message is not None and self.check_content(message) == 'listdict'
            self._validate_message(message)
            prepared.append(message)
        return self._generate_batch_vllm(prepared)

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self._generate_vllm(message)
        return self._parse_output(self._generate_transformers(message))
