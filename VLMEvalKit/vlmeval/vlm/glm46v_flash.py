import os
from typing import List, Dict, Any, Tuple

from PIL import Image

from .base import BaseModel, resolve_vllm_batch_size


class GLM46VFlashChat(BaseModel):
    """
    Local-weight VLMEvalKit wrapper for GLM-4.6V-Flash.

    Default backend: vLLM offline LLM.chat()
    Optional fallback: Hugging Face Transformers

    Reasoning behavior:
      - GLM-4.6V uses the GLM-4.5/4.6 style <think>...</think> protocol.
      - vLLM's online server uses --reasoning-parser glm45.
      - Offline LLM.chat() does not expose server-side reasoning_parser, so this
        wrapper performs an equivalent conservative split:
            reasoning = content before </think>
            answer    = content after </think>
      - In thinking mode, if </think> is never produced, answer is returned as
        an empty string to avoid scoring an unfinished CoT as the final answer.
    """

    INTERLEAVE = True
    INSTALL_REQ = False

    def __init__(
        self,
        model_path="/T_A800_USER_1/glm4.6V-flash",
        use_vllm=True,
        enable_thinking=True,
        max_new_tokens=16384,
        temperature=0.8,
        top_p=0.6,
        top_k=2,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=131072,
        max_num_seqs=4,
        max_images=24,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=False,
        seed=20260801,
        **kwargs,
    ):
        super().__init__()

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"GLM-4.6V-Flash model path does not exist: {model_path}"
            )

        self.model_path = model_path
        self.use_vllm = bool(use_vllm)
        self.enable_thinking = bool(enable_thinking)
        self.max_new_tokens = int(max_new_tokens)

        self.last_reasoning = ""
        self.last_answer = ""
        self.last_raw_output = ""
        self.last_finish_reason = None
        self.last_stop_reason = None
        self.last_batch_raw_outputs = []
        self.last_batch_reasoning = []
        self.last_batch_answers = []
        self.last_batch_finish_reasons = []
        self.last_batch_stop_reasons = []

        self._generation_cfg = dict(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            presence_penalty=float(presence_penalty),
            seed=int(seed),
        )

        if self.use_vllm:
            self._init_vllm(
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_images=max_images,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                enforce_eager=enforce_eager,
                **kwargs,
            )
        else:
            self._init_transformers(
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                **kwargs,
            )

    # ---------------------------------------------------------------------
    # Backend init
    # ---------------------------------------------------------------------

    def _init_vllm(
        self,
        tensor_parallel_size,
        gpu_memory_utilization,
        max_model_len,
        max_num_seqs,
        max_images,
        trust_remote_code,
        dtype,
        enforce_eager,
        **kwargs,
    ):
        from vllm import LLM, SamplingParams

        self.vllm_batch_size = resolve_vllm_batch_size(max_num_seqs)
        llm_kwargs = dict(
            model=self.model_path,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            max_num_seqs=self.vllm_batch_size,
            limit_mm_per_prompt={"image": int(max_images)},
            trust_remote_code=bool(trust_remote_code),
            dtype=dtype,
            enforce_eager=bool(enforce_eager),
        )

        # Only pass known/explicit extra engine kwargs from config if supplied.
        llm_kwargs.update(kwargs)

        self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self._generation_cfg["temperature"],
            top_p=self._generation_cfg["top_p"],
            top_k=self._generation_cfg["top_k"],
            repetition_penalty=self._generation_cfg["repetition_penalty"],
            presence_penalty=self._generation_cfg["presence_penalty"],
            seed=self._generation_cfg["seed"],
        )

    def _init_transformers(self, trust_remote_code, dtype, **kwargs):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=bool(trust_remote_code),
        )

        torch_dtype = (
            torch.bfloat16 if str(dtype).lower() in {"bf16", "bfloat16"}
            else "auto"
        )

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=bool(trust_remote_code),
        ).eval()

    # ---------------------------------------------------------------------
    # Message conversion
    # ---------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB").copy()

    def _to_openai_messages(self, message: List[Dict[str, Any]]):
        """
        Convert VLMEvalKit interleaved message to the OpenAI-style multimodal
        message format accepted by vLLM LLM.chat().
        """
        content = []

        for item in message:
            item_type = item.get("type")
            value = item.get("value")

            if item_type == "text":
                content.append({
                    "type": "text",
                    "text": str(value),
                })

            elif item_type == "image":
                if isinstance(value, Image.Image):
                    image = value.convert("RGB")
                else:
                    image = self._load_image(str(value))

                content.append({
                    "type": "image_pil",
                    "image_pil": image,
                })

            else:
                raise ValueError(
                    f"GLM46VFlashChat only supports text/image inputs, "
                    f"got type={item_type!r}"
                )

        return [{
            "role": "user",
            "content": content,
        }]

    def _to_hf_messages(self, message: List[Dict[str, Any]]):
        """
        Convert VLMEvalKit message to the official HF chat-template format.
        Local image paths are passed as image URLs/paths for AutoProcessor.
        """
        content = []

        for item in message:
            item_type = item.get("type")
            value = item.get("value")

            if item_type == "text":
                content.append({
                    "type": "text",
                    "text": str(value),
                })
            elif item_type == "image":
                content.append({
                    "type": "image",
                    "url": str(value),
                })
            else:
                raise ValueError(
                    f"Unsupported input type for GLM-4.6V-Flash: {item_type!r}"
                )

        return [{
            "role": "user",
            "content": content,
        }]

    # ---------------------------------------------------------------------
    # glm45-equivalent reasoning split
    # ---------------------------------------------------------------------

    @staticmethod
    def _clean_special_tokens(text: str) -> str:
        if not text:
            return ""

        special_tokens = (
            # Common GLM control tokens that may appear when decoding HF output
            # with skip_special_tokens=False.
            "<|endoftext|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<|observation|>",

            # GLM-4.6V final-answer box tokens.
            "<|begin_of_box|>",
            "<|end_of_box|>",
        )

        for tok in special_tokens:
            text = text.replace(tok, "")

        return text.strip()

    def _split_glm45_reasoning(
        self,
        raw_text: str,
        finish_reason=None,
    ) -> Tuple[str, str]:
        """
        Conservative offline equivalent of vLLM --reasoning-parser glm45.

        GLM-4.5/4.6 thinking mode may emit reasoning without a leading <think>
        token, but closes it with </think>. Therefore the closing tag is the
        authoritative boundary.

        If thinking is enabled and the boundary never appears, we consider the
        generation unfinished/malformed and return no final answer.
        """
        text = self._clean_special_tokens(raw_text)

        if not self.enable_thinking:
            # Non-thinking template inserts <think></think> before the answer.
            if "</think>" in text:
                answer = text.split("</think>", 1)[1].strip()
            else:
                answer = text
            return "", self._clean_special_tokens(answer)

        if "</think>" not in text:
            reasoning = text
            return reasoning, ""

        reasoning, answer = text.split("</think>", 1)

        reasoning = reasoning.strip()
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>"):].lstrip()

        # Keep final-answer cleanup here so both vLLM and Transformers backends
        # return the same benchmark-safe content.
        answer = self._clean_special_tokens(answer)
        return reasoning, answer

    # ---------------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------------

    def _generate_batch_vllm(self, messages):
        if not messages:
            self.last_batch_raw_outputs = []
            self.last_batch_reasoning = []
            self.last_batch_answers = []
            self.last_batch_finish_reasons = []
            self.last_batch_stop_reasons = []
            return []

        conversations = [self._to_openai_messages(message) for message in messages]
        outputs = self.llm.chat(
            conversations,
            sampling_params=self.sampling_params,
            use_tqdm=False,
            chat_template_content_format="openai",
            chat_template_kwargs={
                "enable_thinking": self.enable_thinking,
            },
        )
        if len(outputs) != len(messages):
            raise RuntimeError(
                "GLM-4.6V-Flash vLLM batch returned "
                f"{len(outputs)} outputs for {len(messages)} inputs."
            )

        answers = []
        batch_raw_outputs = []
        batch_reasoning = []
        batch_finish_reasons = []
        batch_stop_reasons = []
        for output in outputs:
            candidate = output.outputs[0]
            raw = candidate.text or ""
            finish_reason = getattr(candidate, "finish_reason", None)
            stop_reason = getattr(candidate, "stop_reason", None)
            reasoning, answer = self._split_glm45_reasoning(
                raw,
                finish_reason=finish_reason,
            )

            answers.append(answer)
            batch_raw_outputs.append(raw)
            batch_reasoning.append(reasoning)
            batch_finish_reasons.append(finish_reason)
            batch_stop_reasons.append(stop_reason)

        self.last_batch_raw_outputs = batch_raw_outputs
        self.last_batch_reasoning = batch_reasoning
        self.last_batch_answers = list(answers)
        self.last_batch_finish_reasons = batch_finish_reasons
        self.last_batch_stop_reasons = batch_stop_reasons
        self.last_raw_output = batch_raw_outputs[-1]
        self.last_reasoning = batch_reasoning[-1]
        self.last_answer = answers[-1]
        self.last_finish_reason = batch_finish_reasons[-1]
        self.last_stop_reason = batch_stop_reasons[-1]
        return answers

    def _generate_vllm(self, message):
        return self._generate_batch_vllm([message])[0]

    def _generate_transformers(self, message):
        import torch

        messages = self._to_hf_messages(message)

        # This follows the official GLM-4.6V-Flash HF inference path:
        # AutoProcessor.apply_chat_template(... tokenize=True ...)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)

        # Official model-card examples remove token_type_ids.
        inputs.pop("token_type_ids", None)

        generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self._generation_cfg["temperature"],
            top_p=self._generation_cfg["top_p"],
            top_k=self._generation_cfg["top_k"],
            repetition_penalty=self._generation_cfg["repetition_penalty"],
            use_cache=True,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        input_len = inputs["input_ids"].shape[-1]
        new_ids = generated_ids[0, input_len:]

        raw = self.processor.decode(
            new_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        reasoning, answer = self._split_glm45_reasoning(raw)

        self.last_raw_output = raw
        self.last_reasoning = reasoning
        self.last_answer = answer
        self.last_finish_reason = None
        self.last_stop_reason = None

        return answer

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
            for item in message:
                if item['type'] not in ('text', 'image'):
                    raise ValueError(
                        "GLM46VFlashChat only supports text/image inputs, "
                        f"got type={item['type']!r}"
                    )
            prepared.append(message)
        return self._generate_batch_vllm(prepared)

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self._generate_vllm(message)
        return self._generate_transformers(message)
