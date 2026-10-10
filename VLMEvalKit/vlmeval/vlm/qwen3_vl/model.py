from __future__ import annotations

import logging
import os
import re
import warnings

import torch

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from vlmeval.smp import get_gpu_memory, listinstr


VLLM_MAX_IMAGE_INPUT_NUM = 24
# === QWEN +-MODE EXTENSION BEGIN ===
# Three selectable modes: capped, budgeted (two-stage), and off.
THINKING_MODES = frozenset({'capped', 'budgeted', 'off'})
EARLY_STOPPING_PROMPT = (
    'Considering the limited time available, stop reasoning now and provide the final answer '
    'directly based on the reasoning above.'
)
# === QWEN THINKING-MODE EXTENSION END ===


QWEN35_MODEL_TYPES = {'qwen3_5', 'qwen3_5_moe'}


def is_moe_model(model_path: str) -> bool:
    """Detect MoE checkpoints from active-parameter suffixes such as A3B or A17B."""
    return re.search(r'-A\d+B', model_path, flags=re.IGNORECASE) is not None


def is_qwen35_model(model_type: str, model_path: str) -> bool:
    """Identify Qwen3.5 models from config first and the path as a fallback."""
    if model_type in QWEN35_MODEL_TYPES:
        return True
    normalized_path = model_path.lower().replace('_', '').replace('-', '')
    return 'qwen3.5' in model_path.lower() or 'qwen35' in normalized_path


def patch_qwen35_fa2_position_ids() -> None:
    """
    Prevent Qwen3.5 multi-dimensional position_ids from being interpreted as
    packed-sequence metadata by Transformers' generic FlashAttention-2 wrapper.

    This patch is idempotent and must run before ``from_pretrained``.
    """
    import transformers.modeling_flash_attention_utils as fa_utils

    current = getattr(fa_utils, '_is_packed_sequence', None)
    if current is None:
        raise RuntimeError(
            'Cannot apply the Qwen3.5 FlashAttention-2 patch because '
            'transformers.modeling_flash_attention_utils._is_packed_sequence '
            'does not exist in this Transformers version.'
        )

    if getattr(current, '_qwen35_3d_safe', False):
        return

    def safe_is_packed_sequence(position_ids, batch_size):
        if position_ids is not None and position_ids.dim() > 2:
            return False
        return current(position_ids, batch_size)

    safe_is_packed_sequence._qwen35_3d_safe = True
    safe_is_packed_sequence.__name__ = current.__name__
    safe_is_packed_sequence.__doc__ = current.__doc__
    fa_utils._is_packed_sequence = safe_is_packed_sequence

    logging.info(
        'Enabled Qwen3.5 FlashAttention-2 multi-dimensional position_ids patch.'
    )


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        do_sample: bool | None = None,
        enable_thinking: bool | None = None,
        attn_implementation: str = 'flash_attention_2',
        dtype: torch.dtype | str = 'auto',
        qwen35_fa2_patch: bool = True,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        # === QWEN VLLM RUNTIME-SETTINGS EXTENSION BEGIN ===
        # Keep smoke-test context and cache allocation explicit without a one-off source patch.
        raw_max_model_len = os.environ.get('VLMEVAL_MAX_MODEL_LEN')
        if raw_max_model_len is None:
            self.vllm_max_model_len = None
        else:
            try:
                self.vllm_max_model_len = int(raw_max_model_len)
            except ValueError as err:
                raise ValueError('VLMEVAL_MAX_MODEL_LEN must be a positive integer.') from err
            if self.vllm_max_model_len <= 0:
                raise ValueError('VLMEVAL_MAX_MODEL_LEN must be a positive integer.')
        try:
            self.vllm_gpu_memory_utilization = float(
                os.environ.get('VLMEVAL_GPU_MEMORY_UTILIZATION', kwargs.get('gpu_utils', 0.9))
            )
        except ValueError as err:
            raise ValueError('VLMEVAL_GPU_MEMORY_UTILIZATION must be a number in (0, 1].') from err
        if not 0 < self.vllm_gpu_memory_utilization <= 1:
            raise ValueError('VLMEVAL_GPU_MEMORY_UTILIZATION must be in (0, 1].')
        # === QWEN VLLM RUNTIME-SETTINGS EXTENSION END ===

        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        # === QWEN THINKING-MODE EXTENSION BEGIN ===
        # Make the existing generation cap configurable without editing model registrations.
        try:
            self.max_new_tokens = int(os.environ.get('VLMEVAL_MAX_NEW_TOKENS', max_new_tokens))
        except ValueError as err:
            raise ValueError('VLMEVAL_MAX_NEW_TOKENS must be a positive integer.') from err
        if self.max_new_tokens <= 0:
            raise ValueError(
                f'VLMEVAL_MAX_NEW_TOKENS must be a positive integer, got {self.max_new_tokens}.'
            )
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.thinking_mode = os.environ.get('VLMEVAL_THINKING_MODE', 'capped').strip().lower()
        if self.thinking_mode not in THINKING_MODES:
            raise ValueError(
                f'Invalid VLMEVAL_THINKING_MODE={self.thinking_mode!r}. '
                f'Expected one of: {", ".join(sorted(THINKING_MODES))}.'
            )

        def get_positive_env_int(name, default):
            try:
                value = int(os.environ.get(name, default))
            except ValueError as err:
                raise ValueError(f'{name} must be a positive integer.') from err
            if value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value}.')
            return value

        self.thinking_budget = get_positive_env_int('VLMEVAL_THINKING_BUDGET', 4096)
        self.answer_budget = get_positive_env_int('VLMEVAL_ANSWER_BUDGET', 1024)
        if self.thinking_mode == 'off':
            self.enable_thinking = False
        elif self.thinking_mode == 'budgeted':
            self.enable_thinking = True
        else:
            self.enable_thinking = enable_thinking
        self.attn_implementation = attn_implementation
        self.dtype = dtype
        self.qwen35_fa2_patch = qwen35_fa2_patch
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might be too large, resulting in an overly long input sequence.')

        if do_sample is None:
            do_sample = temperature is not None and temperature > 0
        self.do_sample = do_sample

        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        if do_sample:
            self.generate_kwargs.update(
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            if min_p is not None and min_p > 0:
                self.generate_kwargs['min_p'] = min_p
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        logging.info(
            'Qwen thinking mode=%s, max_new_tokens=%d, thinking_budget=%d, answer_budget=%d',
            self.thinking_mode,
            self.max_new_tokens,
            self.thinking_budget,
            self.answer_budget,
        )
        # === QWEN THINKING-MODE EXTENSION END ===

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        try:
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self.model_type = getattr(model_config, 'model_type', '')
        except Exception as err:
            self.model_type = ''
            logging.warning(
                'Unable to read model_type from %s; falling back to path-based Qwen3.5 detection. Error: %s',
                model_path,
                err,
            )
        self.is_qwen35 = is_qwen35_model(self.model_type, model_path)

        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.vllm_batch_size = max(1, int(kwargs.get("max_num_seqs", 8)))
            # === QWEN VLLM RUNTIME-SETTINGS EXTENSION BEGIN ===
            llm_kwargs = dict(
                model=self.model_path,
                max_num_seqs=self.vllm_batch_size,
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                # === VLLM CUSTOM-ALL-REDUCE COMPATIBILITY FIX BEGIN ===
                # TP=2 custom all-reduce can fail during vLLM CUDA-graph profiling
                # on this runtime; use the stable NCCL fallback instead.
                disable_custom_all_reduce=True,
                # === VLLM CUSTOM-ALL-REDUCE COMPATIBILITY FIX END ===
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                trust_remote_code=True,
            )
            if self.vllm_max_model_len is not None:
                llm_kwargs['max_model_len'] = self.vllm_max_model_len
            self.llm = LLM(**llm_kwargs)
            # === QWEN VLLM RUNTIME-SETTINGS EXTENSION END ===
        else:
            if (
                self.is_qwen35
                and self.qwen35_fa2_patch
                and self.attn_implementation == 'flash_attention_2'
            ):
                patch_qwen35_fa2_position_ids()

            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype=self.dtype,
                    device_map='auto',
                    attn_implementation=self.attn_implementation,
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype=self.dtype,
                    device_map='auto',
                    attn_implementation=self.attn_implementation,
                    trust_remote_code=True,
                )
            self.model.eval()
            self._log_loaded_backends()

        torch.cuda.empty_cache()

    def _log_loaded_backends(self) -> None:
        top_backend = getattr(self.model.config, '_attn_implementation', None)
        text_config = getattr(self.model.config, 'text_config', None)
        vision_config = getattr(self.model.config, 'vision_config', None)
        text_backend = (
            getattr(text_config, '_attn_implementation', None)
            if text_config is not None else None
        )
        vision_backend = (
            getattr(vision_config, '_attn_implementation', None)
            if vision_config is not None else None
        )
        logging.info(
            'Loaded %s (model_type=%s, top_backend=%s, text_backend=%s, vision_backend=%s)',
            self.model_path,
            self.model_type or 'unknown',
            top_backend,
            text_backend,
            vision_backend,
        )
        if self.verbose:
            print(
                '[Qwen3VLChat] '
                f'model_type={self.model_type or "unknown"}, '
                f'top_backend={top_backend}, '
                f'text_backend={text_backend}, '
                f'vision_backend={vision_backend}'
            )

    # === QWEN THINKING-MODE EXTENSION BEGIN ===
    # Supports the hard no-thinking switch and continuation after budgeted reasoning.
    def _render_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ) -> str:
        template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if continue_final_message:
            template_kwargs['continue_final_message'] = True
        if self.enable_thinking is not None:
            tokenizer = getattr(self.processor, 'tokenizer', None)
            if tokenizer is None:
                raise RuntimeError(
                    'Qwen enable_thinking requires processor.tokenizer, but this processor does not expose one.'
                )
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=self.enable_thinking,
                    **template_kwargs,
                )
            except TypeError as err:
                raise RuntimeError(
                    'The current Qwen chat template does not accept `enable_thinking`. '
                    'Update the tokenizer files or Transformers before running the benchmark.'
                ) from err

        return self.processor.apply_chat_template(
            messages,
            **template_kwargs,
        )
    # === QWEN THINKING-MODE EXTENSION END ===

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            text = self._render_chat_template(messages)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            text = self._render_chat_template(messages)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def _apply_post_process(self, response):
        if not self.post_process:
            return response
        resp = response.split('\\boxed{')[-1]
        counter, end = 1, None
        for i, char in enumerate(resp):
            if char == '{':
                counter += 1
            elif char == '}':
                counter -= 1
            if counter == 0:
                end = i
                break
        return resp[:end] if end is not None else resp

    # === QWEN THINKING-MODE EXTENSION BEGIN ===
    # `max_tokens` is overridable for the reasoning and answer phases.
    def _vllm_sampling_params(self, max_tokens=None):
        from vllm import SamplingParams
        sampling_kwargs = dict(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens if max_tokens is None else max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None,
        )
        if self.min_p is not None and self.min_p > 0:
            sampling_kwargs['min_p'] = self.min_p
        return SamplingParams(**sampling_kwargs)

    def _build_vllm_request_with_conversation(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            raise RuntimeError('Batched vLLM generation is not implemented for Qwen3-Omni.')
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
            raise err

        conversation = []
        if self.system_prompt is not None:
            conversation.append({'role': 'system', 'content': self.system_prompt})
        conversation.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        text = self._render_chat_template(conversation)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversation,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs
        return req, conversation

    def _build_vllm_request(self, message, dataset=None):
        req, _ = self._build_vllm_request_with_conversation(message, dataset=dataset)
        return req

    @staticmethod
    def _completed_answer(text):
        """Return a completed final answer, or ``None`` when reasoning needs continuation."""
        if '</think>' not in text:
            return None
        answer = text.split('</think>', 1)[1].strip()
        return answer or None

    def _budgeted_continuation_request(self, request, conversation, reasoning):
        """
        Continue directly from the exact first-stage vLLM prompt/output.

        Do not re-render the assistant message through apply_chat_template with
        continue_final_message=True, because Qwen3.5's thinking template may
        rewrite <think>...</think> blocks and make the final message incompatible
        with Transformers' continuation check.
        """
        first_text = reasoning

        # Reconstruct exactly what vLLM has seen so far:
        # original rendered prompt + actual first-stage generation.
        continuation_prompt = request['prompt'] + first_text

        if '</think>' not in first_text:
            # Stage 1 exhausted its reasoning budget while still inside <think>.
            # Ask the model to stop reasoning, explicitly close the thinking block,
            # then let stage 2 generate the final answer.
            continuation_prompt += (
                    '\n\n'
                    + EARLY_STOPPING_PROMPT
                    + '\n</think>\n\n'
            )
        else:
            # Thinking has already been closed, but no final answer was produced.
            # Continue directly after </think>; do NOT add another closing tag.
            continuation_prompt += '\n\n'

        continuation_request = dict(request)
        continuation_request['prompt'] = continuation_prompt
        return continuation_request

    def _log_vllm_completion_stats(self, phase, outputs):
        """Expose truncation evidence in verbose smoke-test logs without changing outputs."""
        if not self.verbose:
            return
        finish_reasons = [getattr(output.outputs[0], 'finish_reason', 'unknown') for output in outputs]
        token_counts = [len(getattr(output.outputs[0], 'token_ids', [])) for output in outputs]
        logging.info(
            'Qwen vLLM %s: requests=%d finish_reasons=%s generated_tokens=%s',
            phase,
            len(outputs),
            finish_reasons,
            token_counts,
        )
    # === QWEN THINKING-MODE EXTENSION END ===

    def _generate_batch_vllm(self, messages, dataset=None):
        if not messages:
            return []
        # === QWEN THINKING-MODE EXTENSION BEGIN ===
        # Budgeted mode follows Qwen's two-stage protocol: reason, then force a final answer.
        if self.thinking_mode == 'budgeted':
            request_pairs = [
                self._build_vllm_request_with_conversation(message, dataset=dataset)
                for message in messages
            ]
            first_outputs = self.llm.generate(
                [request for request, _ in request_pairs],
                sampling_params=self._vllm_sampling_params(self.thinking_budget),
            )
            self._log_vllm_completion_stats('budgeted-thinking', first_outputs)
            responses = [None] * len(messages)
            continuation_requests, continuation_indices = [], []
            for index, ((request, conversation), output) in enumerate(zip(request_pairs, first_outputs)):
                first_text = output.outputs[0].text
                completed_answer = self._completed_answer(first_text)
                if completed_answer is not None:
                    responses[index] = self._apply_post_process(completed_answer)
                    continue
                continuation_requests.append(
                    self._budgeted_continuation_request(request, conversation, first_text)
                )
                continuation_indices.append(index)

            if continuation_requests:
                answer_outputs = self.llm.generate(
                    continuation_requests,
                    sampling_params=self._vllm_sampling_params(self.answer_budget),
                )
                self._log_vllm_completion_stats('budgeted-answer', answer_outputs)
                for index, output in zip(continuation_indices, answer_outputs):
                    responses[index] = self._apply_post_process(output.outputs[0].text)
        else:
            reqs = [self._build_vllm_request(message, dataset=dataset) for message in messages]
            outputs = self.llm.generate(reqs, sampling_params=self._vllm_sampling_params())
            self._log_vllm_completion_stats(self.thinking_mode, outputs)
            responses = [self._apply_post_process(o.outputs[0].text) for o in outputs]
        # === QWEN THINKING-MODE EXTENSION END ===
        if self.verbose:
            print(f'\033[32m{responses}\033[0m')
        return responses

    def generate_batch(self, messages, dataset=None):
        """Generate one response per input, using native vLLM batching when enabled."""
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                return [self._generate_inner_vllm_legacy(message, dataset=dataset) for message in messages]
            return self._generate_batch_vllm(messages, dataset=dataset)
        if not messages:
            return []
        if listinstr(['omni'], self.model_path.lower()):
            raise RuntimeError('Real batch generation is not implemented for Qwen3-Omni.')
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("Please install it via 'pip install qwen-vl-utils'")
            raise err

        conversations = []
        for message in messages:
            assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], message
            message = self.preproc_content(message)
            assert message is not None and self.check_content(message) == 'listdict'
            for item in message:
                assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

            conversation = []
            if self.system_prompt is not None:
                conversation.append({'role': 'system', 'content': self.system_prompt})
            conversation.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
            conversations.append(conversation)

        texts = [self._render_chat_template(conversation) for conversation in conversations]
        images, videos, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        tokenizer = self.processor.tokenizer
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        try:
            inputs = self.processor(
                text=texts,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                padding=True,
                **(video_kwargs or {}),
            )
        finally:
            tokenizer.padding_side = original_padding_side
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        responses = [self._apply_post_process(response) for response in responses]
        if self.verbose:
            print(f'\033[32m{responses}\033[0m')
        return responses

    def _generate_inner_vllm_legacy(self, message, dataset=None):
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self._render_chat_template(messages)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        sampling_kwargs = dict(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None,
        )
        if self.min_p is not None and self.min_p > 0:
            sampling_kwargs['min_p'] = self.min_p
        sampling_params = SamplingParams(**sampling_kwargs)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner_vllm(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            return self._generate_inner_vllm_legacy(message, dataset=dataset)
        return self._generate_batch_vllm([message], dataset=dataset)[0]

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
