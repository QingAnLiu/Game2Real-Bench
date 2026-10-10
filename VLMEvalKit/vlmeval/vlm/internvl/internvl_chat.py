import math
import pandas as pd
import random
import re
import yaml
import string
import torch
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import warnings
from PIL import Image
from functools import partial
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor

from .utils import (build_multi_choice_prompt,
                    build_mm_multi_choice_prompt,
                    build_video_prompt,
                    build_mpo_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt,
                    mpo_post_processing,
                    format_nav_prompt,
                    pile_action_history,
                    reorganize_prompt,
                    load_image)
from .utils import mpo_prompt_with_final_answer, mpo_prompt_without_final_answer, parse_bbox_internvl

from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY, build_dataset, infer_dataset_basename
from ...smp import *

# load all the gui templates
upper_path = Path(__file__).parent
with open(os.path.join(upper_path, "gui_template.yaml"), "r") as f:
    GUI_TEMPLATE = yaml.load(f, Loader=yaml.FullLoader)

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()


def prepare_messages_list(prompt, image_path, system_prompt=None):
    from lmdeploy.vl.constants import IMAGE_TOKEN
    content = [{'type': 'text', 'text': prompt.replace('<image>', IMAGE_TOKEN)}]

    if isinstance(image_path, str):
        image_path = [image_path]

    for image in image_path:
        img = Image.open(image).convert('RGB')
        b64 = encode_image_to_base64(img)
        img_struct = dict(url=f'data:image/jpeg;base64,{b64}')
        content.append(dict(type='image_url', image_url=img_struct))

    messages = []

    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})

    messages.append({
        'role': 'user',
        'content': content,
    })

    return [messages]


def extract_boxed_content(ans: str):
    idx = ans.rfind(r'\boxed{')
    if idx == -1:
        return ans

    idx += len(r'\boxed{')
    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == '{':
            brace_level += 1
        elif ans[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                break
        i += 1

    if brace_level != 0:
        # Unbalanced braces
        return ans

    content = ans[content_start:i]
    return content


class InternVLChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_path='OpenGVLab/InternVL-Chat-V1-5',
                 load_in_8bit=False,
                 use_mpo_prompt=False,
                 version='V1.0',
                 screen_parse=True,
                 # Best-of-N parameters
                 best_of_n=1,
                 reward_model_path=None,
                 # R1 parameters
                 cot_prompt_version='v1',
                 #
                 use_lmdeploy=False,
                 use_vllm=False,
                 use_postprocess=False,
                 **kwargs):

        assert best_of_n >= 1
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.37.2', 'ge')

        self.use_lmdeploy = use_lmdeploy
        self.use_vllm = use_vllm
        assert int(self.use_lmdeploy) + int(self.use_vllm) <= 1, \
            'Only one of use_lmdeploy/use_vllm can be enabled.'
        self.cot_prompt_version = cot_prompt_version
        self.use_mpo_prompt = use_mpo_prompt
        self.use_cot = (os.getenv('USE_COT') == '1')
        self.use_postprocess = use_postprocess

        if cot_prompt_version == 'r1':
            self.system_prompt = R1_SYSTEM_PROMPT
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
        elif cot_prompt_version == 'v2':
            self.system_prompt = None
            self.cot_prompt = "Answer the preceding multiple-choice question \
            by carefully analyzing the provided image. \nPlease answer with \
            carefully thought step by step. Apply the thinking process \
            recursively at both macro and micro levels. \nVerify consistency \
            of reasoning and look for potential flaws or gaps during \
            thinking. \nWhen realize mistakes, explain why the previous \
            thinking was incorrect, fix it and then continue thinking.\nThe \
            last line of your response should follow this format: 'Answer: \
            \\boxed{$LETTER}' (without quotes), where LETTER is one of the \
            options\n\n"
        else:
            assert cot_prompt_version == 'v1'
            self.system_prompt = None
            self.cot_prompt = None

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        self.screen_parse = screen_parse

        if use_vllm:
            assert best_of_n == 1, 'vLLM backend currently supports best_of_n=1 for InternVLChat.'
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = max(1, gpu_count)
            self.vllm_batch_size = max(1, int(kwargs.pop('max_num_seqs', 4)))
            gpu_utils = float(kwargs.pop('gpu_utils', 0.9))
            max_model_len = int(kwargs.pop('max_model_len', 32768))
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_num_seqs=self.vllm_batch_size,
                max_model_len=max_model_len,
                limit_mm_per_prompt={'image': 24},
                gpu_memory_utilization=gpu_utils,
                seed=0,
            )
            self.device = 'cuda'
        elif use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, PytorchEngineConfig, VisionConfig, pipeline
            engine_type = PytorchEngineConfig if "internvl3_5" in model_path.lower() else TurbomindEngineConfig
            vision_config = VisionConfig(max_batch_size=4)
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                vision_config=vision_config,
                backend_config=engine_type(
                    session_len=max(16384, kwargs.get("max_new_tokens", 16384)),
                    cache_max_entry_count=0.5,
                    tp=num_gpus,
                )
            )
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto").eval()
            self.device = 'cuda'

        if best_of_n > 1:
            assert version == 'V2.0', 'only support BoN evaluation with version==V2.0'
            assert reward_model_path is not None

            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                reward_model_path, trust_remote_code=True, use_fast=False)
            self.reward_model = AutoModel.from_pretrained(
                reward_model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto").eval()

            if not self.use_cot:
                os.environ['USE_COT'] = '1'
                self.use_cot = True
                print('[Warning] Since Best-of-N is enabled, USE_COT is forced to be set to 1.')

            print(f'Enable Best-of-N evaluation with PRM: {reward_model_path}')

        # self.image_size = self.model.config.vision_config.image_size
        self.version = version
        self.best_of_n = best_of_n
        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset, default=None) == 'OCR':
            # OCR transcription datasets own the output contract. Appending the
            # generic InternVL VQA prompt would incorrectly ask for a short answer.
            return False
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench', 'ChartQAPro', 'ChartMuseum'], dataset):  # noqa: E501
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_TYPE(dataset) == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        # --- 开始修改 (替换第 258 行) ---
        # 检查 'image_path' 是否不存在，但 'question_image_path' 存在
        if 'image_path' not in line and 'question_image_path' in line:
            # line 是一个 pandas Series，创建一个副本以避免警告
            line_copy = line.copy()

            # 将 'question_image_path' 的值 赋给 'image_path' 键
            line_copy['image_path'] = line['question_image_path']

            # 使用修正后的 line_copy 调用 dump_image
            tgt_path = self.dump_image(line_copy, dataset)
        else:
            # 如果 'image_path' 已经存在，或者两种路径都不存在，则按原样调用
            tgt_path = self.dump_image(line, dataset)
        # --- 修改结束 ---

        # tgt_path = self.dump_image(line, dataset)
        qimg_paths = None

        if dataset is not None and listinstr(['BMMR'], dataset):
            self.kwargs['max_new_tokens'] = max(self.kwargs.get('max_new_tokens', 4096), 8196)
            print(f'[Warning] BMMR dataset requires a larger max_new_tokens, set to {self.kwargs["max_new_tokens"]}')

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        #elif dataset is not None and DATASET_TYPE(dataset) == 'MM_MCQ':
            #prompt = build_mm_multi_choice_prompt(line, dataset)

           # def _to_list(v):
                """把字段安全转成 list[str]：支持 list/tuple、'[a, b]'、'a;b'、'a，b' 等。"""
               # if v is None:
               #     return []
               # if isinstance(v, (list, tuple)):
              #      return [str(x).strip() for x in v if str(x).strip()]
               # s = str(v).strip()
               # if s.startswith("[") and s.endswith("]"):
                #    s = s[1:-1].strip()
               # if not s:
                #    return []
               # parts = re.compile(r"\s*[;,，；]\s*").split(s)
                #return [p for p in (x.strip() for x in parts) if p]

           # qimg_paths = _to_list(line["question_image_path"])
        # --- 开始修改 (替换 'MM_MCQ' 逻辑) ---
        elif dataset is not None and DATASET_TYPE(dataset) == 'MM_MCQ':

            # 1. 复制 image_mm_mcq.py 中的 _to_list 辅助函数
            # (确保 re 和 pandas 已在文件顶部导入)
            def _to_list(v):
                """把字段安全转成 list[str]：支持 list/tuple、'[a, b]'、'a;b'、'a，b' 等。"""
                if v is None:
                    return []
                if isinstance(v, (list, tuple)):
                    return [str(x).strip() for x in v if str(x).strip()]
                s = str(v).strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1].strip()
                if not s:
                    return []
                parts = re.compile(r"\s*[;,，；]\s*").split(s)
                return [p for p in (x.strip() for x in parts) if p]

            # 2. 复制 image_mm_mcq.py 中的 prompt 构建逻辑 (不使用 line['question'])
            hint = line['hint'] if ('hint' in line and pd.notna(line['hint'])) else None
            prompt_lines = []
            if hint:
                prompt_lines.append(f"Hint: {str(hint).strip()}")

            prompt_lines.append(
                "Read the question and options shown in the image(s). "
                "Answer with only the single letter of the correct option (e.g., A, B, C, D)."
            )
            
            #short prompt
            #prompt_lines.append(
            #    "Read the question and options, then answer with only the single letter (e.g., A, B, C, D)."
            #)

            #long prompt
            #prompt_lines.append(
            #    "Please carefully read the question and options presented in the image(s). "
            #    "Ensure you understand the meaning of each option. "
            #    "Based on the question, choose the most appropriate answer and respond with only the letter of the correct option (e.g., A, B, C, or D). "
            #    "Do not include any additional text or explanations in your response."
            #)
            
            #understanding prompt
            #prompt_lines.append(
            #    "Please take a moment to carefully read the question and all available options shown in the image(s). "
            #    "Ensure you fully understand the context and meaning behind each option. "
            #    "After considering the question and options, choose the most appropriate answer. "
            #    "Respond only with the letter corresponding to the correct option (e.g., A, B, C, or D). "
            #    "Do not include any explanations or comments in your response."
            #)
            
            #CoT prompt
            #prompt_lines.append(
            #    "Analyze the image to identify the question text and the available options. "
            #    "Think step-by-step to deduce the correct answer based on the visual information provided. "
            #    "However, **discard your reasoning process** in the final output. "
            #    "Your final response must consist of **nothing but** the single letter of the correct option (e.g., A). "
            #    "Do not explain why, and do not use punctuation."
            #)
            prompt = "\n".join(prompt_lines).strip()

            # 3. 复制 image_mm_mcq.py 中的 qimg_paths 提取逻辑
            qimg_raw = line.get('question_image_path', line.get('question_image'))
            if not qimg_raw:
                raise ValueError(
                    "MMImageMCQDataset: missing 'question_image'/'question_image_path' for the current sample.")
            qimg_paths = _to_list(qimg_raw)

        # --- 修改结束 ---
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVerse'], dataset):
                question = question.replace("please directly answer the question and", "please")
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'GUI':
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)

            prompt_config = GUI_TEMPLATE[ds_basename]
            if 'history' in prompt_config["placeholders"]:
                traj_dict['history'] = pile_action_history(traj_dict['history'])
            prompt = format_nav_prompt(
                (
                    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>"  # noqa: E501
                    if self.screen_parse
                    else prompt_config["template"]
                ),
                prompt_config["placeholders"],
                action_space=action_space,
                **traj_dict,
            )
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        if qimg_paths is not None:
            message.extend([dict(type='image', value=s) for s in qimg_paths])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        elif DATASET_TYPE(dataset) == 'GUI':
            self.max_num = 12
        else:
            self.max_num = 6

    def generate_v1_2(self, message, dataset=None):
        self.INTERLEAVE = False
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                       question=prompt, generation_config=self.kwargs)
        return response

    def generate_v1_5(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])

        if DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            pixel_values_list = []
            for file_name in image_path:
                pixel_values_list.append(load_image(file_name, max_num=max_num).to(self.device).to(torch.bfloat16))
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            pixel_values = load_image(image_path, max_num=max_num).to(self.device).to(torch.bfloat16)
        else:
            pixel_values = None
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.kwargs,
                verbose=True)
        return response

    @torch.no_grad()
    def generate_v2(self, message, dataset=None):

        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        image_num = len([x for x in message if x['type'] == 'image'])
        #max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        #----------------------------------------------------------------------#
        if image_num == 0:
            # 如果是 MMLU 这样的纯文本数据集，image_num 为 0
            # 此时 max_num 应该设为 self.max_num (或者 1，取决于模型的逻辑)
            max_num = self.max_num 
        else:
            # 只有在 image_num 不为 0 时才执行除法
            max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        #----------------------------------------------------------------------#
        prompt = reorganize_prompt(message, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        response_list = []
        for idx in range(self.best_of_n):
            kwargs_default = self.kwargs.copy()
            kwargs_default['do_sample'] = idx > 0 or kwargs_default.get('do_sample', False)
            kwargs_default['temperature'] = 0.6
            kwargs_default['top_p'] = 0.95

            if self.use_lmdeploy:
                from lmdeploy import GenerationConfig
                gen_config = GenerationConfig(**kwargs_default)
                gen_config.random_seed = None
                messages_list = prepare_messages_list(prompt, image_path, system_prompt=self.system_prompt)
                assert len(messages_list) == 1
                response = self.model(messages_list, gen_config=gen_config)[0]
                response = response.text
            else:
                if self.system_prompt is not None:
                    self.model.system_message = self.system_prompt
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=prompt,
                    generation_config=kwargs_default,
                    verbose=idx == 0,
                )
            response_list.append(response)

        if self.best_of_n > 1:
            response_list = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
            )
        response = response_list[0]

        if dataset is not None and not listinstr(['WeMath'], dataset):
            if use_mpo_prompt:
                response = mpo_post_processing(response, dataset)
            elif self.use_cot and self.use_postprocess:
                response = extract_boxed_content(response)

        if dataset is not None and DATASET_TYPE(dataset) == 'GUI' and self.screen_parse:
            # Parse the bounding box coordinates from the response
            response = parse_bbox_internvl(response)
            # Normalize the coordinates to the range [0, 1]
            if isinstance(response, list):
                response = [item / 1000 for item in response]
                # Convert the coordinates to the format required by the GUI
                response = f"x={response[0]}, y={response[1]}"

        return response

    def _internvl_vllm_sampling_params(self):
        from vllm import SamplingParams
        do_sample = bool(self.kwargs.get('do_sample', False))
        temperature = float(self.kwargs.get('temperature', 0.6) or 0.6) if do_sample else 0.0
        top_p = self.kwargs.get('top_p', 1.0)
        top_p = 1.0 if top_p is None else float(top_p)
        top_k = self.kwargs.get('top_k', -1)
        top_k = -1 if top_k is None else int(top_k)
        repetition_penalty = float(self.kwargs.get('repetition_penalty', 1.0) or 1.0)
        stop_tokens = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|end|>']
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in stop_tokens]
        stop_token_ids = [tok for tok in stop_token_ids if tok is not None and tok >= 0]
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=int(self.kwargs.get('max_new_tokens', 4096)),
            stop_token_ids=stop_token_ids,
        )

    def _build_internvl_vllm_request(self, message, dataset=None):
        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            raise RuntimeError('InternVL vLLM path in VLMEvalKit currently supports image/text inputs, not video.')
        image_num = len([x for x in message if x['type'] == 'image'])
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        chat_messages = []
        if self.system_prompt is not None:
            chat_messages.append({'role': 'system', 'content': self.system_prompt})
        chat_messages.append({'role': 'user', 'content': prompt})
        rendered = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images = []
        for item in message:
            if item['type'] == 'image':
                with Image.open(item['value']) as img:
                    images.append(img.convert('RGB').copy())
        req = {'prompt': rendered}
        if images:
            req['multi_modal_data'] = {'image': images}
        return req

    def _postprocess_internvl_vllm(self, response, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (
            self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench']
        )
        if dataset is not None and not listinstr(['WeMath'], dataset):
            if use_mpo_prompt:
                response = mpo_post_processing(response, dataset)
            elif self.use_cot and self.use_postprocess:
                response = extract_boxed_content(response)
        if dataset is not None and DATASET_TYPE(dataset) == 'GUI' and self.screen_parse:
            response = parse_bbox_internvl(response)
            if isinstance(response, list):
                response = [item / 1000 for item in response]
                response = f"x={response[0]}, y={response[1]}"
        return response

    def generate_batch(self, messages, dataset=None):
        if not self.use_vllm:
            return [self.generate_inner(message, dataset=dataset) for message in messages]
        if self.version != 'V2.0':
            raise RuntimeError('InternVL vLLM backend is enabled only for V2.0/InternVL2+ style models.')
        if not messages:
            return []
        self.set_max_num(dataset)
        reqs = [self._build_internvl_vllm_request(message, dataset=dataset) for message in messages]
        outputs = self.llm.generate(reqs, sampling_params=self._internvl_vllm_sampling_params())
        return [
            self._postprocess_internvl_vllm(o.outputs[0].text, dataset=dataset)
            for o in outputs
        ]

    def generate_inner_vllm(self, message, dataset=None):
        return self.generate_batch([message], dataset=dataset)[0]

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        self.set_max_num(dataset)
        print(f'InternVL model version: {self.version}')
        if self.version in ['V1.1', 'V1.2']:
            return self.generate_v1_2(message, dataset)
        elif self.version == 'V1.5':
            return self.generate_v1_5(message, dataset)
        elif self.version == 'V2.0':
            return self.generate_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version: {self.version}')

    def build_history(self, message):
        # Global Variables
        image_path = []
        image_cnt = 0

        def concat_tilist(tilist):
            nonlocal image_cnt  # Declare image_cnt as nonlocal to modify it
            prompt = ''
            for item in tilist:
                # Substitute the pattern in the text
                if item['type'] == 'text':
                    prompt += re.sub(self.pattern, self.replacement, item['value'])
                elif item['type'] == 'image':
                    image_cnt += 1
                    prompt += '<image>\n'
                    image_path.append(item['value'])
            return prompt

        # Only previous messages
        assert len(message) % 2 == 0
        history = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            history.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))

        return history, image_path, image_cnt

    def chat_inner_v2(self, message, dataset=None):

        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ''

        # If message is just text in the conversation
        if len(current_msg['content']) == 1 and current_msg['content'][0]['type'] == 'text':
            question = current_msg['content'][0]['value']
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per InternVL
        else:
            for msg in current_msg['content']:
                if msg['type'] == 'text':
                    question += re.sub(self.pattern, self.replacement, msg['value'])
                elif msg['type'] == 'image':
                    image_cnt += 1
                    question += '<image>\n'
                    image_path.append(msg['value'])

        if image_cnt > 1:
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_cnt == 1:
            upscale_flag = listinstr(['MMMU_DEV_VAL'], dataset)
            pixel_values = load_image(
                image_path, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=question,
            generation_config=self.kwargs,
            history=history,
            return_history=True
        )

        response = re.sub(self.reverse_pattern, self.reverse_replacement, response)

        return response

    def chat_inner(self, message, dataset=None):
        self.set_max_num(dataset)

        if self.version in ['V1.1', 'V1.2']:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V1.5':
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V2.0':
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
            self.kwargs = kwargs_default
            return self.chat_inner_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
