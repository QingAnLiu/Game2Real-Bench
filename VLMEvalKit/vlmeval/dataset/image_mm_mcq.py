import re
import string
import pandas as pd

from .image_mcq import ImageMCQDataset

_MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")


class MMImageMCQDataset(ImageMCQDataset):
    TYPE = "MM_MCQ"

    DATASET_URL = {
        'MMBench_EN_custom': '',
        'MMBench_EN_custom_sample': '',
        'MMBench_EN_B_custom': '',
        'MMBench_EN_C_custom': '',
        'MMBench_EN_T_custom': '',
        'MMBench_EN_custom_9': '',
        'MMBench_EN_custom_32': '',
        'MMBench_EN_custom_48': '',
        'MMBench_EN_custom_48_sample': '',
        'MMBench_EN_custom_64': '',
        'MMBench_EN_custom_9_cx': '',
        'SEED_IMG_custom': '',
        'MMMU_custom': '',
        'MMMU_custom_sample': '',
        'MMMU_B_custom': '',
        'MMLU_B_custom': '',
        'MMMU_C_custom': '',
        'MMMU_T_custom': '',
        'MMMU_custom_9': '',
        'MMLU_custom_9': '',
        'MMMU_custom_32': '',
        'MMMU_custom_48': '',
        'MMMU_custom_64': '',
        'MMMU_custom_9_cx': '',
        'C_bench':'',
        'C_bench_fixed': '',
        'C_bench_fixed_test': '',
        "MMBench_EN_Arial_16": "",
        "MMBench_EN_Brush_Script_MT_16": "",
        "MMBench_EN_Cambria_16": "",
        "MMBench_EN_Times_New_Roman_16": "",
        "MMLU_32": "",
        "MMLU_48": "",
        "MMLU_64": "",
        "MMLU_9": "",
        "MMLU_16": "",
    }

    DATASET_MD5 = {
        'MMBench_EN_custom': None,
        'MMBench_EN_custom_sample': None,
        'MMBench_EN_B_custom': None,
        'MMBench_EN_C_custom': None,
        'MMBench_EN_T_custom': None,
        'MMBench_EN_custom_9': None,
        'MMBench_EN_custom_32': None,
        'MMBench_EN_custom_48': None,
        'MMBench_EN_custom_48_sample': None,
        'MMBench_EN_custom_64': None,
        'MMBench_EN_custom_9_cx': None,
        'SEED_IMG_custom': None,
        'MMMU_custom': None,
        'MMMU_custom_sample': None,
        'MMMU_B_custom': None,
        'MMLU_B_custom': None,
        'MMMU_C_custom': None,
        'MMMU_T_custom': None,
        'MMMU_custom_9': None,
        'MMLU_custom_9': None,
        'MMMU_custom_32': None,
        'MMMU_custom_48': None,
        'MMMU_custom_64': None,
        'MMMU_custom_9_cx': None,
        'C_bench':None,
        'C_bench_fixed': None,
        'C_bench_fixed_test': None,
        "MMBench_EN_Arial_16.tsv": None,
        "MMBench_EN_Brush_Script_MT_16.tsv": None,
        "MMBench_EN_Cambria_16.tsv": None,
        "MMBench_EN_Times_New_Roman_16.tsv": None,
        "MMLU_32": None,
        "MMLU_48": None,
        "MMLU_64": None,
        "MMLU_9": None,
        "MMLU_16": None,
    }

    import re
    import pandas as pd
    # 如果外层没引入 string，这里不需要它了
    _MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # ---------- 工具函数 ----------
        def _get(col, default=None):
            return line[col] if (col in line) else default

        def _nonempty(v) -> bool:
            """安全地判断“是否有内容”。支持标量/字符串/list/tuple。"""
            if v is None:
                return False
            # list/tuple：只要有一个非空且不是 nan/none/null 的子项就算非空
            if isinstance(v, (list, tuple)):
                for x in v:
                    sx = str(x).strip()
                    if sx and sx.lower() not in {"nan", "none", "null", "[]"}:
                        return True
                return False
            # 其他：优先用字符串判空；再尽量用 pd.isna 判定 NaN
            s = str(v).strip()
            if s == "" or s.lower() in {"nan", "none", "null"}:
                return False
            try:
                return not pd.isna(v)
            except Exception:
                return True  # 无法判定 isna 时按有值处理

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
            parts = _MULTI_SPLIT_RE.split(s)
            return [p for p in (x.strip() for x in parts) if p]

        def _add_imgs(msgs, paths):
            if not paths:
                return
            if isinstance(paths, list):
                msgs.extend([dict(type='image', value=p) for p in paths])
            else:
                msgs.append(dict(type='image', value=paths))

        # ---------- 题干图 ----------
        qimg_raw = _get('question_image_path')
        if not _nonempty(qimg_raw):
            qimg_raw = _get('question_image')
        if not _nonempty(qimg_raw):
            raise ValueError(
                "MMImageMCQDataset: missing 'question_image'/'question_image_path' for the current sample.")
        qimg_paths = _to_list(qimg_raw)

        # ---------- 主图 ----------
        if self.meta_only:
            main_raw = _get('image_path', _get('image'))
            main_paths = _to_list(main_raw)
        else:
            main_paths = self.dump_image(line)

        # ---------- 可选 hint，且不再拼接 A/B/C/D 文本 ----------
        hint = _get('hint')
        prompt_lines = []
        if _nonempty(hint):
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

        # ---------- 组织消息：题干图 -> 主图 -> 文本 ----------
        msgs = []
        _add_imgs(msgs, main_paths)
        _add_imgs(msgs, qimg_paths)
        msgs.append(dict(type='text', value=prompt))

        return msgs


'''class MMImageMCQMMMUDataset(ImageMCQDataset):
    """
    处理“题干文本中混排 <image k> 占位符”的 MMMU 类样式：
    - 先用 ImageMCQDataset.build_prompt 生成文本版 prompt（包含 <image k>）
    - 再用本行 question_image_path / question_image 的图片按 1-based 顺序填充占位符
    - 最终得到interleave image/text 的prompt
    """
    TYPE = "MM_MMMU_MCQ"

    DATASET_URL = {
        'MMMU_custom': '',
    }
    DATASET_MD5 = {
        'MMMU_custom': None,
    }

    _TAG = re.compile(r"<\s*image\s*(\d+)\s*>", re.IGNORECASE)
    _SPLIT = re.compile(r"\s*[;,，；]\s*")

    @staticmethod
    def _nonempty(v) -> bool:
        if v is None: return False
        if isinstance(v, (list, tuple)):
            for x in v:
                sx = str(x).strip().lower()
                if sx and sx not in {"nan", "none", "null", "[]"}:
                    return True
            return False
        s = str(v).strip().lower()
        if s in {"", "nan", "none", "null"}: return False
        try:
            return not pd.isna(v)
        except Exception:
            return True

    def _to_list(self, v):
        """支持 list/tuple、'[a,b]'、'a;b'、'a，b' 等 -> List[str]"""
        if v is None: return []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s: return []
        parts = self._SPLIT.split(s)
        return [p for p in (x.strip() for x in parts) if p]

    @staticmethod
    def _get_text_from_msgs(msgs):
        """从 super().build_prompt 的 msgs 里取唯一的 text，并忽略其中的 image。"""
        text = None
        for s in msgs:
            if s["type"] == "text":
                text = s["value"] if text is None else (text + s["value"])
        return text or ""
    
    def _assemble_by_placeholders(text: str, images: list):
        """
        无论 text 中是否包含 <image k> 占位符，
        都将所有图片(images)放在文本前面。
        返回多模态 msgs 列表。
        """
        out = []
        # 1. 先添加所有图片
        out += [{"type": "image", "value": p} for p in images]

        # 2. 再添加完整文本（不再根据占位符拆分）
        if text:
            out.append({"type": "text", "value": text})

        return out

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # 1) 先用基类生成“文本版”prompt（里面应包含 <image k>）
        base_msgs = super().build_prompt(line)
        text = self._get_text_from_msgs(base_msgs)

        # 2) 从行里拿题干图片列表（按 1..N 顺序与占位符对应）
        qimg_raw = line['question_image_path'] if ('question_image_path' in line) else line.get('question_image', None)
        if not self._nonempty(qimg_raw):
            raise ValueError("MMImageMCQMMMUDataset: missing 'question_image_path'/'question_image'.")

        qimgs = self._to_list(qimg_raw)  # e.g. ["questions/236.png","questions/236_2.png"]

        # 3) 占位符展开：用 qimgs 按 <image k> 替换成真正的多模态序列
        msgs = self._assemble_by_placeholders(text, qimgs)
        return msgs'''


class MMLImageMCQDataset(ImageMCQDataset):
    TYPE = "MML_MCQ"

    DATASET_URL = {
        'MMLU_VAL': '',
    }

    DATASET_MD5 = {
        'MMLU_VAL': None,
    }
    import re
    import pandas as pd
    # 如果外层没引入 string，这里不需要它了
    _MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # 1. 读取所有文本列
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        
        # 2. 拼接成一个纯文本 prompt
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Answer with only the single letter of the correct option (e.g., A, B, C, D). \n'

        # 3. 只返回文本消息，不含任何图片
        msgs = [dict(type='text', value=prompt)]

        return msgs


class MML2ImageMCQDataset(ImageMCQDataset):
    TYPE = "MML2_MCQ"

    DATASET_URL = {
        'MMLU_VAL_custom': '',
    }

    DATASET_MD5 = {
        'MMLU_VAL_custom': None,
    }
    import re
    import pandas as pd
    # 如果外层没引入 string，这里不需要它了
    _MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # ---------- 工具函数 ----------
        def _get(col, default=None):
            return line[col] if (col in line) else default

        def _nonempty(v) -> bool:
            """安全地判断“是否有内容”。支持标量/字符串/list/tuple。"""
            if v is None:
                return False
            # list/tuple：只要有一个非空且不是 nan/none/null 的子项就算非空
            if isinstance(v, (list, tuple)):
                for x in v:
                    sx = str(x).strip()
                    if sx and sx.lower() not in {"nan", "none", "null", "[]"}:
                        return True
                return False
            # 其他：优先用字符串判空；再尽量用 pd.isna 判定 NaN
            s = str(v).strip()
            if s == "" or s.lower() in {"nan", "none", "null"}:
                return False
            try:
                return not pd.isna(v)
            except Exception:
                return True  # 无法判定 isna 时按有值处理

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
            parts = _MULTI_SPLIT_RE.split(s)
            return [p for p in (x.strip() for x in parts) if p]

        def _add_imgs(msgs, paths):
            if not paths:
                return
            if isinstance(paths, list):
                msgs.extend([dict(type='image', value=p) for p in paths])
            else:
                msgs.append(dict(type='image', value=paths))

        # ---------- 题干图 ----------
        qimg_raw = _get('question_image_path')
        if not _nonempty(qimg_raw):
            qimg_raw = _get('question_image')
        if not _nonempty(qimg_raw):
            raise ValueError(
                "MMImageMCQDataset: missing 'question_image'/'question_image_path' for the current sample.")
        qimg_paths = _to_list(qimg_raw)

        hint = _get('hint')
        prompt_lines = []
        if _nonempty(hint):
            prompt_lines.append(f"Hint: {str(hint).strip()}")
        prompt_lines.append(
            "Read the question and options shown in the image(s). "
            "Answer with only the single letter of the correct option (e.g., A, B, C, D)."
        )
        prompt = "\n".join(prompt_lines).strip()

        # ---------- 组织消息：题干图 -> 文本 ----------
        msgs = []
        _add_imgs(msgs, qimg_paths)
        msgs.append(dict(type='text', value=prompt))

        return msgs


'''if __name__ == "__main__":
    
    # --- 准备工作 (让“from .image_mcq”不报错) ---
    import pprint
    import pandas as pd
    import re 
    
    class ImageMCQDataset:
        pass 

    print("--- 启动 MMImageMCQDataset 调试 ---")

    mock_data_line = {
        'index': 101,
        'image_path': "['/path/to/main_A.jpg', '/path/to/main_B.jpg']", # <-- 多个主图
        'question_image_path': '/path/to/question_image.png',           # <-- 单个题干图
        'hint': '这是一个测试提示。',
        'answer': 'A'
    }

    print("创建空白实例 (跳过 __init__)...")
    dataset = MMImageMCQDataset.__new__(MMImageMCQDataset)

    
    def mock_dump_image_for_main_images(line_dict):
        """
        这是一个假的 'dump_image' 方法，专门用于测试。
        它会返回我们模拟的多张主图。
        """
        print(f"[Debug Mock] 'dump_image' (for MAIN images) 被调用，索引: {line_dict.get('index')}")
        # 返回一个包含多张主图的 *列表*
        return ['/path/to/main_A.jpg', '/path/to/main_B.jpg']

    # 把我们的“假”方法，安装到“空白”实例上
    dataset.dump_image = mock_dump_image_for_main_images
    
    # 告诉 build_prompt 去调用 self.dump_image
    dataset.meta_only = False 

    # 5. 调用 build_prompt
    #    ↓↓↓ 在这里设置您的断点 (F9) ↓↓↓
    print("正在调用 build_prompt... (请在此设置断点)")
    result_msgs = dataset.build_prompt(mock_data_line) 
    
    print("\n--- 调试输出 (最终的消息列表): ---")
    pprint.pprint(result_msgs)'''