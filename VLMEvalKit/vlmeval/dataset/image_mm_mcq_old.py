import re
import pandas as pd

from .image_mcq import ImageMCQDataset

_MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")


class MMImageMCQDataset(ImageMCQDataset):
    TYPE = "MM_MCQ"

    DATASET_URL = {
        'MMBench_EN_custom': '',
        'SEED_IMG_custom': '',
    }

    DATASET_MD5 = {
        'MMBench_EN_custom': None,
        'SEED_IMG_custom': None,
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
        prompt = "\n".join(prompt_lines).strip()

        # ---------- 组织消息：题干图 -> 主图 -> 文本 ----------
        msgs = []
        _add_imgs(msgs, main_paths)
        _add_imgs(msgs, qimg_paths)
        msgs.append(dict(type='text', value=prompt))

        return msgs


class MMImageMCQMMMUDataset(ImageMCQDataset):
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

    @staticmethod
    def _assemble_by_placeholders_interleave(text: str, images: list):
        """
        按 <image k> 把 text 拆解，并用 images[k-1] 插入：
        返回多模态 msgs 列表。
        """
        if not text or "<image" not in text.lower():
            # 无占位符：默认把所有图片放在文本前面
            out = []
            out += [{"type": "image", "value": p} for p in images]
            out.append({"type": "text", "value": text})
            return out

        out, last = [], 0
        for m in MMImageMCQMMMUDataset._TAG.finditer(text):
            # 追加前段文本
            seg = text[last:m.start()]
            if seg:
                out.append({"type": "text", "value": seg})
            # 插入图片
            k = int(m.group(1))
            if 1 <= k <= len(images):
                out.append({"type": "image", "value": images[k - 1]})
            else:
                out.append({"type": "text", "value": f"[MISSING_IMAGE_{k}]"})
            last = m.end()
        # 末尾文本
        tail = text[last:]
        if tail:
            out.append({"type": "text", "value": tail})
        return out
    
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
        return msgs
