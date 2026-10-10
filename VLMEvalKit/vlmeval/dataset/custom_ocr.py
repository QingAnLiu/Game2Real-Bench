import re
import unicodedata

import pandas as pd

from .image_base import ImageBaseDataset
from .image_mixedqa import ImageMixedQADataset
from ..smp import LMUDataRoot, dump, load, osp, toliststr
from ..smp.file import get_intermediate_file_path


OCR_TRANSCRIPTION_PROMPT = """Perform OCR on this image. Transcribe all visible text in natural reading order.

Requirements:
1. Do not answer or solve the question in the image.
2. Output only the transcription.
3. Do not add explanations, comments, Markdown, or code fences.
4. Preserve punctuation and capitalization.
5. Preserve logical paragraph breaks, but ignore line wrapping caused only by image width."""


def _text(value):
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value)


def strict_ocr_text(value):
    """Apply transport-only cleanup for strict OCR comparison."""
    value = _text(value).replace('\r\n', '\n').replace('\r', '\n')
    return value.lstrip('\ufeff').strip()


def normalize_ocr_text(value):
    """Normalize layout whitespace while retaining case and punctuation."""
    value = strict_ocr_text(value)
    value = unicodedata.normalize('NFKC', value)

    # Remove a code fence only when it wraps the complete model response.
    fence = re.fullmatch(r'```(?:text)?\s*\n?(.*?)\n?```', value, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        value = fence.group(1)

    lines = [re.sub(r'[ \t\f\v]+', ' ', line).strip() for line in value.split('\n')]
    value = '\n'.join(line for line in lines if line)
    return re.sub(r'\s+', ' ', value).strip()


def edit_distance(reference, hypothesis):
    """Levenshtein distance for strings or token sequences, using O(min(n, m)) memory."""
    if reference == hypothesis:
        return 0
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def calculate_ocr_metrics(references, predictions):
    if len(references) != len(predictions):
        raise ValueError(
            f'OCR references and predictions have different lengths: {len(references)} != {len(predictions)}'
        )
    if not references:
        raise ValueError('Cannot evaluate an empty OCR result file.')

    details = []
    total_char_distance = 0
    total_reference_chars = 0
    total_word_distance = 0
    total_reference_words = 0

    for reference, prediction in zip(references, predictions):
        raw_reference = strict_ocr_text(reference)
        raw_prediction = strict_ocr_text(prediction)
        norm_reference = normalize_ocr_text(raw_reference)
        norm_prediction = normalize_ocr_text(raw_prediction)

        char_distance = edit_distance(norm_reference, norm_prediction)
        reference_words = norm_reference.split()
        prediction_words = norm_prediction.split()
        word_distance = edit_distance(reference_words, prediction_words)

        total_char_distance += char_distance
        total_reference_chars += len(norm_reference)
        total_word_distance += word_distance
        total_reference_words += len(reference_words)

        max_chars = max(len(norm_reference), len(norm_prediction))
        normalized_similarity = 1.0 if max_chars == 0 else 1.0 - char_distance / max_chars
        sample_cer = (
            0.0 if not norm_reference and not norm_prediction
            else char_distance / max(1, len(norm_reference))
        )
        sample_wer = (
            0.0 if not reference_words and not prediction_words
            else word_distance / max(1, len(reference_words))
        )
        details.append({
            'ocr_raw_reference': raw_reference,
            'ocr_raw_prediction': raw_prediction,
            'ocr_normalized_reference': norm_reference,
            'ocr_normalized_prediction': norm_prediction,
            'ocr_raw_exact_match': float(raw_reference == raw_prediction),
            'ocr_normalized_exact_match': float(norm_reference == norm_prediction),
            'ocr_char_distance': char_distance,
            'ocr_word_distance': word_distance,
            'ocr_sample_cer': sample_cer,
            'ocr_sample_wer': sample_wer,
            'ocr_normalized_edit_similarity': normalized_similarity,
            'ocr_empty_prediction': float(not norm_prediction),
        })

    detail_df = pd.DataFrame(details)
    cer = total_char_distance / max(1, total_reference_chars)
    wer = total_word_distance / max(1, total_reference_words)
    summary = {
        'evaluated_samples': len(details),
        'empty_predictions': int(detail_df['ocr_empty_prediction'].sum()),
        'raw_exact_match_rate': float(detail_df['ocr_raw_exact_match'].mean()),
        'normalized_exact_match_rate': float(detail_df['ocr_normalized_exact_match'].mean()),
        'cer': float(cer),
        'character_accuracy': float(max(0.0, 1.0 - cer)),
        'wer': float(wer),
        'mean_normalized_edit_similarity': float(detail_df['ocr_normalized_edit_similarity'].mean()),
        'total_reference_characters': int(total_reference_chars),
        'total_reference_words': int(total_reference_words),
    }
    return summary, detail_df


class CustomOCRDataset(ImageBaseDataset):
    TYPE = 'OCR'
    DATASET_FILES = {
        'QuestionImageOCR': osp.join('resource', 'C_bench', 'QuestionImageOCR.tsv'),
    }

    def __init__(self, dataset='QuestionImageOCR', data_file=None, skip_noimg=True):
        self.custom_data_file = data_file
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def load_data(self, dataset):
        if self.custom_data_file:
            data_path = self.custom_data_file
        elif dataset in self.DATASET_FILES:
            repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
            data_path = osp.join(repo_root, self.DATASET_FILES[dataset])
        else:
            data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        data_path = osp.abspath(osp.expanduser(data_path))
        if not osp.isfile(data_path):
            raise FileNotFoundError(
                f'OCR dataset TSV does not exist: {data_path}. '
                'Generate it with scripts/prepare_question_image_ocr.py.'
            )
        self.data_path = data_path
        data = load(data_path)
        required = {'index', 'image_path', 'answer'}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f'OCR dataset TSV is missing required columns: {missing}')
        if 'question' not in data:
            data['question'] = OCR_TRANSCRIPTION_PROMPT

        # Resolve portable paths relative to the OCR TSV. This lets the checked-in
        # resource work on both Windows and the Linux /defaultShare environment.
        data_dir = osp.dirname(data_path)
        resolved_paths = []
        for value in data['image_path']:
            paths = toliststr(value)
            paths = [
                path if osp.isabs(path) else osp.abspath(osp.join(data_dir, path))
                for path in paths
            ]
            resolved_paths.append(paths[0] if len(paths) == 1 else paths)
        data['image_path'] = resolved_paths
        return data

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_FILES)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_paths = self.dump_image(line)
        message = [dict(type='image', value=path) for path in image_paths]
        message.append(dict(type='text', value=OCR_TRANSCRIPTION_PROMPT))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        data = load(eval_file)
        missing = sorted({'answer', 'prediction'} - set(data.columns))
        if missing:
            raise ValueError(f'OCR prediction file is missing required columns: {missing}')

        summary, details = calculate_ocr_metrics(
            data['answer'].tolist(),
            data['prediction'].tolist(),
        )
        detailed_data = pd.concat([data.reset_index(drop=True), details], axis=1)

        detail_file = get_intermediate_file_path(eval_file, '_ocr_details')
        summary_file = get_intermediate_file_path(eval_file, '_ocr_metrics', 'csv')
        dump(detailed_data, detail_file)

        summary_df = pd.DataFrame({
            'metric': list(summary.keys()),
            'value': list(summary.values()),
        })
        dump(summary_df, summary_file)
        return summary_df


class CBenchOCRTextDataset(ImageMixedQADataset):
    """C-Bench with OCR-recovered text prompts and original main images."""

    TYPE = 'MCQ'

    DATASET_FILES = {
        'C_bench_Qwen3VL8B_OCRText': osp.join(
            'resource', 'C_bench', 'C_bench_Qwen3VL8B_OCRText.tsv'
        ),
        'C_bench_InternVL35_8B_OCRText': osp.join(
            'resource', 'C_bench', 'C_bench_InternVL35_8B_OCRText.tsv'
        ),
    }

    def __init__(self, dataset='C_bench_Qwen3VL8B_OCRText', data_file=None, skip_noimg=True):
        self.custom_data_file = data_file
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def load_data(self, dataset):
        if self.custom_data_file:
            data_path = self.custom_data_file
        elif dataset in self.DATASET_FILES:
            repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
            data_path = osp.join(repo_root, self.DATASET_FILES[dataset])
        else:
            data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        data_path = osp.abspath(osp.expanduser(data_path))
        if not osp.isfile(data_path):
            raise FileNotFoundError(
                f'OCR text-condition TSV does not exist: {data_path}. '
                'Generate it with scripts/ocr_transcription_to_cbench_text.py.'
            )
        self.data_path = data_path
        self.data_dir = osp.dirname(data_path)
        data = load(data_path)
        required = {'index', 'question', 'question_type', 'answer', 'image_path'}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f'OCR text-condition TSV is missing required columns: {missing}')
        return self.normalize_data_paths(data, self.data_dir)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_FILES)
