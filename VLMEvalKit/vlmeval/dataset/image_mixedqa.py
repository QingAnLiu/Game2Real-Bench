import os
import os.path as osp
import re
import string
import warnings
import json

import pandas as pd
from tqdm import tqdm

from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge
from ..smp import *
from ..utils import can_infer


def split_mixedqa_paths(value):
    paths = toliststr(value)
    if len(paths) == 1 and isinstance(paths[0], str):
        text = paths[0].strip()
        if ',' in text and not text.startswith('['):
            paths = [x.strip() for x in text.split(',') if x.strip()]
    return paths


def is_mixedqa_path_metadata(value):
    """Whether a legacy `image` value is a path string rather than base64 data."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text or text.startswith('data:image'):
        return False
    has_image_suffix = any(suffix in text for suffix in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
    return has_image_suffix and ('/' in text or '\\' in text)


def normalize_mixedqa_open_answer(answer):
    answer = str(answer).strip().lower()
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'\s*;\s*', '; ', answer)
    answer = re.sub(r'\s*\|\s*', ' | ', answer)
    return answer.rstrip('.')


def normalize_mixedqa_choice_answer(answer, choices):
    valid = set(choices)
    compact = re.sub(r'[^A-Z]', '', str(answer).upper())
    if not compact or not set(compact).issubset(valid) or len(set(compact)) != len(compact):
        return False
    return ''.join(sorted(compact))


def build_mixedqa_choices(item):
    return {
        cand: item[cand]
        for cand in string.ascii_uppercase
        if cand in item and not pd.isna(item[cand])
    }


def build_mixedqa_option_str(choices):
    return '\n'.join([f'{k}. {v}' for k, v in choices.items()])


def is_mixedqa_option_image(path):
    """Whether an image filename explicitly identifies it as a candidate image."""
    filename = osp.basename(str(path).replace('\\', '/'))
    return re.search(r'(?i)(?:^|[_-])(?:option|candidate)[_-][A-Z](?:[_\-.]|$)', filename) is not None


def split_mixedqa_text_list(value):
    """Parse optional TSV metadata containing a JSON list of short text descriptions."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    # Human-editable fallback if a TSV is authored without JSON.
    if '||' in text:
        return [x.strip() for x in text.split('||') if x.strip()]
    return [text]


def infer_mixedqa_view_name(path):
    """Infer an orthographic view label from a canonical GameBench filename."""
    filename = osp.basename(str(path).replace('\\', '/')).lower()
    match = re.search(r'_(top|front|right|left|bottom|back)_(?:base|question)\.(?:png|jpg|jpeg|webp|bmp)$', filename)
    if not match:
        match = re.search(r'_(top|front|right|left|bottom|back)_base\.(?:png|jpg|jpeg|webp|bmp)$', filename)
    return match.group(1).title() if match else None


def infer_mixedqa_question_image_roles(line, question_paths):
    """
    Return one description per question/reference image.

    Preferred source: TSV column `question_image_roles`, stored as a JSON list.
    Task-specific fallbacks keep older GameBench TSVs readable.
    """
    explicit = split_mixedqa_text_list(line.get('question_image_roles', '')) if line is not None else []
    if len(explicit) == len(question_paths):
        return explicit

    subfield = str(line.get('subfield', '')).strip().lower() if line is not None else ''

    if subfield == 'rolling_cube' and len(question_paths) == 2:
        return [
            'Board and ordered roll path from the Start cell S to the End cell E.',
            "Cube's initial 3x3 sticker layouts on the six world-facing sides before the first roll.",
        ]

    if subfield == 'cylinder_coloring' and len(question_paths) >= 2:
        roles = []
        for path in question_paths:
            view_name = infer_mixedqa_view_name(path)
            if not view_name:
                return []
            roles.append(f'{view_name} view of the same cylinder model.')
        return roles

    # Legacy safeguard only. The released Toy Game TSV uses only the annotated
    # question image for region-based Hidden Object items.
    if subfield == 'hidden_object' and len(question_paths) == 2:
        return [
            'Annotated question image showing the labeled query region(s).',
            'Clean view of the same underlying scene without the region overlay.',
        ]

    return []


def infer_mixedqa_question_image_note(line, question_paths):
    """Return an optional relation/instruction spanning the question images."""
    if line is not None:
        value = line.get('question_image_note', '')
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if not missing and str(value).strip():
            return str(value).strip()

    subfield = str(line.get('subfield', '')).strip().lower() if line is not None else ''

    if subfield == 'rolling_cube' and len(question_paths) == 2:
        return 'Use Image 2 as the initial cube state and execute the numbered roll sequence shown in Image 1.'

    if subfield == 'cylinder_coloring' and len(question_paths) >= 2:
        return 'Use the given orthographic views together to infer the requested missing view.'

    if subfield == 'hidden_object' and len(question_paths) == 2:
        return (
            'The two images depict the same scene; use the annotated image for region boundaries '
            'and do not count the same animal twice.'
        )

    return ''


def build_mixedqa_extractor_prompt(question, options, prediction, question_type):
    if question_type in ['multiple_choice', 'single_or_multiple_choice']:
        tmpl = (
            'You are an answer extractor for a visual benchmark.\n'
            'The question may have one or more correct options.\n'
            'Given the question, options, and a model response, extract the final selected option letters.\n'
            'Use the final answer when it appears in forms like boxed{{}}, <answer>, "final answer", or "answer is".\n'
            'Ignore earlier reasoning that mentions other options.\n'
            'If the response gives option content instead of letters, map it to matching option letters.\n'
            'Output only the uppercase option letters with no spaces, commas, or explanation, such as A, AC, or BCD.\n'
            'If no unique valid option set can be extracted, output Z.\n\n'
            'Question:\n{}\n\nOptions:\n{}\n\nModel response:\n{}\n\nOutput:'
        )
    else:
        tmpl = (
            'You are an answer extractor for a visual benchmark.\n'
            'The question has exactly one correct option.\n'
            'Given the question, options, and a model response, extract the final selected option letter.\n'
            'Use the final answer when it appears in forms like boxed{{}}, <answer>, "final answer", or "answer is".\n'
            'Ignore earlier reasoning that mentions other options.\n'
            'If the response gives option content instead of a letter, map it to the matching option letter.\n'
            'Output only one uppercase option letter, such as A, B, C, or D.\n'
            'If no unique valid option can be extracted, output Z.\n\n'
            'Question:\n{}\n\nOptions:\n{}\n\nModel response:\n{}\n\nOutput:'
        )
    return tmpl.format(question, options, prediction)


def build_mixedqa_open_judge_prompt(question, reference, prediction):
    tmpl = (
        'You are a strict judge for a visual benchmark.\n'
        'First extract the model response final answer; ignore reasoning, failed attempts, and explanations.\n'
        'Then decide whether the final answer is equivalent to the reference answer.\n'
        'Be tolerant of harmless formatting differences such as JSON vs text, whitespace, punctuation, and capitalization.\n'
        'Be strict about required content: numbers, colors, coordinates, order, paths, moves, board states, and fields must match.\n'
        'Do not solve the task yourself. Do not give partial credit.\n'
        'If the final answer misses, adds, reorders, or changes any required item, it is incorrect.\n'
        'Output only Correct or Incorrect.\n\n'
        'Question:\n{}\n\nReference answer:\n{}\n\nModel response:\n{}\n\nJudgement:'
    )
    return tmpl.format(question, reference, prediction)


class ImageMixedQADataset(ImageBaseDataset):
    TYPE = 'MixQA'
    # Game2Real-Bench release aliases. The corresponding TSV files are
    # prepared under $LMUData by utils/prepare_game2real_data.py.
    DATASET_URL = {}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        data_path = self.DATASET_URL.get(dataset, None)
        if data_path is None:
            data_path = dataset + '.tsv'
        if osp.exists(data_path):
            data = load(data_path)
            self.data_path = data_path
        else:
            file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
            data = self.prepare_tsv(data_path, file_md5)
            data_path = self.data_path
        self.data_dir = osp.dirname(data_path)

        # Some Game-Bench TSVs retain a legacy `image` column containing
        # semicolon-separated path metadata while `image_path` already carries
        # canonical JSON paths. ImageBaseDataset treats `image` as base64, so
        # remove it only when every non-null value is clearly path-like.
        if 'image' in data and 'image_path' in data:
            image_values = data['image'].dropna()
            if len(image_values) and all(is_mixedqa_path_metadata(value) for value in image_values):
                data = data.drop(columns=['image'])
        return self.normalize_data_paths(data, self.data_dir)

    @staticmethod
    def normalize_data_paths(data, data_dir):
        if 'image_path' not in data and 'question_image' in data:
            image_paths = []
            for value in data['question_image']:
                paths = []
                for path in split_mixedqa_paths(value):
                    path = str(path).replace('\\', '/').strip()
                    if osp.isabs(path) or re.match(r'^[A-Za-z]:/', path):
                        paths.append(path)
                    else:
                        paths.append(osp.abspath(osp.join(data_dir, path)).replace('\\', '/'))
                image_paths.append(paths[0] if len(paths) == 1 else paths)
            data['image_path'] = image_paths

        if 'image_path' in data:
            fixed = []
            for value in data['image_path']:
                paths = []
                for path in split_mixedqa_paths(value):
                    path = str(path).replace('\\', '/').strip()
                    if osp.isabs(path) or re.match(r'^[A-Za-z]:/', path):
                        paths.append(path)
                    else:
                        paths.append(osp.abspath(osp.join(data_dir, path)).replace('\\', '/'))
                fixed.append(paths[0] if len(paths) == 1 else paths)
            data['image_path'] = fixed

        return data

    @staticmethod
    def infer_question_type(line, has_options):
        question_type = str(line.get('question_type', '')).strip().lower()
        aliases = {
            'single': 'single_choice',
            'single-choice': 'single_choice',
            'multiple': 'multiple_choice',
            'multi_choice': 'multiple_choice',
            'multi-choice': 'multiple_choice',
            'single_or_multiple_choice': 'multiple_choice',
            'open_ended': 'open',
            'open-ended': 'open',
            'free_form': 'open',
        }
        question_type = aliases.get(question_type, question_type)
        if question_type:
            return question_type
        answer = str(line.get('GT', line.get('answer', ''))).strip()
        if has_options and re.search(r'[A-Z]\s*,\s*[A-Z]', answer.upper()):
            return 'multiple_choice'
        if has_options:
            return 'single_choice'
        return 'open'

    @staticmethod
    def answer_instruction(question_type, has_options):
        base = (
            'Answer directly. Do not explain your reasoning.\n'
            'For single-choice questions, answer with one option letter only, such as A.\n'
            'For multiple-answer questions, answer with all correct option letters separated by commas, such as A,C,E.\n'
            'For open questions, follow the required answer format exactly.\n'
        )
        if question_type == 'single_choice':
            return base + 'This is a single-choice question. Return one option letter only.\n'
        if question_type in ['multiple_choice', 'single_or_multiple_choice']:
            return (
                base
                + 'This is a multiple-answer question. Select all correct options. '
                'There may be more than one correct answer. Return option letters only, separated by commas.\n'
            )
        if question_type == 'open':
            return base + 'This is an open question. Return only the requested structured answer.\n'
        if has_options:
            return base + 'Return only the option letter(s) in the required format.\n'
        return base + 'Return only the final answer in the required format.\n'

    @staticmethod
    def image_instruction(image_paths, choices, line=None):
        paths = image_paths if isinstance(image_paths, list) else [image_paths]
        if not paths:
            return ''

        option_letters = list(choices.keys())
        option_flags = [is_mixedqa_option_image(path) for path in paths]
        option_count = len(option_letters)
        option_start = None

        if option_count:
            marked_indices = [idx for idx, marked in enumerate(option_flags) if marked]
            if len(marked_indices) == option_count:
                candidate_start = len(paths) - option_count
                if marked_indices == list(range(candidate_start, len(paths))):
                    option_start = candidate_start
            elif not marked_indices and len(paths) == option_count + 1:
                # Backward compatibility for legacy one-base-plus-options data.
                option_start = 1

        question_count = option_start if option_start is not None else len(paths)
        question_paths = paths[:question_count]
        roles = infer_mixedqa_question_image_roles(line, question_paths)
        relation_note = infer_mixedqa_question_image_note(line, question_paths)

        # Preserve the historical no-extra-text behavior for a plain single-image
        # open question unless explicit multi-image/option semantics need describing.
        if len(paths) == 1 and not roles and not relation_note:
            return ''

        lines = []

        if question_count == 1:
            if roles:
                lines.append(f'Image 1: {roles[0]}')
            elif option_start is not None:
                lines.append('Image 1 is the main question image.')
        elif question_count > 1:
            lines.append(f'Images 1-{question_count} are question/reference images.')
            if len(roles) == question_count:
                for idx, role in enumerate(roles, start=1):
                    lines.append(f'Image {idx}: {role}')
            else:
                lines.append('Use all question/reference images together in the order provided.')
            # Even with explicit roles, make the multi-image contract explicit.
            if len(roles) == question_count:
                lines.append('Use all question/reference images together in the order provided.')

        if relation_note:
            lines.append(relation_note)

        if option_start is None:
            if option_letters:
                lines.append('The answer options are the text options listed above, not image numbers.')
        else:
            for offset, letter in enumerate(option_letters):
                lines.append(f'Image {option_start + offset + 1} is option {letter}.')
            first_option_image = option_start + 1
            lines.append(
                f'If the correct candidate is Image {first_option_image}, '
                f'answer {option_letters[0]}, not {first_option_image}.'
            )
            lines.append('Answer with option letters only, not image numbers.')

        return '\n'.join(lines) + '\n'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = build_mixedqa_choices(line)
        question_type = self.infer_question_type(line, has_options=bool(options))

        prompt = ''
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if options:
            prompt += 'Options:\n'
            for key, item in options.items():
                prompt += f'{key}. {item}\n'
        prompt += self.image_instruction(tgt_path, options, line=line)
        prompt += self.answer_instruction(question_type, has_options=bool(options))

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def build_judge_model(self, **judge_kwargs):
        model = judge_kwargs.get('model', 'exact_matching')
        supported = ['chatgpt-0125', 'exact_matching', 'gpt-4-0125', 'qwen35-397b', 'deepseek-v4-flash']
        assert model in supported
        name_str_map = {
            'chatgpt-0125': 'openai',
            'gpt-4-0125': 'gpt4',
        }
        name_str = name_str_map.get(model, model)
        if model == 'exact_matching':
            return None, 'exact_matching'

        if model == 'qwen35-397b' and not os.environ.get('DASHSCOPE_API_KEY', None):
            warnings.warn('DASHSCOPE_API_KEY is not set properly, will use exact matching for evaluation')
            return None, name_str
        if model == 'deepseek-v4-flash' and not os.environ.get('DEEPSEEK_API_KEY', None):
            warnings.warn('DEEPSEEK_API_KEY is not set properly, will use exact matching for evaluation')
            return None, name_str
        if model in ['chatgpt-0125', 'gpt-4-0125'] and not gpt_key_set():
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            return None, name_str

        judge = build_judge(**judge_kwargs)
        if not judge.working():
            warnings.warn('Judge API is not working properly, will use exact matching for evaluation')
            warnings.warn(DEBUG_MESSAGE)
            return None, name_str
        return judge, name_str

    def extract_choice_answer(self, model, item, question_type):
        choices = build_mixedqa_choices(item)
        if question_type in ['multiple_choice', 'single_or_multiple_choice']:
            ret = normalize_mixedqa_choice_answer(item['prediction'], choices)
        else:
            ret = can_infer(item['prediction'], choices)
        if ret:
            return ret, item['prediction']
        if model is None:
            return 'Z', 'Failed in Prefetch, no judge-based answer matching under exact_matching policy.'

        prompt = build_mixedqa_extractor_prompt(
            item['question'], build_mixedqa_option_str(choices), item['prediction'], question_type
        )
        for _ in range(3):
            ans = model.generate(prompt)
            if question_type in ['multiple_choice', 'single_or_multiple_choice']:
                ret = normalize_mixedqa_choice_answer(ans, choices)
            else:
                ret = can_infer(ans, choices)
            if ret:
                return ret, ans
        return 'Z', 'Failed to extract a valid option.'

    def eval_item(self, model, item):
        choices = build_mixedqa_choices(item)
        question_type = self.infer_question_type(item, has_options=bool(choices))

        if question_type == 'open':
            pred = normalize_mixedqa_open_answer(item['prediction'])
            gt = normalize_mixedqa_open_answer(item['GT'])
            if pred == gt:
                return dict(hit=1, log=f'Normalized prediction: {pred}; normalized answer: {gt}. ')
            if model is None:
                return dict(hit=0, log=f'Normalized prediction: {pred}; normalized answer: {gt}. ')
            prompt = build_mixedqa_open_judge_prompt(item['question'], item['GT'], item['prediction'])
            judgement = str(model.generate(prompt)).strip()
            hit = int(judgement.lower().startswith('correct'))
            return dict(hit=hit, log=f'Judge: {judgement}. Normalized prediction: {pred}; normalized answer: {gt}. ')

        opt, match_log = self.extract_choice_answer(model, item, question_type)
        if question_type in ['multiple_choice', 'single_or_multiple_choice']:
            gt = normalize_mixedqa_choice_answer(item['GT'], choices)
            pred = normalize_mixedqa_choice_answer(opt, choices)
            hit = int(bool(gt) and gt == pred)
        else:
            hit = int(opt == item['GT'])
        return dict(hit=hit, log=f'Match Log: {match_log}. ')

    def evaluate(self, eval_file, **judge_kwargs):
        model, name_str = self.build_judge_model(**judge_kwargs)
        data = load(eval_file).sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
        data = data[data['index'].isin(answer_map)]
        data['GT'] = [answer_map[idx] for idx in data['index']]

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')
        result = load(result_file) if osp.exists(result_file) else {}
        for _, item in tqdm(data.iterrows(), total=len(data), desc='Mixed QA Evaluation Progress'):
            if item['index'] not in result:
                result[item['index']] = self.eval_item(model, item)
                dump(result, result_file)

        data['hit'] = [result[i]['hit'] for i in data['index']]
        data['log'] = [result[i]['log'] for i in data['index']]
        if 'GT' in data:
            data.pop('GT')

        eval_record = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, eval_record)

        from .utils.multiple_choice import report_acc
        acc = report_acc(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)
        return acc
