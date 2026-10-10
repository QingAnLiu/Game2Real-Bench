import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def split_thinking_response(model, response):
    """Split a tagged reasoning trace while retaining the complete raw response elsewhere."""
    text = str(response)

    def default_split(s):
        if '</think>' in s:
            splits = s.split('</think>')
            prediction = splits[-1].strip()
            if len(splits) == 2 and '<think>' in splits[0]:
                thinking = splits[0].split('<think>')[1].strip()
            else:
                thinking = '</think>'.join(splits[:-1])
                thinking += '</think>'
                warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
        else:
            thinking = ''
            prediction = s
        return prediction, thinking

    split_func = model.split_thinking if hasattr(model, 'split_thinking') else default_split
    return split_func(text)


def use_model_custom_prompt(model, dataset, dataset_name):
    if getattr(dataset, 'TYPE', None) == 'MixQA':
        return False
    return hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if use_model_custom_prompt(model, dataset, dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(
    model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4,
    use_vllm=False, batch_size=1
):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    # 统计推理时间的字典
    timing_by_index = {}  # {idx: seconds}
    fail_count = 0

    # vLLM gets most of its throughput benefit when multiple requests are
    # submitted together.  Historically VLMEvalKit explicitly disabled the
    # generate_batch path for use_vllm models, which reduced vLLM to a
    # single-request loop.  Prefer each wrapper's advertised vLLM batch size
    # when the caller leaves batch_size at its legacy default of 1.  The value
    # can always be overridden without touching code:
    #   export VLMEVAL_VLLM_BATCH_SIZE=8
    is_vllm = bool(getattr(model, 'use_vllm', False))
    if is_vllm:
        env_batch_size = os.environ.get('VLMEVAL_VLLM_BATCH_SIZE')
        if env_batch_size is not None:
            try:
                batch_size = max(1, int(env_batch_size))
            except ValueError:
                warnings.warn(
                    f'Invalid VLMEVAL_VLLM_BATCH_SIZE={env_batch_size!r}; '
                    f'using batch_size={batch_size}.'
                )
        elif batch_size <= 1:
            batch_size = max(1, int(getattr(model, 'vllm_batch_size', 1)))

    use_real_batch = (
        batch_size > 1
        and callable(getattr(model, 'generate_batch', None))
    )
    if batch_size > 1 and not use_real_batch:
        warnings.warn(
            f'{model_name} does not implement generate_batch; '
            'falling back to the legacy single-sample path.'
        )
    if is_vllm and use_real_batch:
        print(
            f'[vLLM] {model_name}: request batch_size={batch_size}',
            flush=True,
        )
    step_size = batch_size if use_real_batch else 1

    for i in tqdm(range(0, lt, step_size), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        if use_real_batch:
            batch_data = data.iloc[i: i + batch_size]
            batch_indices = list(batch_data['index'])
            batch_structs = []
            for _, line in batch_data.iterrows():
                if use_model_custom_prompt(model, dataset, dataset_name):
                    batch_structs.append(model.build_prompt(line, dataset=dataset_name))
                else:
                    batch_structs.append(dataset.build_prompt(line))

            if torch.cuda.is_available() and not is_vllm:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if os.environ.get('SKIP_ERR', False) == '1':
                fail_msg = 'Failed to obtain answer'
                try:
                    responses = model.generate_batch(batch_structs, dataset=dataset_name)
                    if len(responses) != len(batch_indices):
                        raise RuntimeError(
                            f'generate_batch returned {len(responses)} responses for '
                            f'{len(batch_indices)} inputs.'
                        )
                except RuntimeError as err:
                    warnings.warn(f'{type(err)} {str(err)}')
                    responses = [f'{fail_msg}: {type(err)} {str(err)}'] * len(batch_indices)
                    fail_count += len(batch_indices)
            else:
                responses = model.generate_batch(batch_structs, dataset=dataset_name)
                if len(responses) != len(batch_indices):
                    raise RuntimeError(
                        f'generate_batch returned {len(responses)} responses for '
                        f'{len(batch_indices)} inputs.'
                    )

            if torch.cuda.is_available() and not is_vllm:
                torch.cuda.synchronize()
            elapsed_per_sample = (time.perf_counter() - t0) / len(batch_indices)
            for idx, response in zip(batch_indices, responses):
                res[idx] = response
                timing_by_index[idx] = elapsed_per_sample
            if torch.cuda.is_available() and not is_vllm:
                torch.cuda.empty_cache()
            if verbose:
                for response in responses:
                    print(response, flush=True)
            progress = min(i + len(batch_indices), lt)
            if progress % 10 == 0 or progress == lt:
                dump(res, out_file)
            continue

        # Legacy single-sample path (preserved for batch_size=1 and unsupported models).
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if use_model_custom_prompt(model, dataset, dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # ==== 计时开始（NEW）====
        if torch.cuda.is_available() and not is_vllm:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                if torch.cuda.is_available() and not is_vllm:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                elapsed = t1 - t0
                timing_by_index[idx] = elapsed
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
                fail_count += 1
            else:
                if torch.cuda.is_available() and not is_vllm:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                elapsed = t1 - t0
                timing_by_index[idx] = elapsed
        else:
            response = model.generate(message=struct, dataset=dataset_name)
            if torch.cuda.is_available() and not is_vllm:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            timing_by_index[idx] = elapsed

        # ==== 计时结束（NEW）====
        if torch.cuda.is_available() and not is_vllm:
            torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)
            vals = list(timing_by_index.values())
            if vals:
                arr = np.array(vals, dtype=float)
                avg = float(arr.mean())
                p50 = float(np.percentile(arr, 50))
                p90 = float(np.percentile(arr, 90))
                p95 = float(np.percentile(arr, 95))
                p99 = float(np.percentile(arr, 99))
                thr = len(vals) / arr.sum() if arr.sum() > 0 else 0.0
                print(
                    f"[Rank {rank}] progress={i + 1}/{lt} | "
                    f"avg={avg:.3f}s p50={p50:.3f}s p90={p90:.3f}s p95={p95:.3f}s p99={p99:.3f}s | "
                    f"thr={thr:.2f} it/s | fails={fail_count}",
                    flush=True
                )
    # ------- 收尾：最终统计 + 写 timing 文件 -------
    vals = list(timing_by_index.values())
    if vals:
        arr = np.array(vals, dtype=float)
        total_time = float(arr.sum())
        summary = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "rank": int(rank),
            "world_size": int(world_size),
            "num_samples": int(len(vals)),
            "fail_count": int(fail_count),
            "total_time_sec": total_time,
            "throughput_it_per_sec": (len(vals) / total_time) if total_time > 0 else 0.0,
            "avg_sec": float(arr.mean()),
            "p50_sec": float(np.percentile(arr, 50)),
            "p90_sec": float(np.percentile(arr, 90)),
            "p95_sec": float(np.percentile(arr, 95)),
            "p99_sec": float(np.percentile(arr, 99)),
        }
    else:
        summary = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "rank": int(rank),
            "world_size": int(world_size),
            "num_samples": 0,
            "fail_count": int(fail_count)
        }
    timing_path = f"{os.path.splitext(out_file)[0]}.timing.rank{rank}.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Rank {rank}] timing summary written to: {timing_path}", flush=True)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False,
    use_vllm=False, batch_size=1
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc,
        use_vllm=use_vllm, batch_size=batch_size)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            raw_prediction = [str(data_all[x]) for x in data['index']]
            print(f'Prediction format: {os.getenv("SPLIT_THINK")}, preserving raw prediction and splitting thinking.')
            tups = [split_thinking_response(model, x) for x in raw_prediction]
            data['raw_prediction'] = raw_prediction
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
