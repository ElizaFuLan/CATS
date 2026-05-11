import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = get_conversation_template("vicuna")
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            total_time = 0.0
            output = "ERROR"
            idx = 0
            new_token = 0

            try:
                torch.cuda.synchronize()
                start_time = time.time()
                result = forward_func(inputs, model, tokenizer, max_new_tokens, **kwargs)

                if len(result) == 5:
                    output_ids, new_token, idx, accept_length_tree, draft_log = result
                else:
                    output_ids, new_token, idx, accept_length_tree = result

                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]

                if conv.stop_token_ids:
                    stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except Exception as e:
                print(f"ERROR question ID: {question['question_id']}, Error: {e}")
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    accept_lengths_tree = []
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            all_draft_logs = []

            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids

                total_time = 0.0
                output = "ERROR"
                idx = 0
                new_token = 0
                draft_log = []
                accept_length_tree = []

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()

                    result = forward_func(inputs, model, tokenizer, max_new_tokens, **kwargs)

                    if len(result) == 5:
                        output_ids, new_token, idx, accept_length_tree, draft_log = result
                    else:
                        output_ids, new_token, idx, accept_length_tree = result
                        draft_log = []
                    if output_ids is None:
                        raise ValueError("forward_func returned None as output_ids")

                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()

                    all_draft_logs.append({"turn": j, "draft_verification": draft_log})

                except Exception as e:
                    print(f"ERROR question ID: {question['question_id']}, Error: {e}")
                    output = "ERROR"
                    all_draft_logs.append({"turn": j, "draft_verification": [], "error": str(e)})
                    accept_length_tree = []

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                cur_accept_lengths_tree.extend(accept_length_tree)
                conv.messages[-1][-1] = output

            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
                "accept_lengths": cur_accept_lengths_tree,
                "draft_verification_log": all_draft_logs
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            # fout.write(json.dumps(ans_json) + "\n")
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
