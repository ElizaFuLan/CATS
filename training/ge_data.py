import argparse
from tqdm import tqdm
import sys

sys.stdout.flush()
sys.stderr.flush()

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()

print(f"[INFO] Arguments: {args}", flush=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
print(f"[INFO] GPU set to: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

print("[INFO] Importing PyTorch...", flush=True)
import torch
import torch.nn.functional as F
print(f"[INFO] PyTorch {torch.__version__} imported successfully", flush=True)

print("[INFO] Importing transformers...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
print("[INFO] Transformer imported", flush=True)
print("[INFO] Importing datasets...", flush=True)
from datasets import load_dataset
print("[INFO] Datasets imported", flush=True)
print("[INFO] Importing model_adapter...", flush=True)
from fastchat.model.model_adapter import get_conversation_template
print("[INFO] Libraries imported", flush=True)

bigname = "lmsys/vicuna-7b-v1.3"

def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))
    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break
    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length

def build_dataset_rank(
        tokenizer, split="train",
        select=None,
        data_path = "your_path/ShareGPT_V4.3_unfiltered_cleaned_split.json"
):
    print(f"[INFO] Loading dataset from {data_path}...", flush=True)
    print("[INFO] This may take 5-10 minutes for the first time...", flush=True)
    ds = load_dataset('json', data_files=data_path)
    print("[INFO] Dataset loaded successfully!", flush=True)
    
    ds = ds['train']
    print(f"[INFO] Total samples: {len(ds)}", flush=True)
    
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    print(f"[INFO] Selected samples from {args.start} to {args.end}", flush=True)
    
    original_columns1 = ds1.column_names
    num_proc = 1
    
    print("[INFO] Starting data preprocessing...", flush=True)

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            conv = get_conversation_template("vicuna")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)

            sep = conv.sep + conv.roles[1] + ": "
            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    instruction_len -= 1

                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    cur_len -= 1

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )
    
    print("[INFO] Preprocessing completed!", flush=True)
    ds1.set_format(type="torch")
    return ds1

print("[INFO] Loading tokenizer...", flush=True)
bigtokenizer = AutoTokenizer.from_pretrained(bigname, use_fast=False)
print("[INFO] Tokenizer loaded", flush=True)

ds = build_dataset_rank(bigtokenizer)
print(f"[INFO] Dataset ready: {ds}", flush=True)

print("[INFO] Loading model... (this may take 1-2 minutes)", flush=True)
bigmodel = AutoModelForCausalLM.from_pretrained(bigname, device_map="auto", torch_dtype=torch.float16)
print("[INFO] Model loaded successfully!", flush=True)
bigmodel.eval()

@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    assert len(outs_big.hidden_states) == 33

    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp =probs[0].max(dim=1).values
    td={"input_ids":input_ids.cpu()[0],
        "loss_mask":data["loss_mask"].cpu()[0]}
    td[f"hidden_state_layer15"] = outs_big.hidden_states[15].cpu()[0]
    td[f"hidden_state_layer16"] = outs_big.hidden_states[16].cpu()[0]
    td[f"hidden_state"] = outs_big.hidden_states[-1].cpu()[0]
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)
print(f"[INFO] Output directory: {outdir}", flush=True)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

print("[INFO] Starting data processing...", flush=True)
for data in tqdm(ds):
    outdata = ge(data)
    writedata(outdir,outdata)

print("[INFO] All done!", flush=True)