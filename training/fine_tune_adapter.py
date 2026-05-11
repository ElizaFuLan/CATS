import os, torch, shutil, glob, time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_home", type=str, default=None)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=20)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--exit_layer", type=int, default=3)
parser.add_argument("--topk", type=int, default=20,
                    help='Top-K focused KL loss for train_finetune.py. 0 = full-vocab KL.')
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()
if args.data_home is None:
    args.data_home = f"your_path/sharegpt_0_67999_mufp16_layer{args.exit_layer}"
print(args)

work_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(work_dir)

def data_moxing(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)

LOCAL_TRAIN_DATA = args.data_home
LOCAL_CKPT_DATA = f"your_path/FineTuneAdapter_Layer{args.exit_layer}_topk{args.topk}/"

MASTER_ADDR = "172.16.2.146"
MASTER_PORT = 62275
rank = "0"
nnodes = 1
processes = int(nnodes) * torch.cuda.device_count()

available_gpus = torch.cuda.device_count()
print(f"Detected {available_gpus} available GPU(s)")

if available_gpus < 2:
    print(f"Warning: only {available_gpus} GPU detected, script is configured for 2 GPUs")
    print("Consider checking GPU availability or switching to single-GPU training")

processes = 2

os.makedirs(LOCAL_CKPT_DATA, exist_ok=True)

for epoch in range(args.start_epoch, args.end_epoch):
    print("start epoch: ", epoch)
    dst_dir = LOCAL_TRAIN_DATA

    command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch --multi_gpu \
        --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
            --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 train_finetune.py \
                --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
                    --basepath lmsys/vicuna-7b-v1.3 --configpath ./data/vicuna_7B_config.json \
                        --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer} --topk {args.topk}"

    print(command)
    os.system(command)

print("Plotting loss curves...")
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tb_dir = os.path.join(LOCAL_CKPT_DATA, "tensorboard")
    ea = EventAccumulator(tb_dir)
    ea.Reload()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves (exit_layer={args.exit_layer})", fontsize=14)

    plot_items = [
        ("train/loss",         axes[0, 0], "Train Loss",         "Loss"),
        ("train/prob_accept",  axes[0, 1], "Accept Probability", "Prob"),
        ("train/lr",           axes[1, 0], "Learning Rate",      "LR"),
        ("train/accuracy",     axes[1, 1], "Accuracy",           "Acc"),
    ]

    steps_per_epoch = None
    if "train/loss" in ea.Tags().get("scalars", []):
        loss_steps = [e.step for e in ea.Scalars("train/loss")]
        if loss_steps and args.end_epoch > 1:
            steps_per_epoch = max(loss_steps) // (args.end_epoch - 1)

    for tag, ax, title, ylabel in plot_items:
        if tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps  = [e.step  for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values)
            if steps_per_epoch:
                for ep in range(1, args.end_epoch):
                    ax.axvline(x=ep * steps_per_epoch, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
                    ax.text(ep * steps_per_epoch, ax.get_ylim()[1], f'E{ep}',
                            color='red', fontsize=6, ha='center', va='top')
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True)
        else:
            ax.set_title(f"{title} (no data)")

    plt.tight_layout()
    save_path = os.path.join(LOCAL_CKPT_DATA, f"loss_curves_layer{args.exit_layer}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {save_path}")

except Exception as e:
    print(f"Warning: Failed to plot loss curves: {e}")