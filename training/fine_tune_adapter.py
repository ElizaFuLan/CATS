import os, torch, shutil, glob, time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_home", type=str, default=None)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=20)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-5)  # smaller lr for fine-tuning
parser.add_argument("--exit_layer", type=int, default=3)
parser.add_argument("--topk", type=int, default=20,
                    help='Top-K focused KL loss for train_finetune.py. 0 = full-vocab KL.')
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()
if args.data_home is None:
    args.data_home = f"/blue/sun.jingwei/yuninghan/Yuning_Han/Kangaroo/Data/sharegpt_0_67999_mufp16_layer{args.exit_layer}"
print(args)

work_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(work_dir)

def data_moxing(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)

LOCAL_TRAIN_DATA = args.data_home
LOCAL_CKPT_DATA = f"/blue/sun.jingwei/yuninghan/Yuning_Han/Kangaroo/cache/CKPT/FineTuneAdapter_Layer{args.exit_layer}_topk{args.topk}/"

MASTER_ADDR = "172.16.2.146"
MASTER_PORT = 62275
rank = "0"
nnodes = 1
processes = int(nnodes) * torch.cuda.device_count()

# for epoch in range(args.start_epoch, args.end_epoch):
#     print("start epoch: ", epoch)
#     dst_dir = LOCAL_TRAIN_DATA

#     command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch --multi_gpu \
#         --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
#             --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 train.py \
#                 --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
#                     --basepath /cache/CKPT/vicuna-7b-v1.3/ --configpath ./data/vicuna_7B_config.json \
#                         --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer}"

#     print(command)
#     os.system(command)

# for epoch in range(args.start_epoch, args.end_epoch):
#     print("start epoch: ", epoch)
#     dst_dir = LOCAL_TRAIN_DATA
    
#     # 单GPU训练配置
#     processes = 1
#     command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch \
#         --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
#             --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 train.py \
#                 --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
#                     --basepath lmsys/vicuna-7b-v1.3 --configpath ./data/vicuna_7B_config.json \
#                         --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer}"
    
#     print(command)
#     os.system(command)

# 检测可用GPU数量并设置为双GPU训练
available_gpus = torch.cuda.device_count()
print(f"检测到 {available_gpus} 个可用GPU")

if available_gpus < 2:
    print(f"警告: 只检测到 {available_gpus} 个GPU，但脚本配置为使用2个GPU")
    print("建议检查GPU可用性或修改为单GPU训练")

# 双GPU训练配置
processes = 2

# # 四GPU训练配置
# processes = 4

# # 单GPU训练配置
# processes = 1

# 确保checkpoints目录存在
os.makedirs(LOCAL_CKPT_DATA, exist_ok=True)

for epoch in range(args.start_epoch, args.end_epoch):
    print("start epoch: ", epoch)
    dst_dir = LOCAL_TRAIN_DATA
    
    # 双GPU训练命令（使用 train_finetune.py，含 Top-K loss）
    command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch --multi_gpu \
        --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
            --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 train_finetune.py \
                --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
                    --basepath lmsys/vicuna-7b-v1.3 --configpath ./data/vicuna_7B_config.json \
                        --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer} --topk {args.topk}"
    
    print(command)
    os.system(command)

    # # 四GPU训练命令
    # command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch --multi_gpu \
    #     --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
    #         --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 train.py \
    #             --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
    #                 --basepath lmsys/vicuna-7b-v1.3 --configpath ./data/vicuna_7B_config.json \
    #                     --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer}"
    
    # print(command)
    # os.system(command)

    # # 单GPU训练命令
    # command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch \
    #     --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
    #         --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 train.py \
    #             --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
    #                 --basepath lmsys/vicuna-7b-v1.3 --configpath ./data/vicuna_7B_config.json \
    #                     --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer}"
    
    # print(command)
    # os.system(command)
    
    
# # 训练完成后，复制最终的adapter到包含exit_layer信息的路径
# print("Training completed. Copying final adapter...")
# final_epoch = args.end_epoch - 1
# source_path = f"{LOCAL_CKPT_DATA}/state_{final_epoch}/"

# # 动态生成包含exit_layer的模型名
# target_path = f"/blue/sun.jingwei/yuninghan/Yuning_Han/Kangaroo/cache/CKPT/kangaroo-vicuna-7b-v1.3-layer{args.exit_layer}/"

# if os.path.exists(source_path):
#     # 创建目标目录
#     os.makedirs(target_path, exist_ok=True)
    
#     # 加载safetensors文件并转换为pytorch_model.bin
#     from safetensors.torch import load_file
#     import glob
    
#     # 查找所有safetensors文件
#     safetensors_files = glob.glob(os.path.join(source_path, "*.safetensors"))
#     if safetensors_files:
#         print("Converting safetensors to pytorch_model.bin...")
        
#         # 合并所有safetensors文件
#         combined_weights = {}
#         for file_path in sorted(safetensors_files):
#             weights = load_file(file_path)
#             combined_weights.update(weights)
#             print(f"Loaded weights from {os.path.basename(file_path)}")
        
#         # 保存为pytorch_model.bin
#         pytorch_model_path = os.path.join(target_path, "pytorch_model.bin")
#         torch.save(combined_weights, pytorch_model_path)
#         print(f"Saved combined weights to {pytorch_model_path}")
    
#     # 复制config文件（如果存在的话）
#     config_files = ["config.json", "vicuna_7B_config.json"]
#     for config_file in config_files:
#         if os.path.exists(os.path.join("./data", config_file)):
#             shutil.copy2(os.path.join("./data", config_file), os.path.join(target_path, "config.json"))
#             print(f"Copied config file: {config_file}")
#             break
#     else:
#         print("Warning: No config file found to copy")
    
#     # 复制其他非safetensors文件（如果需要的话）
#     for file in os.listdir(source_path):
#         if not file.endswith('.safetensors'):
#             src_file = os.path.join(source_path, file)
#             dst_file = os.path.join(target_path, file)
#             if os.path.isfile(src_file):
#                 shutil.copy2(src_file, dst_file)
    
#     print(f"Final adapter copied from {source_path} to {target_path}")
#     print(f"Model saved with exit layer {args.exit_layer} in the path name")
#     print(f"Model weights saved as pytorch_model.bin format")
# else:
#     print(f"Warning: Source path {source_path} does not exist")

# 训练完成后，复制最终的adapter到包含exit_layer信息的路径
print("Training completed. Copying final adapter...")
final_epoch = args.end_epoch - 1
source_path = f"{LOCAL_CKPT_DATA}/state_{final_epoch}/"

# 动态生成包含exit_layer和topk信息的模型名
target_path = f"/blue/sun.jingwei/yuninghan/Yuning_Han/Kangaroo/cache/CKPT/kangaroo-vicuna-7b-v1.3-layer{args.exit_layer}-topk{args.topk}/"

if os.path.exists(source_path):
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    # 直接复制整个源目录的所有内容到目标目录
    print(f"Copying all contents from {source_path} to {target_path}")
    
    # 使用 shutil.copytree 复制整个目录内容
    try:
        # 如果目标目录已存在，先清空它
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        
        # 复制整个目录
        shutil.copytree(source_path, target_path)
        print(f"Successfully copied all files from {source_path} to {target_path}")
        
    except Exception as e:
        print(f"Error during directory copy: {e}")
        # 备选方案：逐个文件复制
        print("Attempting file-by-file copy...")
        os.makedirs(target_path, exist_ok=True)
        
        for item in os.listdir(source_path):
            src_item = os.path.join(source_path, item)
            dst_item = os.path.join(target_path, item)
            
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                print(f"Copied directory: {item}")
            else:
                shutil.copy2(src_item, dst_item)
                print(f"Copied file: {item}")
    
    # 可选：复制配置文件到目标目录（如果需要的话）
    config_files = ["config.json", "vicuna_7B_config.json"]
    for config_file in config_files:
        config_source = os.path.join("./data", config_file)
        if os.path.exists(config_source):
            config_target = os.path.join(target_path, "config.json")
            if not os.path.exists(config_target):  # 只在目标不存在时复制
                shutil.copy2(config_source, config_target)
                print(f"Copied config file: {config_file}")
            break
    
    print(f"Final adapter directory copied from {source_path} to {target_path}")
    print(f"Model saved with exit layer {args.exit_layer} in the path name")
    print(f"All original files preserved in their original format")
    
else:
    print(f"Warning: Source path {source_path} does not exist")

# 训练完成后绘制 loss 曲线
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

    # 计算每个 epoch 的步数用于绘制分隔线
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
            # 添加 epoch 分隔线
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