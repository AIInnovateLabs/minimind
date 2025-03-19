"""
直接偏好优化(DPO)训练脚本
此脚本实现了基于人类偏好的直接优化训练，通过成对的偏好数据来优化模型的输出分布
主要特点：
1. 使用人类偏好数据对（chosen和rejected）进行训练
2. 采用参考模型（固定权重）来计算相对概率比
3. 通过DPO损失函数直接优化模型对齐人类偏好
4. 使用较小的学习率（1e-8）防止灾难性遗忘
"""

import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import DPODataset

warnings.filterwarnings('ignore')


def Logger(content):
    """
    日志打印函数，在分布式训练时只在主进程上打印
    Args:
        content: 需要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    获取当前学习率，使用余弦退火策略
    Args:
        current_step: 当前步数
        total_steps: 总步数
        lr: 初始学习率
    Returns:
        当前步数对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    """
    将模型输出的logits转换为对应标签的概率
    Args:
        logits: 模型输出的logits，形状为(batch_size, seq_len, vocab_size)
        labels: 真实标签，形状为(batch_size, seq_len)
    Returns:
        对应标签位置的概率，形状为(batch_size, seq_len)
    """
    # 计算每个位置的log概率分布
    log_probs = F.log_softmax(logits, dim=2)
    # 获取标签对应位置的概率
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs, probs, beta):
    """
    计算DPO（Direct Preference Optimization）损失
    Args:
        ref_probs: 参考模型（固定权重）的输出概率，形状为(batch_size, seq_len)
        probs: 训练模型的输出概率，形状为(batch_size, seq_len)
        beta: 温度参数，用于控制损失的尺度
    Returns:
        DPO损失值
    """
    # 计算每个样本的平均概率（序列级别）
    ref_probs = ref_probs.mean(dim=1)
    probs = probs.mean(dim=1)

    # 将chosen和rejected数据分开
    # 数据的前半部分是chosen样本，后半部分是rejected样本
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]  # 参考模型对chosen样本的概率
    reject_ref_probs = ref_probs[batch_size // 2:]  # 参考模型对rejected样本的概率
    chosen_probs = probs[:batch_size // 2]          # 训练模型对chosen样本的概率
    reject_probs = probs[batch_size // 2:]          # 训练模型对rejected样本的概率

    # 计算概率比值的对数差
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    # 计算最终的logits
    logits = pi_logratios - ref_logratios
    # 使用logsigmoid计算损失
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, wandb):
    """
    训练一个epoch
    Args:
        epoch: 当前epoch数
        wandb: wandb日志工具对象
    """
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        # 准备输入数据
        # 每个batch包含chosen和rejected两组数据
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        
        # 将chosen和rejected数据拼接在一起处理
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播和损失计算
        with ctx:
            # 首先用参考模型（固定权重）计算概率
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask  # 应用mask，忽略padding位置
            
            # 然后用训练模型计算概率
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask  # 应用mask，忽略padding位置
            
            # 计算DPO损失
            loss = dpo_loss(ref_probs, probs, beta=0.1)
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积和更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 打印训练日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录wandb日志
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """
    初始化模型
    Args:
        lm_config: 模型配置参数
    Returns:
        model: 需要训练的模型
        ref_model: 参考模型（固定权重）
        tokenizer: 分词器
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    # 初始化训练模型
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    
    # 初始化参考模型（加载相同的权重但固定参数）
    ref_model = MiniMindLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练环境
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=3000, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl")

    args = parser.parse_args()

    # 初始化模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置随机种子和设备类型
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称
    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 初始化分布式训练环境
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型、参考模型和tokenizer
    model, ref_model, tokenizer = init_model(lm_config)

    # 初始化数据加载器
    train_ds = DPODataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化优化器和梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 设置分布式训练参数
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
