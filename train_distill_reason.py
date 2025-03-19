"""
推理蒸馏训练脚本
此脚本实现了针对模型推理能力的蒸馏训练，通过特殊的标记和损失函数设计，增强模型的推理过程表达
主要特点：
1. 使用<think>和</think>标记来标识推理过程
2. 使用<answer>和</answer>标记来标识最终答案
3. 对推理过程相关的token赋予更高的训练权重
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
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """
    日志打印函数
    在分布式训练时只在主进程中打印日志
    
    参数:
        content: 需要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算当前学习率
    使用余弦退火策略动态调整学习率
    
    参数:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率
    返回:
        当前步数对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    训练一个epoch
    实现了推理蒸馏的核心训练逻辑
    
    参数:
        epoch: 当前epoch编号
        wandb: wandb日志工具实例
    """
    # 定义推理过程和答案的特殊标记token
    start_of_think_ids = tokenizer('<think>').input_ids  # 推理开始标记
    end_of_think_ids = tokenizer('</think>').input_ids   # 推理结束标记
    start_of_answer_ids = tokenizer('<answer>').input_ids  # 答案开始标记
    end_of_answer_ids = tokenizer('</answer>').input_ids   # 答案结束标记
    
    # 使用交叉熵作为基础损失函数
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据准备
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码
        
        # 计算当前步的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播和损失计算
        with ctx:
            res = model(X)
            # 计算基础交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            
            # 识别特殊标记位置
            sp_ids = torch.isin(Y.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            
            # 对特殊标记位置增加10倍的损失权重
            loss_mask = loss_mask.view(-1)
            loss_mask_sum = loss_mask.sum()
            loss_mask[sp_ids] = 10  # 增加推理过程相关token的权重
            loss_mask = loss_mask.view(Y.size())
            
            # 计算加权后的最终损失
            loss = (loss * loss_mask).sum() / loss_mask_sum
            loss += res.aux_loss  # 添加辅助损失
            loss = loss / args.accumulation_steps  # 梯度累积

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志记录
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

            # wandb日志记录
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 模型保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/reason_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """
    初始化模型和分词器
    
    参数:
        lm_config: 语言模型配置
    返回:
        model: 初始化好的模型
        tokenizer: 分词器
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/rlhf_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练环境
    设置分布式训练相关的参数
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
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind Distill Reasoning")
    parser.add_argument("--out_dir", type=str, default="out")  # 输出目录
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=8)  # 批次大小
    parser.add_argument("--learning_rate", type=float, default=1e-6)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用wandb记录
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")  # wandb项目名
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载线程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=1)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热迭代次数
    parser.add_argument("--log_interval", type=int, default=1)  # 日志记录间隔
    parser.add_argument("--save_interval", type=int, default=50)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地进程序号
    parser.add_argument('--dim', default=512, type=int)  # 模型维度
    parser.add_argument('--n_layers', default=8, type=int)  # 模型层数
    parser.add_argument('--max_seq_len', default=1024, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE
    parser.add_argument("--data_path", type=str, default="./dataset/r1_mix_1024.jsonl")  # 数据路径

    args = parser.parse_args()

    # 模型配置初始化
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 训练参数设置
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)  # 设置随机种子
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb运行名称设置
    args.wandb_run_name = f"MiniMind-Distill-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 分布式训练设置
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # wandb初始化
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 模型和分词器初始化
    model, tokenizer = init_model(lm_config)

    # 数据加载器设置
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
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

    # 优化器和混合精度训练设置
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 分布式模型包装
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
