import pandas as pd
import argparse      # 用于解析命令行参数
from utils import set_seed  # 设置随机种子确保实验可重现
import numpy as np
import wandb
import os            
import torch         # PyTorch深度学习框架
import torch.nn as nn # PyTorch神经网络模块
from torch.utils.data import DataLoader  # 数据加载器
from torch.nn import functional as F     # PyTorch函数式API
from torch.cuda.amp import GradScaler   # 自动混合精度训练的梯度缩放器

# 模型和训练相关导入
from model import GPT, GPTConfig         # GPT模型和配置类
from trainer import Trainer, TrainerConfig  # 训练器和训练配置类
from dataset import Mol3DDataset, SimpleTokenizer, SubChTokenizer, NewMol3DDataset  # 数据集和分词器类
import math          # 数学函数库
import re            # 正则表达式库

# 分布式训练相关导入
import torch.distributed as dist        # PyTorch分布式训练模块
import torch.multiprocessing as mp      # 多进程处理模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行训练

import time          # 时间处理模块
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录器

def setup(rank, world_size):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的排名(对应GPU编号)
        world_size: 参与训练的总进程数(GPU数量)
    """
    # 设置分布式训练的主节点地址和通信端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    # 初始化进程组，使用NCCL作为后端进行GPU间通信
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 为当前进程指定使用的GPU设备
    torch.cuda.set_device(rank)

def cleanup():
    """销毁分布式进程组，清理分布式训练环境"""
    dist.destroy_process_group()

def load_dataset_from_files(root_path, split, ids):
    """
    从多个分片文件中加载数据集
    
    Args:
        root_path: 数据文件的根路径前缀
        split: 数据集分割类型 ('train', 'val', 'test')
        ids: 要加载的文件分片数量
        
    Returns:
        dataset: 合并后的数据集列表，包含所有文件的行数据
    """
    dataset = []
    for id in range(ids):
        with open(root_path+'_'+split+'_'+str(id)+'.txt', 'r') as file:
            dataset.extend(file.readlines())
            print('loaded dataset from '+root_path+'_'+split+'_'+str(id)+'.txt')
    return dataset

def load_tokenizer(tokenizer_path,max_length, support_rag=False):
    """
    从保存的文件中加载预训练的分词器
    
    Args:
        tokenizer_path: 分词器词汇表文件路径
        max_length: 序列的最大长度限制
        support_rag: 是否支持RAG(检索增强生成)功能
        
    Returns:
        tokenizer: 加载的分词器对象
    """
    tokenizer = SimpleTokenizer(max_length, support_rag=support_rag)  # Update max_length if needed
    tokenizer.load_vocab(tokenizer_path)
    return tokenizer


def run_DDP(rank, world_size, args):
    """
    分布式训练的包装函数，设置环境后调用主训练函数
    
    Args:
        rank: 当前进程排名
        world_size: 总进程数
        args: 训练配置参数
    """
    setup(rank, world_size)    # 设置分布式环境
    run(args, rank)            # 执行主训练流程
    cleanup()                  # 清理分布式环境


def run(args, rank=None):
    """
    主训练函数，包含完整的模型训练流程
    
    Args:
        args: 包含所有训练配置的参数对象
        rank: 当前进程排名(用于分布式训练)
    """
    # ===== 基础设置 =====
    set_seed(45)  # 设置随机种子为45，确保实验结果可重现
    # wandb.init(project="lig_gpt", name=args.run_name)  # 初始化wandb实验跟踪(已注释)
    os.environ["WANDB_MODE"] = "dryrun"  # 设置wandb为离线模式，不上传数据

    print("making tokenizer")

    # ===== 分词器创建和加载 =====
    max_len = args.max_len  # 获取最大序列长度
    print("tokenizer:")
    tokenizer_path = args.output_tokenizer_dir  # 分词器保存目录
    # 如果目录不存在则创建
    if not os.path.isdir(tokenizer_path):
        os.makedirs(tokenizer_path)
    tokenizer_path = args.output_tokenizer_dir + "/vocab.json"  # 词汇表文件完整路径
    print(tokenizer_path)
    
    # 检查是否已存在训练好的分词器
    if os.path.exists(tokenizer_path):
        print(f"The file '{tokenizer_path}' exists.")
        # 加载已有的分词器
        tokenizer = load_tokenizer(tokenizer_path, max_len, support_rag=args.rag)
    else:
        # 创建新的分词器
        tokenizer = SimpleTokenizer(max_length=max_len, support_rag=args.rag)
        # 如果指定使用SubCh分词器，则切换
        if args.tokenizer == 'subch':
            tokenizer = SubChTokenizer(max_length=max_len)

        # 在主要训练和验证数据上训练分词器
        tokenizer.fit_on_file(args.root_path + '.txt')
        tokenizer.fit_on_file(args.root_path + '_val.txt')
        
        # 如果存在条件数据，也在条件数据上训练分词器
        if args.conditions_path is not None:
            tokenizer.fit_on_file(args.conditions_path + '.txt')
            tokenizer.fit_on_file(args.conditions_path + '_val.txt')

        # 如果存在预训练数据，在预训练数据上训练分词器
        if args.pre_root_path is not None:
            tokenizer.fit_on_file(args.pre_root_path + '.txt')
            # 如果预训练验证集不存在，从训练集末尾取100行创建验证集
            if not os.path.exists(args.pre_root_path + '_val.txt'):
                with open(args.pre_root_path + '.txt', 'r') as f:
                    train_seq = f.readlines()
                last_100_lines = train_seq[-100:]
                with open(args.pre_root_path + '_val.txt', 'w') as file:
                    file.writelines(last_100_lines)
            tokenizer.fit_on_file(args.pre_root_path + '_val.txt')
        
        # 如果使用RAG功能，在RAG数据库上训练分词器
        if args.rag_db_path is not None:
            tokenizer.fit_on_lmdb(args.rag_db_path)
            tokenizer.fit_on_lmdb(args.rag_db_val_path)
            tokenizer.fit_on_lmdb(args.rag_db_test_path)
            

        # 保存训练好的分词器词汇表
        tokenizer.save_vocab(tokenizer_path)
        print("tokenizer saved")

    print(tokenizer.get_vocab())  # Print vocabulary - 打印词汇表内容
    vocab_size = tokenizer.get_vocab_size()  # 获取词汇表大小

    print("making dataset")

    # ===== 数据集加载 =====
    # 如果是预训练模式，切换到预训练数据路径
    if args.pretrain:
        # switch to pretrain dataset
        args.root_path = args.pre_root_path
        
    # 加载训练数据
    with open(args.root_path + '.txt', 'r') as file:
        train_data = file.readlines()
    file.close()

    # 加载验证数据
    with open(args.root_path + '_val.txt', 'r') as file:
        val_data = file.readlines()
    file.close()

    # 加载条件数据(如果存在)，用于条件生成任务
    if args.conditions_path is not None:
        print("loading conditions")
        with open(args.conditions_path + '.txt', 'r') as file:
            conditions_data = file.readlines()
        file.close()
        with open(args.conditions_path + '_val.txt', 'r') as file:
            conditions_data_val = file.readlines()
    else:
        conditions_data = None
        conditions_data_val = None
        
    # 加载条件分割ID数据(如果存在)，用于指定条件数据的分割位置
    if args.conditions_split_id_path is not None:
        print("loading conditions split id")
        with open(args.conditions_split_id_path + '.txt', 'r') as file:
            conditions_split_id = file.readlines()
        file.close()
        with open(args.conditions_split_id_path + '_val.txt', 'r') as file:
            conditions_split_id_val = file.readlines()
    else:
        conditions_split_id = None
        conditions_split_id_val = None


    # 根据是否使用ESM蛋白质嵌入选择对应的数据集类
    if args.ESM_protein:
        # 使用包含蛋白质嵌入的数据集类
        train_dataset = NewMol3DDataset(train_data, tokenizer, max_len, conditions_data, conditions_split_id, 
                                        db_path=args.protein_embedding_path)
        valid_dataset = NewMol3DDataset(val_data, tokenizer, max_len, conditions_data_val, conditions_split_id_val, 
                                        db_path=args.protein_embedding_val_path)
    else:
        # 使用标准的分子3D数据集类
        train_dataset = Mol3DDataset(train_data, tokenizer, max_len, conditions_data, conditions_split_id)
        valid_dataset = Mol3DDataset(val_data, tokenizer, max_len, conditions_data_val, conditions_split_id_val)
    
    
    print(f"train dataset size: {len(train_dataset)}")  # 打印训练集大小
    print(f"val dataset size: {len(valid_dataset)}")    # 打印验证集大小

    # 判断是否为条件生成模式
    if args.conditions_path is not None or args.conditions_split_id_path is not None:
        isconditional = True
    else:
        isconditional = False

    # ===== 模型初始化 =====
    print("loading model")
    if args.model == 'gpt':
        # 创建GPT模型配置
        # mconf = GPTConfig(vocab_size, max_len, num_props=args.num_props,  # args.num_props,
        #                   n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold,
        #                   scaffold_maxlen=max_len, lstm=args.lstm, lstm_layers=args.lstm_layers, isconditional=isconditional, 
        #                   mode=args.mode, ESM_protein=args.ESM_protein, rag=args.rag, alpha=args.alpha)
        mconf = GPTConfig(vocab_size, max_len,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,isconditional=isconditional, 
                          mode=args.mode, ESM_protein=args.ESM_protein, rag=args.rag, alpha=args.alpha)
        model = GPT(mconf)  # 创建GPT模型实例
    else:
        raise ValueError("model is not supported")  # 如果指定了不支持的模型类型则报错
        
    # 如果指定了预训练模型路径，加载预训练权重
    if args.pre_model_path is not None:
        print("loading pretrained model: ", args.pre_model_path)
        model_path = args.pre_model_path
        model.load_state_dict(torch.load(model_path), strict=False)  # strict=False允许部分加载
    print('total params:', sum(p.numel() for p in model.parameters()))  # 打印模型总参数量
    
    # ===== 训练环境设置 =====
    # 创建时间戳用于文件夹命名
    cur_time = time.strftime("%Y%m%d_%H%M")
    args.cur_time = cur_time
    checkpoint_folder = os.path.join('./checkpoint/', cur_time + args.run_name)
    # os.makedirs(f'./cond_gpt/weights/', exist_ok=True)
    
    # 创建检查点保存目录(分布式训练时只有rank 0创建)
    if args.dist:
        if rank == 0:
            os.makedirs(f'{checkpoint_folder}', exist_ok=True)
    else:
        os.makedirs(f'{checkpoint_folder}', exist_ok=True)
        
    # 设置检查点文件路径
    checkpoint_path = os.path.join(checkpoint_folder, f'{args.run_name}.pt')          # 常规检查点
    checkpoint_best_path = os.path.join(checkpoint_folder, f'{args.run_name}_best.pt') # 最优检查点
    
    # 创建TensorBoard日志目录
    log_dir = os.path.join('./log/', cur_time + args.run_name)
    if not os.path.exists(log_dir):
        # 分布式训练时只有rank 0创建日志目录
        if args.dist:
            if rank == 0:
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)
        
    else:
        # 如果日志目录已存在，清空其中的文件
        # shutil.rmtree(log_dir)
        for f in os.listdir(log_dir):
            os.remove(os.path.join(log_dir, f))
    
    # 创建TensorBoard写入器用于记录训练过程
    writer = SummaryWriter(log_dir=log_dir)
    
    # ===== 训练器配置 =====
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,  # 学习率预热的token数
                          final_tokens=args.max_epochs * len(train_data) * max_len,       # 总训练token数
                          num_workers=args.num_workers, ckpt_path=checkpoint_path, best_ckpt_path=checkpoint_best_path,
                          run_name=args.run_name, block_size=max_len, generate=False, save_start_epoch=args.save_start_epoch,
                          grad_norm_clip=args.grad_norm_clip,  # 梯度裁剪阈值
                          save_interval_epoch=args.save_interval_epoch, dist=args.dist, rank=rank, 
                          ckpt_folder=checkpoint_folder, tensorboard_writer=writer,
                          resume=args.resume, resume_ckpt_path=args.resume_ckpt_path)  # 恢复训练相关配置
    
    # ===== 开始训练 =====
    trainer = Trainer(model, train_dataset, valid_dataset, tconf)  # , train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)  # 执行训练并返回训练历史数据框
    
    # 保存训练历史到CSV文件(分布式训练时只有rank 0保存)
    if args.dist:
        if rank == 0:
            df.to_csv(f'{args.run_name}.csv', index=False)
    else:
        df.to_csv(f'{args.run_name}.csv', index=False)


if __name__ == '__main__':
    # ===== 命令行参数解析 =====
    parser = argparse.ArgumentParser()

    # 基础运行参数
    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)  # wandb运行名称
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')  # 调试模式开关
    parser.add_argument('--data_name', type=str, default='',
                        help="name of the dataset to train on", required=False)  # 数据集名称
    parser.add_argument('--model', type=str, default='gpt',
                        help="name of the model", required=False)  # 模型类型
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help="name of the tokenizer", required=False)  # 分词器类型
    
    # 模型架构参数
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)  # Transformer层数
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)  # 多头注意力的头数
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)  # 嵌入维度
    
    # 训练超参数
    parser.add_argument('--max_epochs', type=int, default=60,
                        help="total epochs", required=False)  # 最大训练轮数
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size", required=False)  # 批大小
    parser.add_argument('--num_workers', type=int, default=12,
                        help="number of workers for data loaders", required=False)  # 数据加载工作进程数
    parser.add_argument('--save_start_epoch', type=int, default=120,
                        help="save model start epoch", required=False)  # 开始保存模型的轮数
    parser.add_argument('--save_interval_epoch', type=int, default=10,
                        help="save model epoch interval", required=False)  # 模型保存间隔轮数
    parser.add_argument('--learning_rate', type=float,
                        default=4e-4, help="learning rate", required=False)  # 学习率
    parser.add_argument('--max_len', type=int, default=512,
                        help="max_len", required=False)  # 最大序列长度
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                        help="gradient norm clipping. smaller values mean stronger normalization.", required=False)  # 梯度裁剪阈值
    
    # 混合精度训练参数
    parser.add_argument('--auto_fp16to32', action='store_true',
                        default=False, help='Auto casting fp16 tensors to fp32 when necessary')  # 自动类型转换
    
    # 数据路径参数
    parser.add_argument('--pre_root_path', default=None,
                        help="Path to the pretrain data directory", required=False)  # 预训练数据目录
    parser.add_argument('--pre_model_path', default=None,
                        help="Path to the pretrain model", required=False)  # 预训练模型路径
    parser.add_argument('--root_path', default='',
                        help="Path to the root data directory", required=False)  # 主数据目录
    parser.add_argument('--output_tokenizer_dir', default='',
                        help="Path to the saved tokenizer directory", required=False)  # 分词器保存目录
    parser.add_argument('--conditions_path', default=None,
                        help="Path to the generation condition", required=False)  # 条件生成数据路径
    parser.add_argument('--conditions_split_id_path', default=None,
                        help="Path to the conditions_split_id", required=False)  # 条件分割ID路径
    
    # 分布式训练参数
    parser.add_argument('--dist', action='store_true',
                        default=False, help='use torch.distributed to train the model in parallel')  # 启用分布式训练
    
    # 蛋白质嵌入相关参数
    parser.add_argument('--protein_embedding_path', default='',
                    help="Path to the train database of pre-calculated protein embedding", required=False)  # 训练用蛋白质嵌入数据库
    parser.add_argument('--protein_embedding_val_path', default='',
                    help="Path to the validation database of pre-calculated protein embedding", required=False)  # 验证用蛋白质嵌入数据库
    parser.add_argument('--mode', default='concat',
                    help="mode to incorporate protein embedding, ['concat', 'cross']", required=False)  # 蛋白质嵌入融合模式
    parser.add_argument('--ESM_protein', action='store_true',
                        default=False, help='use ESM protein embedding')  # 使用ESM蛋白质嵌入
    
    # 预训练相关参数
    parser.add_argument('--pretrain', action='store_true',
                        default=False, help='whether to pretrain the model')  # 是否进行预训练
    
    # RAG(检索增强生成)相关参数
    parser.add_argument('--rag_db_path', type=str, default=None,
                        help='path to the train rag database')  # 训练用RAG数据库路径
    parser.add_argument('--rag_db_val_path', type=str, default=None,
                        help='path to the validation rag database')  # 验证用RAG数据库路径
    parser.add_argument('--rag_db_test_path', type=str, default=None,
                        help='path to the test rag database')  # 测试用RAG数据库路径
    parser.add_argument('--rag', action='store_true',
                        default=False, help='whether to use rag')  # 是否使用RAG功能
    
    # 训练恢复相关参数
    parser.add_argument('--resume', action='store_true',
                        default=False, help='whether to resume training')  # 是否恢复训练
    parser.add_argument('--resume_ckpt_path', type=str, default=None,
                    help='path to the checkpoint for resume training')  # 恢复训练的检查点路径
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of loss for retrieved part')  # 检索部分损失权重

    args = parser.parse_args()  # 解析命令行参数

    # ===== 启动训练流程 =====
    if args.dist:
        # 分布式训练：获取可用GPU数量并启动多进程训练
        world_size = torch.cuda.device_count()
        # import pdb; pdb.set_trace()
        mp.spawn(run_DDP,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        # 单机训练：直接调用训练函数
        run(args)
