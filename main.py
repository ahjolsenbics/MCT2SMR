import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.UNet3D import UNet3D
from modules.loss import compute_loss
from models.ct2rep import MCT2SMR
from modules.data_ct import MCT2SMRDataset
import torch.distributed as dist
from tqdm import tqdm
from modules.tools import *
from modules.logger import *
from config import parse_args
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():

    args = parse_args()
    args.save_dir = create_paths(args.save_dir)
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    args.logger = Logger(f"{args.save_dir}/log/train_log.txt").get_logger()
    show_args(args, logger=args.logger)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer(args)
    unique_idx = set()
    unique_idx.add(args.bos_idx)
    unique_idx.add(args.eos_idx)
    unique_idx.add(args.pad_idx)
    args.total_vocab_size = len(tokenizer.idx2token) + len(unique_idx)

    model = MCT2SMR(args, tokenizer)
    model.to(args.device)

    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    train_ds = MCT2SMRDataset(args, data_folder=args.trainfolder, xlsx_file=args.trainxlsxfile, tokenizer=tokenizer,  num_frames=2, mode='train')
    valid_ds = MCT2SMRDataset(args, data_folder=args.validfolder, xlsx_file=args.validxlsxfile, tokenizer=tokenizer,  num_frames=2, mode='val')

    train_dataloader = R2DataLoader(args, train_ds, tokenizer, shuffle=True)
    val_dataloader = R2DataLoader(args, valid_ds, tokenizer,shuffle=False)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, val_dataloader, train_sampler)
    trainer.train()


if __name__ == '__main__':
    main()
