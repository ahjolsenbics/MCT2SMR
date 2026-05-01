import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


class R2DataLoader(DataLoader):
    def __init__(self, args, dataset, tokenizer,shuffle=None,sampler=None):
        self.args = args
        self.batch_size = args.batch_size
        # self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.shuffle = shuffle
        self.dataset = dataset
        # self.split = split
        if self.shuffle==None:
            self.init_kwargs = {
                'dataset': self.dataset,
                'batch_size': self.batch_size,
                # 'shuffle': self.shuffle,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                'sampler': self.sampler
            }
        else:
            self.init_kwargs = {
                'dataset': self.dataset,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                # 'sampler': self.sampler
            }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, seg_images, reports_ids, reports_masks, seq_lengths, block_ids, blocktext_len_list = zip(*data)
        images = torch.stack(images, 0)
        seg_images = torch.stack(seg_images, 0)
        max_seq_length = max(seq_lengths)
        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)


        max_rows = max(len(seq) for seq in block_ids)
        max_cols = max(len(row) for seq in block_ids for row in seq)
        block_targets = np.zeros((len(block_ids), max_rows, max_cols), dtype=int)
        blockids_len_list = np.zeros((len(blocktext_len_list), max_rows), dtype = int)

        for i, block_report_ids in enumerate(block_ids):
            for j, block_report_id in enumerate(block_report_ids):
                block_targets[i, j, :len(block_report_id)] = block_report_id

        for i, blockid_len in enumerate(blocktext_len_list):
            blockids_len_list[i, :len(blockid_len)] = blockid_len

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, seg_images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(block_targets), torch.LongTensor(blockids_len_list)

