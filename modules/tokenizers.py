import json
import re
from collections import Counter
import pandas as pd
import jieba
import os

class Tokenizer(object):
    def __init__(self, args,mode="train"):
        self.threshold = args.threshold
        self.json_path = args.save_dir

        self.bos_idx=args.bos_idx
        self.eos_idx=args.eos_idx
        self.pad_idx=args.pad_idx
        unique=set()
        unique.add(self.bos_idx)
        unique.add(self.eos_idx)
        unique.add(self.pad_idx)
        self.special_token_length=len(unique)
        if mode=="train":
            self.accession_to_text = self.load_accession_text(args.xlsxfile)
            self.token2idx, self.idx2token = self.train()
        else:
            self.token2idx, self.idx2token = self.read_vocabulary()

    def read_vocabulary(self):
        with open(os.path.join(self.json_path, "token/idx2token.json"), 'r') as json_file:
            # Write the dictionary to the file using JSON format
            idx2token=json.load(json_file)

        with open(os.path.join(self.json_path, "token/token2idx.json"), 'r') as json_file:
            # Write the dictionary to the file using JSON format
            token2idx=json.load(json_file)

        return token2idx, idx2token

    def load_accession_text(self, xlsx_file):
        df = pd.read_excel(xlsx_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            row_text = (
                f"报告表现：{row['报告表现']} 报告结论：{row['报告结论']}。"
            )
            accession_to_text[row['图像名称']] = row_text
        return accession_to_text

    def train(self):
        total_tokens = []

        for example in self.accession_to_text.values():
            example=self.clean_report_mimic_cxr(example)
            # tokens =jieba.cut(example, cut_all=False)
            tokens=list(str(example))
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + self.special_token_length
            idx2token[idx + self.special_token_length] = token

        with open(os.path.join(self.json_path, "token/idx2token.json"), 'w') as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(idx2token, json_file)

        with open(os.path.join(self.json_path, "token/token2idx.json"), 'w') as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(token2idx, json_file)


        return token2idx, idx2token


    def clean_report_mimic_cxr(self, report):
        report=str(report).strip()
        report = report.replace('\n', ' ')
        report=report if report.endswith("。") else report+"。"
        # print(report)
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        report = self.clean_report_mimic_cxr(report)
        tokens=list(str(report))
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [self.bos_idx] + ids + [self.eos_idx]   
        return ids

    def confusion_matrix_tokens2ids(self, report):
        report = self.clean_report_mimic_cxr(report)
        tokens=list(str(report))
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [self.bos_idx] + ids + [self.eos_idx]  
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
