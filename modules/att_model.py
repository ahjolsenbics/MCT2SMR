from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import modules.utils as utils
from modules.caption_model import CaptionModel


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, args, tokenizer):
        super(AttModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        self.input_encoding_size = args.d_model
        self.rnn_size = args.d_ff
        self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob_lm
        self.max_seq_length = args.max_seq_length
        self.att_feat_size = args.d_vf
        self.att_hid_size = args.d_model
        self.d_model = args.d_model

        self.device = args.device

        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.pad_idx = args.pad_idx

        unique_idx = set()
        unique_idx.add(args.bos_idx)
        unique_idx.add(args.eos_idx)
        unique_idx.add(args.pad_idx)

        self.total_vocab_size = len(unique_idx)+self.vocab_size

        self.use_bn = args.use_bn
        self.re_logit = nn.Linear(self.total_vocab_size, self.d_model)

        self.embed = lambda x: x
        self.fc_embed = lambda x: x
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))
        self.x = 0

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None: # att_masks为none，所以不通过下面的if语句里面
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def get_meta_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, block_num, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state, meta_token = self.meta_core(xt, p_att_feats, state, att_masks, block_num)  # output:[i,2048] state(list):tensor  这里的output会随着循环一直增加[i, 2048] 会将下一个词也加进来。
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)  # [1, 778]
        else:
            logprobs = self.logit(output)

        return logprobs, state, meta_token

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)  # output:[i,2048] state(list):tensor  这里的output会随着循环一直增加[i, 2048] 会将下一个词也加进来。
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)  # [1, 778]
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _meta_sample(self, fc_feats, att_feats, seg_feats, att_masks=None):
        opt = self.args.__dict__
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)
        # fc_feats[1,3072] att_feats[1,8000,3072]  p_fc_feats[1,1] p_att_feats[1,8000, 2048]
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        #add visual features and seg features
        pp_att_feats = pp_att_feats + seg_feats

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,[p_fc_feats, p_att_feats, pp_att_feats, p_att_masks])

        trigrams = []  # will be a list of batch_size dictionaries
        meta_tokens = []

        seq = pp_att_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = pp_att_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.total_vocab_size)

        block_seq = [fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long) for _ in range(6)]  # list: [1, 512]
        block_seqLogprobs = [fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.total_vocab_size) for _ in range(6)]    # list: [1,512, 778]

        for b_i in range(6):
            for t in range(self.max_seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)  # (1,) tensor[self.bos_idx]

                logprobs, state, meta_token = self.get_meta_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, b_i, output_logsoftmax=output_logsoftmax)

                if decoding_constraint and t > 0:
                    tmp = logprobs.new_zeros(logprobs.size())
                    tmp.scatter_(1, block_seq[b_i][:, t - 1].data.unsqueeze(1), float('-inf'))
                    logprobs = logprobs + tmp

                # Mess with trigrams
                # Copy from https://github.com/lukemelas/image-paragraph-captioning
                if block_trigrams and t >= 3:
                    # Store trigram generated at last step
                    prev_two_batch = block_seq[b_i][:, t - 3:t - 1]
                    for i in range(batch_size):  # = seq.size(0)
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        current = block_seq[b_i][i][t - 1]
                        if t == 3:  # initialize
                            trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                        elif t > 3:
                            if prev_two in trigrams[i]:  # add to list
                                trigrams[i][prev_two].append(current)
                            else:  # create list
                                trigrams[i][prev_two] = [current]
                    # Block used trigrams at next step
                    prev_two_batch = block_seq[b_i][:, t - 2:t]
                    mask = torch.zeros(logprobs.size(), requires_grad=False).to(self.device)  # batch_size x vocab_size
                    for i in range(batch_size):
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        if prev_two in trigrams[i]:
                            for j in trigrams[i][prev_two]:
                                mask[i, j] += 1
                    # Apply mask to log probs
                    # logprobs = logprobs - (mask * 1e9)
                    alpha = 2.0  # = 4
                    logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

                # sample the next word
                if t == self.max_seq_length:  # skip if we achieve maximum length
                    break
                it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

                # stop when all finished
                if t == 0:
                    unfinished = it != self.eos_idx
                else:
                    it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                    logprobs = logprobs * unfinished.unsqueeze(1).float()
                    unfinished = unfinished * (it != self.eos_idx)

                seq[:, t] = it
                seqLogprobs[:, t] = logprobs

                block_seq[b_i][:, t] = it
                block_seqLogprobs[b_i][:, t] = logprobs

                # quit loop if all sequences have finished
                if unfinished.sum() == 0:
                    break

            meta_tokens.append(meta_token)

        # meta_tokens = []
        # for _ in range(6):
        #     meta_token = torch.rand(1, 1, 2048)
        #     meta_tokens.append(meta_token.to(pp_att_feats.device))

        # fusion predict: using visual features and meta tokens
        out_seq, out_block_seqLogprobs = self._sample(p_fc_feats, p_att_feats, pp_att_feats, meta_tokens)

        return out_seq, out_block_seqLogprobs

    def _sample(self, p_fc_feats, p_att_feats, pp_att_feats, meta_tokens):
        opt = self.args.__dict__
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)

        batch_size = pp_att_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)
        # fc_feats[1,3072] att_feats[1,8000,3072]  p_fc_feats[1,1] p_att_feats[1,8000, 2048]

        trigrams = []  # will be a list of batch_size dictionaries

        # #add visual features and seg features
        # pp_att_feats = pp_att_feats + seg_feats

        # cat visual features and meta tokens
        for b in range(6):
            pp_att_feats = torch.cat([pp_att_feats, meta_tokens[b]], dim=1)
        # print(pp_att_feats.size())
        p_att_masks = self._prepare_fusion_feature_forward(pp_att_feats)

        seq = pp_att_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)  # [1, 512]      全0
        seqLogprobs = pp_att_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.total_vocab_size)    # [1,512, 778]  全0
        for t in range(self.max_seq_length + 1):
            if t == 0:  # input <bos>
                it = pp_att_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)  # (1,) tensor[self.bos_idx]

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(self.device)  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.max_seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)]  # will be a list of batch_size dictionaries

        seq_table = [fc_feats.new_full((batch_size, self.max_seq_length), self.pad_idx, dtype=torch.long) for _ in range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.max_seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.max_seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.max_seq_length - 1:
                    if t == 0:  # input <bos>
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t - 1]  # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state_table[divm])  # changed
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda

                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:, t - 3:t - 1]
                        for i in range(batch_size):  # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current = seq[i][t - 1]
                            if t == 3:  # initialize
                                trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]:  # add to list
                                    trigrams[i][prev_two].append(current)
                                else:  # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:, t - 2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).to(self.device)  # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i, j] += 1
                        # Apply mask to log probs
                        # logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0  # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = seq[:, t - 1] != self.pad_idx & seq[:, t - 1] != self.eos_idx
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx)  # changed
                    seq[:, t] = it
                    seqLogprobs[:, t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table,1).reshape(batch_size * group_size, -1)

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.total_vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.total_vocab_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats,
                                                                                   pp_att_feats, p_att_masks]
                                                                                  )
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs