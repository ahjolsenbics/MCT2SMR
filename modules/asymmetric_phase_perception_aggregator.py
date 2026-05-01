import numpy as np
import torch
import torch.nn as nn



class DynamicPhaseAwareCrossAttention(nn.Module):
    def __init__(self, args, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_phases = len(args.phase_name)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.c_phase_norm = nn.LayerNorm(embed_dim)
        self.a_phase_norm = nn.LayerNorm(embed_dim)
        self.p_phase_norm = nn.LayerNorm(embed_dim)
        self.v_phase_norm = nn.LayerNorm(embed_dim)

        self.phase_embeddings = nn.Embedding(self.num_phases * (self.num_phases - 1), embed_dim) 

        self.aggregation_weights = nn.ParameterDict({      
            'c_gate': nn.Parameter(torch.ones(self.num_phases -1)),  # C <- (A, P, V)
            'a_gate': nn.Parameter(torch.ones(self.num_phases -1)),
            'p_gate': nn.Parameter(torch.ones(self.num_phases -1)),
            'v_gate': nn.Parameter(torch.ones(self.num_phases -1)),
        })

        self.scale = 1.0 / (embed_dim ** 0.5)

    def _add_phase_bias(self, scores, query, phase_emb, seq_len):
        bias = torch.matmul(query, phase_emb.transpose(-1, -2)) * self.scale
        return scores + bias

    def forward(self, tokens):
        c_phase, a_phase, p_phase, v_phase = tokens
        batch_size, seq_len, _ = c_phase.size()

        query_c, key_c, value_c = self.query(c_phase), self.key(c_phase), self.value(c_phase)
        query_a, key_a, value_a = self.query(a_phase), self.key(a_phase), self.value(a_phase)
        query_p, key_p, value_p = self.query(p_phase), self.key(p_phase), self.value(p_phase)
        query_v, key_v, value_v = self.query(v_phase), self.key(v_phase), self.value(v_phase)

        # Get phase embeddings
        phase_emb = self.phase_embeddings(torch.arange(self.num_phases * (self.num_phases - 1)).to(c_phase.device))
        phase_emb = phase_emb.view(self.num_phases * (self.num_phases - 1), 1, -1).expand(-1, seq_len, -1)

        attn_ca = self.softmax(
            self._add_phase_bias(torch.matmul(query_c, key_a.transpose(-1, -2)) * self.scale, query_c, phase_emb[0],seq_len))
        
        attn_cp = self.softmax(
            self._add_phase_bias(torch.matmul(query_c, key_p.transpose(-1, -2)) * self.scale, query_c, phase_emb[1],seq_len))
        attn_cv = self.softmax(
            self._add_phase_bias(torch.matmul(query_c, key_v.transpose(-1, -2)) * self.scale, query_c, phase_emb[2],seq_len))

        attn_ac = self.softmax(
            self._add_phase_bias(torch.matmul(query_a, key_c.transpose(-1, -2)) * self.scale, query_a, phase_emb[3],seq_len))
        attn_ap = self.softmax(
            self._add_phase_bias(torch.matmul(query_a, key_p.transpose(-1, -2)) * self.scale, query_a, phase_emb[4],seq_len))
        attn_av = self.softmax(
            self._add_phase_bias(torch.matmul(query_a, key_v.transpose(-1, -2)) * self.scale, query_a, phase_emb[5],seq_len))

        attn_pc = self.softmax(
            self._add_phase_bias(torch.matmul(query_p, key_c.transpose(-1, -2)) * self.scale, query_p, phase_emb[6],seq_len))
        attn_pa = self.softmax(
            self._add_phase_bias(torch.matmul(query_p, key_a.transpose(-1, -2)) * self.scale, query_p, phase_emb[7],seq_len))
        attn_pv = self.softmax(
            self._add_phase_bias(torch.matmul(query_p, key_v.transpose(-1, -2)) * self.scale, query_p, phase_emb[8],seq_len))


        attn_vc = self.softmax(
            self._add_phase_bias(torch.matmul(query_v, key_c.transpose(-1, -2)) * self.scale, query_v, phase_emb[9],seq_len))
        attn_va = self.softmax(
            self._add_phase_bias(torch.matmul(query_v, key_a.transpose(-1, -2)) * self.scale, query_v, phase_emb[10],seq_len))
        attn_vp = self.softmax(
            self._add_phase_bias(torch.matmul(query_v, key_p.transpose(-1, -2)) * self.scale, query_v, phase_emb[11],seq_len))

        w_c = torch.softmax(self.aggregation_weights['c_gate'], dim=0)
        w_a = torch.softmax(self.aggregation_weights['a_gate'], dim=0)
        w_p = torch.softmax(self.aggregation_weights['p_gate'], dim=0)
        w_v = torch.softmax(self.aggregation_weights['v_gate'], dim=0)

        c_interacted = (w_c[0] * torch.matmul(attn_ca, value_a) +
                        w_c[1] * torch.matmul(attn_cp, value_p) +
                        w_c[2] * torch.matmul(attn_cv, value_v))

        a_interacted = (w_a[0] * torch.matmul(attn_ac, value_c) +
                        w_a[1] * torch.matmul(attn_ap, value_p) +
                        w_a[2] * torch.matmul(attn_av, value_v))

        p_interacted = (w_p[0] * torch.matmul(attn_pc, value_c) +
                        w_p[1] * torch.matmul(attn_pa, value_a) +
                        w_p[2] * torch.matmul(attn_pv, value_v))

        v_interacted = (w_v[0] * torch.matmul(attn_vc, value_c) +
                        w_v[1] * torch.matmul(attn_va, value_a) +
                        w_v[2] * torch.matmul(attn_vp, value_p))

        c_final = self.c_phase_norm(c_phase + c_interacted)
        a_final = self.a_phase_norm(a_phase + a_interacted)
        p_final = self.p_phase_norm(p_phase + p_interacted)
        v_final = self.v_phase_norm(v_phase + v_interacted)

        fusion_final = torch.cat([c_final, a_final, p_final, v_final], dim=-1)

        return fusion_final
