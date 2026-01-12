import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('/root/home/SELFRec')
from util.sampler import next_batch_pairwise, next_user_samples, next_item_samples
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import json
from data.loader import FileIO
from util.conf import ModelConf
import pandas as pd
from data.ui_graph import Interaction
from torch.utils.tensorboard import SummaryWriter
from util.algorithm import find_k_largest
from util.evaluation import ranking_evaluation
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20
class UniTRecAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            device: str = 'cpu'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

class TextEnchancer(torch.nn.Module):
    def __init__(self,
                 training_set,
                 test_set,
                 out_dim,
                 device,
                 conf,
                 args =None,
                 topN=20):
        super(TextEnchancer, self).__init__()
        self.device = device
        self.latent_dim = out_dim
        self.f = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.data = Interaction(conf, training_set, test_set)
        self.num_users = self.data.user_num
        self.num_items = self.data.item_num
        self.args = args
        self.user_id_embedding = nn.Embedding(self.num_users, self.latent_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.num_items, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # self.user_text_embedding = nn.Embedding(self.num_users, self.latent_dim).to(self.device)
        # self.item_text_embedding = nn.Embedding(self.num_items, self.latent_dim).to(self.device)
        # nn.init.xavier_uniform_(self.user_text_embedding.weight)
        # nn.init.xavier_uniform_(self.item_text_embedding.weight)


        self.__init_weight()
        self.max_N = topN
        self.bestPerformance = []
        self.best_type = -1
        self.norm_adj = self.data.norm_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

    def __init_weight(self):
        num_types = 2
        self.edge_q_linears = nn.ModuleList()
        self.textTrans = nn.Linear(768, self.latent_dim).to(self.device)
        for t in range(num_types):
            linear_layer = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
            nn.init.xavier_uniform_(linear_layer.weight)
            self.edge_q_linears.append(linear_layer)
        self.edge_k_linear = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.edge_k_linear.weight)
        self.edge_v_linear = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.edge_k_linear.weight)
        self.text = torch.load("/root/home/SELFRec/textFiles/Train.pt").to(self.device).detach()
        with open("/root/home/SELFRec/textFiles/reviewer_id_index.json", 'r') as file:
            self.reviewer_id_index = json.load(file)
        with open("/root/home/SELFRec/textFiles/item_id_index.json", 'r') as file:
            self.item_id_index = json.load(file)

        self.wordWiseAttention = UniTRecAttention(embed_dim=512,
                                                  num_heads=4,
                                                  dropout=0.2, device=self.device)
        self.Dropout = nn.Dropout(p=0.2).to(self.device)
        self.adjust_parameter = torch.nn.Parameter(torch.empty(1)).to(self.device)
        nn.init.uniform_(self.adjust_parameter, a=0, b=1)
        # self.SWAttn = nn.MultiheadAttention(embed_dim=768, num_heads=4, dropout=0.2).to(self.device)
        self.SWAttn = UniTRecAttention(embed_dim=768, num_heads=4, dropout=0.2, device=self.device)
        self.preprocessTextfeat()
        self.graphTrans = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)

    def preprocessTextfeat(self):
        self.usertextfeat = torch.empty(self.num_users, 768).to(self.device)
        self.itemtextfeat = torch.empty(self.num_items, 768).to(self.device)
        if self.args.SWAttn == 'False':
            print('no SWAttn')
            for key in self.reviewer_id_index.keys():
                indices = self.reviewer_id_index[key]
                text_reviewer = torch.mean(self.text[indices])
                self.usertextfeat[int(key)].copy_(text_reviewer)
            for key in self.item_id_index.keys():
                if int(key) >= self.num_items:
                    continue
                indices = self.item_id_index[key]
                text_item = torch.mean(self.text[indices])
                self.itemtextfeat[int(key)].copy_(text_item)
        elif self.args.SWAttn == 'True':
            print('SWAttn')
            for key in self.reviewer_id_index.keys():
                indices = self.reviewer_id_index[key]
                text_reviewer = torch.mean(self.SWAttn(self.text[indices].unsqueeze(1)).squeeze(), dim=0)
                self.usertextfeat = self.usertextfeat.clone()
                self.usertextfeat[int(key)] = text_reviewer
            for key in self.item_id_index.keys():
                if int(key) >= self.num_items:
                    continue
                indices = self.item_id_index[key]
                text_item = torch.mean(self.SWAttn(self.text[indices].unsqueeze(1)).squeeze(), dim=0)
                self.itemtextfeat = self.itemtextfeat.clone()
                self.itemtextfeat[int(key)] = text_item
        else:
            print('WRONG SWAttn')

    def concatEmb(self, user, item):
        return torch.cat([user, item], dim=0)

    def spiltEmb(self, emb):
        return torch.split(emb, [self.num_users, self.num_items], dim=0)

    def forward(self):
        u_g_embeddings = self.user_id_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight
        all_emb = self.concatEmb(u_g_embeddings, i_g_embeddings)
        all_emb_list = [all_emb]

        for i in range(2):
            if i == (1):
                all_emb = self.softmax(torch.sparse.mm(self.sparse_norm_adj, all_emb))
            else:
                all_emb = torch.sparse.mm(self.sparse_norm_adj, all_emb)
            all_emb_list.append(all_emb)
        u_g_embeddings, i_g_embeddings = self.spiltEmb(torch.mean(torch.stack(all_emb_list), dim=0))


        text_feats = self.Dropout(self.textTrans(self.concatEmb(self.usertextfeat, self.itemtextfeat)))
        user_text_feats, item_text_feats = self.spiltEmb(text_feats)

        user_q_feature = self.edge_q_linears[0](u_g_embeddings)
        item_q_feature = self.edge_q_linears[1](i_g_embeddings)


        text_user_k_values = self.edge_k_linear(user_text_feats)
        text_user_v_values = self.edge_v_linear(user_text_feats)
        user_attention_scores = torch.matmul(text_user_k_values, user_q_feature.t())
        user_attention_scores = F.softmax(user_attention_scores, dim=-1)
        user_text_v = torch.matmul(user_attention_scores, text_user_v_values)

        text_item_k_values = self.edge_k_linear(item_text_feats)
        text_item_v_values = self.edge_v_linear(item_text_feats)
        item_attention_scores = torch.matmul(text_item_k_values, item_q_feature.t())
        item_attention_scores = F.softmax(item_attention_scores, dim=-1)
        item_text_v = torch.matmul(item_attention_scores, text_item_v_values)

        user_text_feats_aug, item_text_feats_aug = self.spiltEmb(torch.sparse.mm(self.sparse_norm_adj, self.concatEmb(user_text_v, item_text_v)))

        u_g_embeddings = u_g_embeddings + 0.02 * F.normalize(user_text_feats_aug, p=2,
                                                                            dim=1)

        i_g_embeddings = i_g_embeddings + 0.02 * F.normalize(item_text_feats_aug, p=2,
                                                                            dim=1)

        return u_g_embeddings, i_g_embeddings, user_text_feats, item_text_feats


    def test(self, test_users, test_items, eva_type=1):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum * 2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user, test_users, test_items, evaluate_type=eva_type)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def fast_evaluation(self, epoch, test_users, test_items):
        print('Evaluating the model...')
        measures = []

        rec_list = self.test(test_users, test_items, eva_type=1)
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                # self.save()
            measures.append(measure)
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            # self.save()
            print('-' * 120)
            print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
            measure = [m.strip() for m in measure[1:]]
            print(f'*Current Performance*')
            print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
            measures.append(measure)
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print(f'*Best Performance in Type {self.best_type}* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return np.array(measures)

    def predict(self, u, test_users, test_items, evaluate_type=1):
        u = self.data.get_user_id(u)
        score = torch.matmul(test_users[u], test_items.transpose(0, 1))
        return score.cpu().numpy()


if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from tqdm import tqdm
    import argparse
    from datetime import datetime

    current_time = datetime.now()
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 5],
                        help='Number of neighbors to sample at each layer')
    parser.add_argument('--prefix', type=str, default='', help='tensorboard Prefix')
    parser.add_argument('--device', type=str, default='1', help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--SWAttn', type=str, default='False', help='Use Sentence-Weighted Attention')
    parser.add_argument('--WWAttn', type=str, default='False', help='Use Word-Wise Attention')
    args = parser.parse_args()
    writer = SummaryWriter(f'runs/experiment_{args.prefix}_{current_time.strftime("%Y-%m-%d %H:%M:%S")}')
    epochs = 1001
    model = 'LightGCN'
    conf = ModelConf('/root/home/SELFRec/conf/' + model + '.conf')
    training_data = FileIO.load_data_set(conf['training.set'], conf['model.type'])
    test_data = FileIO.load_data_set(conf['test.set'], conf['model.type'])
    DEVICE = f'cuda:{args.device}'
    model = TextEnchancer(training_set=training_data, test_set=test_data, out_dim=64, device=DEVICE, conf=conf, args=args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    train_data = model.data
    batch_size = 512
    with torch.autograd.set_detect_anomaly(True):
        batch_index = 0
        for epoch in range(epochs):
            for n, batch in enumerate(next_batch_pairwise(train_data, batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, rec_user_text_emb, rec_item_text_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                user_text_emb, pos_item_text_emb, neg_item_text_emb = rec_user_text_emb[user_idx], rec_item_text_emb[pos_idx], rec_item_text_emb[
                    neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(0.0001, user_emb,
                                                                                          pos_item_emb,
                                                                                          neg_item_emb) / batch_size
                batch_text_loss = bpr_loss(user_text_emb, pos_item_text_emb, neg_item_text_emb) + l2_reg_loss(0.0001, user_text_emb,
                                                                                          pos_item_text_emb,
                                                                                          neg_item_text_emb) / batch_size
                batch_all_loss = batch_loss + batch_text_loss
                optimizer.zero_grad()
                # for name, param in model.named_parameters():
                #     print(f"Parameter: {name}, Version: {param._version}")
                batch_all_loss.backward(retain_graph=True)
                optimizer.step()
                print(batch_loss)
                batch_index += 1
                writer.add_scalar('training loss', batch_all_loss.item(), batch_index)
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_all_loss.item())
            print(epoch)
            if epoch % 5 == 0:
                with torch.no_grad():
                    rec_user_emb, rec_item_emb, rec_user_text_emb, rec_item_text_emb = model()
                    measures = model.fast_evaluation(epoch, rec_user_emb, rec_item_emb)[0]

        writer.close()