from transformers import RobertaForMaskedLM, RobertaTokenizer
import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F
import os
import pickle
from torch.nn import TransformerEncoder, TransformerEncoderLayer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add SELFRec to path using relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
selfrec_path = os.path.join(project_root, 'SELFRec')
sys.path.insert(0, selfrec_path)

from util.sampler import next_batch_pairwise, next_user_samples, next_item_samples
from base.torch_interface import TorchGraphInterface
from base.graph_recommender import GraphRecommender
from data.loader import FileIO
from data.ui_graph import InteractionPlus
import random
from tqdm import tqdm


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, graph_token_dim, text_token_dim):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = text_token_dim
        self.graph_dim = graph_token_dim
        self._num_embeddings = num_embeddings
        self.intermediate_dim = 512
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self.W_graph = nn.Parameter(torch.rand(graph_token_dim, text_token_dim))
        # self.W_text = nn.Parameter(torch.rand(text_token_dim, self.intermediate_dim))
        nn.init.xavier_uniform_(self.W_graph)
        # nn.init.xavier_uniform_(self.W_text)

    def set_codebook(self, codebook):
        self._embedding = nn.Embedding.from_pretrained(codebook, freeze=False)
        self._num_embeddings = codebook.shape[0]

    def text_graph_dim_align(self, graph_emb):
        Graph_proj = torch.matmul(graph_emb, self.W_graph)   # shape: [n, d_e]  
        Graph_proj = F.normalize(Graph_proj, p=2, dim=1)  # [n, d_e]
        return Graph_proj

    def orthogonality_loss(self, x):
        B, K, D = x.shape
        x = F.normalize(x, p=2, dim=2)
        sim_matrix = torch.bmm(x, x.transpose(1, 2))
        identity = torch.eye(K, device=x.device).unsqueeze(0)  # shape: [1, K, K]
        off_diag = sim_matrix * (1 - identity)
        if K == 1:
            return torch.tensor(0.0)
        loss = (off_diag ** 2).sum(dim=(1, 2)) / (K * (K - 1))  # per-sample average
        return loss.mean()  # mean over batch

    def cal_ras_loss(self, input_text_token):
        flat_input = input_text_token.view(-1, self._embedding_dim)
        flat_input = F.normalize(flat_input, p=2, dim=-1)
        embedding_weight = F.normalize(self._embedding.weight, p=2, dim=-1)

        cos_sim = torch.matmul(flat_input, embedding_weight.t())
        cos_distance = 1 - cos_sim

        encoding_indices = torch.argmin(cos_distance, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=input_text_token.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized_aligned = torch.matmul(encodings, self._embedding.weight).view(input_text_token.shape[0], 2, -1)
        cos_text_sim = F.cosine_similarity(quantized_aligned[:, 0, :], quantized_aligned[:, 1, :], dim=-1)
        return cos_text_sim


    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.graph_dim)
        orth_loss = self.orthogonality_loss(inputs.reshape(inputs.shape[0], -1, self.graph_dim))
        flat_input = self.text_graph_dim_align(flat_input)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_aligned = torch.matmul(encodings, self._embedding.weight).view(input_shape[0], -1)
        flat_input = flat_input.view(input_shape[0], -1)
        e_latent_loss = F.mse_loss(quantized_aligned.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized_aligned, flat_input.detach())
        loss = q_latent_loss + 0.5 * e_latent_loss
        quantized = flat_input + (quantized_aligned - flat_input).detach()
        return quantized.squeeze(), loss + orth_loss, encoding_indices.reshape(input_shape[0], -1), distances

class VectorQuantizerGlobal(nn.Module):
    def __init__(self, num_embeddings, graph_token_dim, text_token_dim):
        super(VectorQuantizerGlobal, self).__init__()
        self._embedding_dim = text_token_dim
        self.graph_dim = graph_token_dim
        self._num_embeddings = num_embeddings
        self.intermediate_dim = 512
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self.W_graph = nn.Parameter(torch.rand(graph_token_dim, text_token_dim))
        # self.W_text = nn.Parameter(torch.rand(text_token_dim, self.intermediate_dim))
        nn.init.xavier_uniform_(self.W_graph)
        # nn.init.xavier_uniform_(self.W_text)

    def set_codebook(self, codebook):
        self._embedding = nn.Embedding.from_pretrained(codebook, freeze=False)
        self._num_embeddings = codebook.shape[0]

    def text_graph_dim_align(self, graph_emb):
        Graph_proj = torch.matmul(graph_emb, self.W_graph)   # shape: [n, d_e]  
        Graph_proj = F.normalize(Graph_proj, p=2, dim=1)  # [n, d_e]
        return Graph_proj

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.graph_dim)
        flat_input = self.text_graph_dim_align(flat_input)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_aligned = torch.matmul(encodings, self._embedding.weight).view(input_shape[0], -1)
        flat_input = flat_input.view(input_shape[0], -1)
        e_latent_loss = F.mse_loss(quantized_aligned.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized_aligned, flat_input.detach())
        loss = q_latent_loss + 0.5 * e_latent_loss
        quantized = flat_input + (quantized_aligned - flat_input).detach()
        return quantized.squeeze(), loss, encoding_indices.reshape(input_shape[0], -1), distances
    

# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20
class TextEnchancer(torch.nn.Module):
    def __init__(self,
                 training_set,
                 test_set,
                 out_dim,
                 device,
                 conf,
                 text_dataloader,
                 text_encoder,
                 args=None,
                 token_dim=64,
                 unified_length=10,
                 num_tokens=512,
                 llm_dim=4096,
                 topN=20):
        super(TextEnchancer, self).__init__()
        self.device = device
        self.latent_dim = out_dim
        self.hidden_dim = 1024
        self.f = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.data = InteractionPlus(conf, training_set, test_set)
        self.num_users = self.data.user_num
        self.num_items = self.data.item_num
        self.args = args
        self.token_dim = token_dim
        self.user_id_embedding = nn.Embedding(self.num_users, self.latent_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.num_items, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.max_N = topN
        self.text_dataloader = text_dataloader
        self.text_model = text_encoder
        self.bestPerformance = []
        self.norm_adj = self.data.norm_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)
        self.disenQuan_u_local = VectorQuantizer(num_embeddings=num_tokens, graph_token_dim=token_dim, text_token_dim=llm_dim)
        self.disenQuan_i_local = VectorQuantizer(num_embeddings=num_tokens, graph_token_dim=token_dim, text_token_dim=llm_dim)
        self.disenQuan_i_global = VectorQuantizerGlobal(num_embeddings=num_tokens, graph_token_dim=args.token_len * token_dim, text_token_dim=llm_dim)
        self.disenQuan_u_global = VectorQuantizerGlobal(num_embeddings=num_tokens, graph_token_dim=args.token_len * token_dim, text_token_dim=llm_dim)
        self.interaction_mat = torch.from_numpy(self.data.interaction_mat.toarray()).to(self.device)
        self.llm_dim = llm_dim
        self.layers = 2

    def concatEmb(self, user, item):
        return torch.cat([user, item], dim=0)

    def init_weight(self, linear_module):
        import torch.nn.init as init
        init.xavier_uniform_(linear_module[0].weight)  # 初始化第一层线性变换的权重
        if linear_module[0].bias is not None:  # 初始化偏置
            init.zeros_(linear_module[0].bias)
        if len(linear_module) > 1:  # 如果有第二层，初始化第二层
            init.xavier_uniform_(linear_module[2].weight)  # 初始化第二层线性变换的权重
            if linear_module[2].bias is not None:  # 初始化偏置
                init.zeros_(linear_module[2].bias)

    def spiltEmb(self, emb):
        return torch.split(emb, [self.num_users, self.num_items], dim=0)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def forward(self):
        u_g_embeddings = self.user_id_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight
        ego_embeddings = torch.cat([u_g_embeddings, i_g_embeddings], dim=0)
        cge_embs = [ego_embeddings]
        for _ in range(2):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            cge_embs += [ego_embeddings]
        cge_embs = torch.stack(cge_embs, dim=1).mean(dim=1, keepdim=False)
        u_g, i_g = torch.split(cge_embs, [self.num_users, self.num_items], dim=0)
        
        u_rep_local, loss_u_local, u_indices_local, _ = self.disenQuan_u_local(u_g)
        i_rep_local, loss_i_local, i_indices_local, _ = self.disenQuan_i_local(i_g)
        u_rep_global, loss_u_global, u_indices_global, _ = self.disenQuan_u_global(u_g)
        i_rep_global, loss_i_global, i_indices_global, _ = self.disenQuan_i_global(i_g)
        return u_rep_local, i_rep_local, u_rep_global, i_rep_global, loss_u_local + loss_i_local + loss_u_global + loss_i_global, u_indices_local, i_indices_local, u_indices_global, i_indices_global

    def get_embeddings(self):
        return self.forward()


    def reconstructing_edge(self, user, item, user_idx, item_idx):
        # user = user.reshape(-1, self.token_dim)
        # item = item.reshape(-1, self.token_dim)
        recon_edge = (user[user_idx] * item[item_idx]).sum(1)
        adj_quantized = (recon_edge - recon_edge.min()) / (recon_edge.max() - recon_edge.min())
        # item_recon = self.edge_decoder_i(item[item_idx])
        edge_rec_loss = torch.sqrt(F.mse_loss(self.interaction_mat[user_idx, item_idx], adj_quantized.squeeze()))
        return edge_rec_loss

class CrossAttention(nn.Module):
    def __init__(self, token_dim):
        super(CrossAttention, self).__init__()
        self.query_fc = nn.Linear(token_dim, token_dim)
        self.key_fc = nn.Linear(token_dim, token_dim)
        self.value_fc = nn.Linear(token_dim, token_dim)
        self.out_fc = nn.Linear(token_dim, token_dim)

    def forward(self, x1, x2, masked_indices):
        # x1: [batch_size, token_len_1, token_dim]
        # x2: [batch_size, token_len_2, token_dim]

        # 计算 queries, keys 和 values
        queries = self.query_fc(x1)  # [batch_size, token_len_1, token_dim]
        keys = self.key_fc(x2)        # [batch_size, token_len_2, token_dim]
        values = self.value_fc(x2)    # [batch_size, token_len_2, token_dim]

        # 注意力机制
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [batch_size, token_len_1, token_len_2]
        attention_scores = attention_scores / (keys.size(-1) ** 0.5)  # 缩放
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [batch_size, token_len_1, token_len_2]

        # 加权和值
        context = torch.matmul(attention_probs, values)  # [batch_size, token_len_1, token_dim]

        output = context[torch.arange(masked_indices.shape[0]).unsqueeze(1), masked_indices]  # [batch_size, token_len_1, token_dim]
        return output



class MaskedWordModel(nn.Module):
    def __init__(self, embed_size, max_len, mask_ratio, device, num_heads=4, num_layers=2):
        super(MaskedWordModel, self).__init__()
        self.mask_token = nn.Parameter(torch.randn(max_len, embed_size)).to(device)  # 学习able mask token
        self.max_len = max_len
        self.mask_attn = nn.MultiheadAttention(embed_size, num_heads, batch_first=True).to(device)
        self.cross_attn = CrossAttention(embed_size).to(device)
        self.mask_ratio = mask_ratio
        # decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads, embed_size).to(device)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers).to(device)
        # self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, text_emb, behave_emb):
        B, seq_len, _ = text_emb.size()
        behave_emb = behave_emb.reshape(text_emb.size())
        num_tokens_to_mask = int(self.mask_ratio * seq_len)
        masked_input = text_emb.clone()
        rand_positions = torch.randperm(seq_len, generator=torch.manual_seed(2025)).unsqueeze(0)  # 生成随机排列
        rand_positions = rand_positions.repeat(B, 1)  # 重复以适应全部批次
        masked_indices = rand_positions[:, :num_tokens_to_mask].clone().cuda()  
        mask = torch.zeros_like(masked_input).bool()
        mask[:, masked_indices] = True  # 将选择的索引位置标记为 True
        masked_input[mask] = self.mask_token[masked_indices - 1].reshape(-1)
        masked_repre = self.mask_attn(masked_input, masked_input, masked_input)[0]
        cross_fusion_repre = self.cross_attn(masked_repre, behave_emb, masked_indices)
        target_masked_tokens = text_emb.gather(dim=1, index=masked_indices.unsqueeze(-1).expand(-1, -1, text_emb.size(-1))).squeeze()
        predicted_masked_tokens = cross_fusion_repre.squeeze()  # shape: [B, num_mask, hidden]
        mse_loss = F.mse_loss(predicted_masked_tokens, target_masked_tokens)
        return mse_loss

class Trainer(GraphRecommender):
    def __init__(self, conf, training_set, test_set, device, args, train_dataloader, output_dim, text_model, llm_dim, token_dim, word_embeddings, mask_code):
        super(Trainer, self).__init__(conf, training_set, test_set)
        self.token_num = args.codebook_size
        self.model = TextEnchancer(training_set=training_set, test_set=test_set, out_dim=output_dim, device=device, token_dim=token_dim,
                                   num_tokens=self.token_num, conf=conf, args=args, text_dataloader=train_dataloader, text_encoder=text_model, llm_dim=llm_dim)
        self.args = args
        self.device = device
        self.train_data = self.model.data
        self.batch_size = args.batch_size
        self.rating_loss = nn.MSELoss()
        self.best_user_emb = None
        self.reverse_user_map = {v: k for k, v in self.train_data.user.items()}
        self.reverse_item_map = {v: k for k, v in self.train_data.item.items()}
        self.train_dataloader = train_dataloader
        self.text_model = text_model
        self.token_dim = token_dim
        self.token_per_num = output_dim // token_dim
        self.word_embeddings = word_embeddings
        self.mask_token_id = mask_code
        self.masked_token_model = MaskedWordModel(embed_size=llm_dim, max_len=self.token_per_num, mask_ratio=0.2, device=device, num_heads=4)
        self.reformat_user = [int(item) for item in self.data.id2user.values()]
        self.reformat_item = [int(item) for item in self.data.id2item.values()]

        # Use relative data path
        data_dir = os.path.join(project_root, 'data', args.dataset)

        if self.args.zero_rate == 0:
            trn_file = os.path.join(data_dir, 'trn.pkl')
            with open(trn_file, "rb") as file:
                trn_data = pickle.load(file)
                self.trn_data = trn_data.to_dict("list")
        else:
            trn_file = os.path.join(data_dir, f'trn_{str(self.args.zero_rate)}_uid_pruned.pkl')
            with open(trn_file, "rb") as file:
                trn_data = pickle.load(file)
                self.trn_data = trn_data

        self.load_language_tokens(self.args.dataset)
        uid_mapped = [self.data.id2user[item] for item in self.trn_data['uid']]
        iid_mapped = [self.data.id2item[item] for item in self.trn_data['iid']]
        self.uid_iid_to_idx = { (int(uid), int(iid)): idx for idx, (uid, iid) in enumerate(zip(uid_mapped, iid_mapped)) }
        self.uid_iid_to_idx_test = { (int(uid), int(iid)): idx for idx, (uid, iid) in enumerate(zip(self.trn_data['uid'], self.trn_data['iid'])) }

    
    def _ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def load_language_tokens(self, dataset):
        data_dir = os.path.join(project_root, 'data', dataset)

        user_token_file = os.path.join(data_dir, 'user_interest_token.pkl')
        item_token_file = os.path.join(data_dir, 'item_interest_token.pkl')
        cls_token_file = os.path.join(data_dir, 'cls_token.pt')

        with open(user_token_file, 'rb') as f:
            user_interest_token = pickle.load(f)
        with open(item_token_file, 'rb') as f:
            item_interest_token = pickle.load(f)

        self.exp_cls_token = torch.load(cls_token_file).cuda()

        self.user_interest_token = {
            int(self.data.id2user[int(k)]): v[:self.token_per_num].cuda() for k, v in user_interest_token.items()
        }
        self.item_interest_token = {
            int(self.data.id2item[int(k)]): v[:self.token_per_num].cuda() for k, v in item_interest_token.items()
        }

        
    def get_language_tokens(self, user_idx, item_idx, user_repre, item_repre):
        user_idx = torch.unique(user_idx).cpu().numpy().tolist()
        item_idx = torch.unique(item_idx).cpu().numpy().tolist()
        user_lan_token = [(self.user_interest_token[key], user_repre[key]) for key in user_idx if key in self.user_interest_token]
        item_lan_token = [(self.item_interest_token[key], item_repre[key]) for key in item_idx if key in self.item_interest_token]
        user_token_list, user_repre_list = zip(*user_lan_token)
        user_token_tensor = torch.stack(user_token_list)   # shape: [N, ...]
        user_repre_tensor = torch.stack(user_repre_list)
        item_token_list, item_repre_list = zip(*item_lan_token)
        item_token_tensor = torch.stack(item_token_list)   # shape: [N, ...]
        item_repre_tensor = torch.stack(item_repre_list)
        return user_token_tensor, user_repre_tensor, item_token_tensor, item_repre_tensor
    
    def get_cls_tokens(self, user_idx, item_idx, user_repre, item_repre):
        user_idx = user_idx.cpu().numpy().tolist()
        item_idx = item_idx.cpu().numpy().tolist()
        pair_indices = [
            (self.exp_cls_token[self.uid_iid_to_idx[(u, i)]], user_repre[u] + item_repre[i])
            for u, i in zip(user_idx, item_idx)
            if (u, i) in self.uid_iid_to_idx
        ]
        if pair_indices == []:
            return None, None
        user_token_list, user_repre_list = zip(*pair_indices)
        exp_cls_tensor = torch.stack(user_token_list)   # shape: [N, ...]
        exp_behave_tensor = torch.stack(user_repre_list)
        return exp_cls_tensor, exp_behave_tensor


    def train(self):
        model = self.model.to(self.device)
        params = list(self.model.parameters()) + list(self.masked_token_model.parameters())
        optimizer_model = torch.optim.Adam(params, lr=self.args.lr)  # 对整个模型优化
        self.best_hr = 0
        self.no_improvement_epochs = 0
        with torch.autograd.set_detect_anomaly(True):
            batch_index = 0
            print("start Training")
            for epoch in range(epochs):
                batch_dataloader = next_batch_pairwise(self.train_data, self.batch_size)
                for batch in batch_dataloader:
                    user_idx, item_idx, neg_item_idx = batch
                    user_idx = torch.tensor(user_idx + user_idx, dtype=torch.long).to(self.device)

                    item_idx = torch.tensor(item_idx + neg_item_idx, dtype=torch.long).to(self.device)
                    shuffle_indices = torch.randperm(item_idx.size(0))
                    shuffled_item_idx = item_idx[shuffle_indices]
                    user_idx = user_idx[shuffle_indices]
                    u_rep_local, i_rep_local, u_rep_global, i_rep_global, loss_vq, u_indices_local, i_indices_local, u_indices_global, i_indices_global = model()
                    rec_user_emb = u_rep_local + u_rep_global.repeat(1, self.args.token_len)
                    rec_item_emb = i_rep_local + i_rep_global.repeat(1, self.args.token_len)
                    recon_loss = model.reconstructing_edge(rec_user_emb, rec_item_emb, user_idx, shuffled_item_idx)
                    user_lan_token, user_behave_token, item_lan_token, item_behave_token = self.get_language_tokens(user_idx, item_idx, u_rep_local, i_rep_local)
                    exp_cls_token, exp_behave_token = self.get_cls_tokens(user_idx, item_idx, u_rep_global, i_rep_global)
                    # token_loss = self.get_ras_loss(user_lan_token, item_lan_token)
                    graph_loss = loss_vq + recon_loss
                    masked_loss = self.masked_token_model(user_lan_token, user_behave_token) + self.masked_token_model(item_lan_token, item_behave_token)
                    if exp_cls_token is None:
                        cls_loss = 0
                    else:
                        cls_loss = self.cal_loss_lw(exp_cls_token, exp_behave_token)
                    align_loss = graph_loss + masked_loss + 0.2 * cls_loss #+ token_loss
                    optimizer_model.zero_grad()
                    align_loss.backward()
                    optimizer_model.step()
                    batch_index += 1
                    if batch_index % 10 == 0:
                        print(f"Epoch {epoch} Batch {batch_index} Loss: {graph_loss.item()} = CLS_loss: {cls_loss}" \
                         f"VQ loss: {loss_vq.item()} + Masked_loss loss: {masked_loss.item()} + Recon loss: {recon_loss.item()} + Token loss: {0}")
                    # break
                print("Now at Epoch", epoch)
                if epoch % 1 == 0:
                    self.user_emb, self.item_emb = rec_user_emb, rec_item_emb
                    metrics = self.fast_evaluation(epoch)
                    now_hr = float([item.split(':')[1] for item in metrics if 'Hit Ratio' in item][0])
                    # 如果当前 Hit Ratio 优于最佳值，更新最佳值并重置早停计数器
                    if now_hr > self.best_hr:
                        self.best_hr = now_hr
                        self.best_user_emb = torch.cat([u_rep_global, u_rep_local], dim=1)
                        self.best_item_emb = torch.cat([i_rep_global, i_rep_local], dim=1)
                        self.best_u_indices = torch.cat([u_indices_global, u_indices_local], dim=1) 
                        self.best_i_indices = torch.cat([i_indices_global, i_indices_local], dim=1) 
                        self.no_improvement_epochs = 0  # 重置计数器
                    else:
                        self.no_improvement_epochs += 1

                    # Save embeddings to relative path
                    repre_dir = os.path.join(project_root, 'data', args.dataset, 'repre', args.task_name)
                    self._ensure_dir(repre_dir)
                    self.save_embeddings()
                    if self.no_improvement_epochs >= 5:
                        print(f"Early stopping at epoch {epoch} due to no improvement in Hit Ratio for 15 epochs.")
                        break

                model_save_path = os.path.join(repre_dir, 'model.pth')
                torch.save(model, model_save_path)
        # writer.close()

    def kl_loss(self, P, EPSILON=1e-10):
        num_token, token_dim = P.shape
        P = P / (P.sum(dim=1, keepdim=True) + EPSILON)  # 对每一行进行归一化
        Q = torch.full((num_token, token_dim), 1 / token_dim).to(self.device)  # 均匀分布
        kl_div = torch.sum(P * torch.log((P + EPSILON) / (Q + EPSILON)), dim=1)  # 按行求和
        return torch.mean(kl_div)  # 计算平均 KL 散度

    def get_ras_loss(self, user_lan_token, item_lan_token):
        # ===== 对 user 做采样 =====
        B_u, seq_len_u, token_dim = user_lan_token.size()
        B_i, seq_len_i, token_dim = item_lan_token.size()

        num_tokens_to_sample = 2  # 只采2个

        rand_positions_u = torch.randperm(seq_len_u, generator=torch.manual_seed(2025)).unsqueeze(0).repeat(B_u, 1).to(user_lan_token.device)
        sampled_indices_u = rand_positions_u[:, :num_tokens_to_sample]  # [B_u, 2]
        
        rand_positions_i = torch.randperm(seq_len_i, generator=torch.manual_seed(2025)).unsqueeze(0).repeat(B_i, 1).to(user_lan_token.device)
        sampled_indices_i = rand_positions_i[:, :num_tokens_to_sample]

        # 用 gather 提取两个位置
        sampled_user_tokens = user_lan_token.gather(dim=1, index=sampled_indices_u.unsqueeze(-1).expand(-1, -1, token_dim))
        sampled_item_tokens = item_lan_token.gather(dim=1, index=sampled_indices_i.unsqueeze(-1).expand(-1, -1, token_dim))

        # ===== 调用 ras loss =====
        u_cos_sim = self.model.disenQuan_u_local.cal_ras_loss(sampled_user_tokens)
        i_cos_sim = self.model.disenQuan_i_local.cal_ras_loss(sampled_item_tokens)
        user_mask = (u_cos_sim < (1 - 1e-4))
        item_mask = (i_cos_sim < (1 - 1e-4))
        user_lan_token = sampled_user_tokens[user_mask]
        user_lan_sim = F.cosine_similarity(user_lan_token[:, 0, :], user_lan_token[:, 1, :], dim=-1)
        item_lan_token = sampled_item_tokens[item_mask]
        item_lan_sim = F.cosine_similarity(item_lan_token[:, 0, :], item_lan_token[:, 1, :], dim=-1)
        if item_lan_sim.shape[0] == 0 and user_lan_sim.shape[0] == 0:
            return 0
        if item_lan_sim.shape[0] == 0:
            return torch.norm(u_cos_sim[user_mask] - user_lan_sim, p=2, dim=-1) / user_lan_sim.shape[0]
        if user_lan_sim.shape[0] == 0:
            return torch.norm(i_cos_sim[item_mask] - item_lan_sim, p=2, dim=-1) / item_lan_sim.shape[0]
        return torch.norm(u_cos_sim[user_mask] - user_lan_sim, p=2, dim=-1) / user_lan_sim.shape[0] + torch.norm(i_cos_sim[item_mask] - item_lan_sim, p=2, dim=-1) / item_lan_sim.shape[0]

    def cal_loss_lw(self, emb1, emb2, temp=0.2):
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def save(self):
        with torch.no_grad():
            _ = self.model()

    def save_embeddings(self):
        repre_dir = os.path.join(project_root, 'data', args.dataset, 'repre', args.task_name)
        self._ensure_dir(repre_dir)

        torch.save(self.best_user_emb[self.reformat_user],
                   os.path.join(repre_dir, 'user_emb_token.pt'))
        torch.save(self.best_item_emb[self.reformat_item],
                   os.path.join(repre_dir, 'item_emb_token.pt'))
        torch.save(self.best_u_indices[self.reformat_user],
                   os.path.join(repre_dir, 'u_indices.pt'))
        torch.save(self.best_i_indices[self.reformat_item],
                   os.path.join(repre_dir, 'i_indices.pt'))

        torch.save(self.model.disenQuan_u_local._embedding.weight,
                   os.path.join(repre_dir, 'codebook_user_local.pt'))
        torch.save(self.model.disenQuan_u_global._embedding.weight,
                   os.path.join(repre_dir, 'codebook_user_global.pt'))
        torch.save(self.model.disenQuan_i_local._embedding.weight,
                   os.path.join(repre_dir, 'codebook_item_local.pt'))
        torch.save(self.model.disenQuan_i_global._embedding.weight,
                   os.path.join(repre_dir, 'codebook_item_global.pt'))

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()

    def load_model(self):
        from collections import Counter

        repre_dir = os.path.join(project_root, 'data', args.dataset, 'repre', args.task_name)

        user_token = torch.load(os.path.join(repre_dir, 'user_emb_token.pt'))
        item_token = torch.load(os.path.join(repre_dir, 'item_emb_token.pt'))
        u_indices = torch.load(os.path.join(repre_dir, 'u_indices.pt'))
        i_indices = torch.load(os.path.join(repre_dir, 'i_indices.pt'))
        # print(u_indices.shape)
        flat_array_u = u_indices.detach().cpu().numpy()
        flat_array_i = i_indices.detach().cpu().numpy()

        def calculate_distribution(flat_array):
            element_counter = Counter(flat_array.flatten())  
            
            # 分析结果  
            print("Indices Distri:", element_counter)

            avg = 0
            for i in range(flat_array.shape[0]):
                user_indices = flat_array[i]
                all_ele = set(user_indices)
                avg += len(all_ele)
                # if len(all_ele) < 5:
                    # print("no")
            avg = avg / (flat_array.shape[0] * flat_array.shape[1])
            return avg

        avg_u = calculate_distribution(flat_array_u)
        avg_i = calculate_distribution(flat_array_i)
        print("avg_u:", avg_u)
        print("avg_i:", avg_i)


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    from roberta_pre import init_model
    import debugpy
    current_time = datetime.now()
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 5],
                        help='Number of neighbors to sample at each layer')
    parser.add_argument('--task_name', type=str, default='new_global_explict', help='tensorboard Prefix')
    parser.add_argument('--device', type=str, default='0', help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--SWAttn', type=str, default='False', help='Use Sentence-Weighted Attention')
    parser.add_argument('--WWAttn', type=str, default='False', help='Use Word-Wise Attention')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10240, help='learning rate')
    parser.add_argument('--epochs', type=int, default=201, help='learning rate')
    parser.add_argument('--token_len', type=int, default=5, help='tensorboard Prefix')
    parser.add_argument('--token_dim', type=int, default=64, help='tensorboard Prefix')
    parser.add_argument('--codebook_size', type=int, default=512, help='tensorboard Prefix')
    parser.add_argument('--dataset', type=str, default="amazon", help='tensorboard Prefix')
    parser.add_argument('--zero_rate', type=float, default=0, help='tensorboard Prefix')
    args = parser.parse_args()
    epochs = args.epochs
    model = 'LightGCN'
    dataset = args.dataset
    conf = {  
    'model.name': 'LightGCN',  
    'model.type': 'graph',  
    'item.ranking': '-topN 10,20,50',  
    'embedding.size': 64,  
    'num.max.epoch': 501,  
    'batch_size': 10240,  
    'learnRate': 0.001,  
    'reg.lambda': 0.0001,  
    'LightGCN': '-n_layer 2 -dataset amazon',  
    'output.setup': '-dir ./results/'
    }

    # Use relative paths for train and test data
    data_dir = os.path.join(project_root, 'data', dataset)
    train_path = os.path.join(data_dir, 'train.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    conf["training.set"] = train_path
    conf["test.set"] = test_path
    training_data = FileIO.load_data_set(conf['training.set'], conf['model.type'])
    test_data = FileIO.load_data_set(conf['test.set'], conf['model.type'])
    text_batch_size = 128
    DEVICE = f'cuda:{args.device}'
    #################################################
    args.output_dim = args.token_len * args.token_dim
    #################################################
    text_model, text_tokenizer, word_embeddings, mask_code, train_dataloader = init_model(DEVICE, text_batch_size, dataset=dataset)

    Trainers = Trainer(conf=conf, training_set=training_data, test_set=test_data, device=DEVICE, args=args, token_dim=args.token_dim, word_embeddings=word_embeddings,
                       output_dim=args.output_dim, text_model=text_model, train_dataloader=train_dataloader, llm_dim=word_embeddings.shape[1], mask_code=mask_code)

    Trainers.train()
    # Trainers.load_model()
