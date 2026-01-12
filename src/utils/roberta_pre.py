from transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
import json
# from arg import parse_configure
from typing import Dict, Sequence
import copy
from dataclasses import dataclass
# from data_loader import DataHandler
from torch.utils.data import DataLoader
from tqdm import tqdm
# args = parse_configure()

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim


    def set_codebook(self, codebook):
        self._embedding = codebook
        self._num_embeddings = codebook.shape[0]

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + 0.5 * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized.squeeze(), loss


def load_model(model_directory, device):
    tokenizer = RobertaTokenizer.from_pretrained(model_directory)
    model = RobertaForMaskedLM.from_pretrained(model_directory, output_hidden_states=True).to(device)
    word_embeddings = model.roberta.embeddings.word_embeddings.weight.data

    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    mask_code = word_embeddings[mask_token_id]
    return model, tokenizer, word_embeddings, mask_code


# def process_interest()


# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20

def init_model(DEVICE, batch_size, dataset):
    # DEVICE = 'cuda:3'
    # batch_size = 4
    model, tokenizer, word_embeddings, mask_code = load_model('roberta-base',
                                                              DEVICE)
    data_loader = DataHandler(args=args)
    return model, tokenizer, word_embeddings, mask_code, data_loader

def get_cls_token(trn_data, dataset):
    batch_size = 1024
    total = len(trn_data['explanation'])
    dict_save = {}

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        
        texts = trn_data['explanation'][start:end].tolist()
        uids = trn_data['uid'][start:end].tolist()
        iids = trn_data['iid'][start:end].tolist()

        # 1. Tokenize
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 2. Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        last_hidden = outputs['hidden_states'][-1]         # [batch_size, seq_len, hidden_dim]
        cls_embeddings = last_hidden[:, 0, :]              # [batch_size, hidden_dim]

        # 3. Detach + move to CPU + convert to list to save memory
        cls_embeddings = cls_embeddings.detach().cpu()

        # 4. Save directly into dict (no accumulation in GPU)
        for uid, iid, cls_vec in zip(uids, iids, cls_embeddings):
            dict_save[(uid, iid)] = cls_vec  # or cls_vec.tolist() if you want to save in JSON style

        print(f"Processed batch {start} ~ {end}")

    # 5. Save dict to pickle
    with open(f'../data/{dataset}/cls_token.pkl', 'wb') as f:
        pickle.dump(dict_save, f)


def init_model(DEVICE, batch_size, dataset):

    model, tokenizer, word_embeddings, mask_code = load_model('roberta-base',
                                                              DEVICE)
    data_loader = 0
    return model, tokenizer, word_embeddings, mask_code, data_loader

def get_cls_token_pt(trn_data, dataset):
    batch_size = 512
    total = len(trn_data['explanation'])
    dict_save = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        
        texts = trn_data['explanation'][start:end].tolist()
        uids = trn_data['uid'][start:end].tolist()
        iids = trn_data['iid'][start:end].tolist()

        # 1. Tokenize
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 2. Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        last_hidden = outputs['hidden_states'][-1]         # [batch_size, seq_len, hidden_dim]
        cls_embeddings = last_hidden[:, 0, :]              # [batch_size, hidden_dim]

        # 3. Detach + move to CPU + convert to list to save memory
        cls_embeddings = cls_embeddings.detach().cpu()

        dict_save.append(cls_embeddings)
          # or cls_vec.tolist() if you want to save in JSON style

        print(f"Processed batch {start} ~ {end}")


    torch.save(torch.cat(dict_save, dim=0), f"../data/{dataset}/cls_token.pt")

def encode_user_interests(dataset, role, model_name="roberta-base", device="cuda"):
    with open(f"../data/{dataset}/{role}_disentangled_interest.json", "r", encoding="utf-8") as f:
        user_interest_dict = json.load(f)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)
    model.eval()

    hidden_dim = model.config.hidden_size
    user_embeddings = {}

    for uid, texts in tqdm(user_interest_dict.items()):
        # Step 1: 截断或补齐到 10 个兴趣文本
        if len(texts) > 10:
            texts = texts[:10]
        elif len(texts) < 10:
            texts += [""] * (10 - len(texts))  # 空文本将生成 0 向量（或几乎为 0）

        # Step 2: 编码
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs['hidden_states'][-1]  # shape: [batch_size, seq_len, hidden_dim]
            cls_embeddings = last_hidden[:, 0, :]       # shape: [batch_size, hidden_dim]

        # Step 3: 保存到字典中
        user_embeddings[uid] = cls_embeddings.cpu()  # shape: [10, hidden_dim]

    with open(f'../data/{dataset}/{role}_interest_token.pkl', 'wb') as f:
        pickle.dump(user_embeddings, f)



if __name__ == "__main__":
    DEVICE = 'cuda'
    batch_size = 4
    data_path = '../data/amazon/trn.pkl'
    st_data_path_user = "../data/amazon/user_emb.pkl"
    st_data_path_item ="../data/amazon/item_emb.pkl"
    model, tokenizer, word_embeddings, mask_code = load_model('roberta-base', DEVICE)
    text = ["Hello, how are you?"]
    import debugpy
    import argparse
    import pickle
    from collections import defaultdict

    # === 添加 argparse 接收 dataset 参数 ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset name, e.g. yelp")
    args = parser.parse_args()

    dataset = args.dataset
    with open(f"../data/{dataset}/trn.pkl", "rb") as file:
        trn_data = pickle.load(file)

    get_cls_token_pt(trn_data, dataset=dataset)
    encode_user_interests(dataset=dataset, role="user")
    encode_user_interests(dataset=dataset, role="item")