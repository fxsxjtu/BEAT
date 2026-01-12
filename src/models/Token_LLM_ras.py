import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
import torch.nn.functional as F  
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import os
class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """

    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin((self.dropout(x) - self.bias.to(x.device)).to(self.lin.weight.device))


class MLPAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.adapter(x)


class wrsLayer(nn.Module):

    def __init__(self, quant_dim, text_dim):
        super().__init__()
        self.fc = nn.Linear(quant_dim, text_dim)
        # self.gelu = nn.GELU()
        # self.ln = nn.LayerNorm(text_dim)
        # self.bias = nn.Parameter(torch.zeros(1, 1, text_dim))
        torch.nn.init.normal_(self.fc.weight, std=0.01)


    def forward(self, x):
        x = self.fc(x)
        # x = self.gelu(x)
        # output = self.ln(x)
        # logits = logits + self.bias
        return x

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """

    def __init__(self, n_exps=8, layers=[64, 4096], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate.to(x.device)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise.to(x.device)
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)  # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]  # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs.to(x.device)
        return multiple_outputs.sum(dim=-2)


def compute_kl_divergence(input_dir, standard_distri, epsilon=1e-8):  
    input_dir = input_dir / (input_dir.sum(dim=1, keepdim=True) + epsilon)  
    standard_distri = standard_distri / (standard_distri.sum(dim=1, keepdim=True) + epsilon)  
    
    input_dir = torch.clamp(input_dir, min=epsilon, max=1.0)  
    standard_distri = torch.clamp(standard_distri, min=epsilon, max=1.0)  
    
    kl_divs = input_dir * (torch.log(input_dir + epsilon) - torch.log(standard_distri + epsilon))  
    
    kl_divs = torch.nan_to_num(kl_divs, nan=0.0, posinf=0.0, neginf=0.0)  
    
    return kl_divs.mean() 

class Explainer(torch.nn.Module):
    def __init__(self, model_name="/home/fengxs/LargeModels/", args=None, token_size=4096, user_embed_size=64, item_embed_size=64):
        super(Explainer, self).__init__()
        from huggingface_hub import login
        self.args = args

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="cuda",  
            load_in_8bit=True,  
            trust_remote_code=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,  
            trust_remote_code=False,
            model_max_length=4096
        )
        # add special tokens for user and item embeddings
        special_tokens_dict = {"additional_special_tokens": ["<USER_EMBED>", "<ITEM_EMBED>", "<EXPLAIN_POS>"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token = "<pad>"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.user_embed_size = args.user_embed_size
        self.item_embed_size = args.item_embed_size
        self._init_indices_()
        self.token_type_embedding = nn.Embedding(2, args.user_embed_size)

        if args.moe is True:
            self.user_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[args.user_embed_size, self.model.config.hidden_size], dropout=0.1,
                                                        noise=False)
            self.item_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[args.item_embed_size, self.model.config.hidden_size], dropout=0.1,
                                                        noise=False)
        else:
            self.user_embedding_converter = MLPAdapter(
                input_dim=args.user_embed_size,
                output_dim=self.model.config.hidden_size,
                hidden_dim=(args.user_embed_size + self.model.config.hidden_size) // 2,
            )
            self.item_embedding_converter = MLPAdapter(
                input_dim=args.item_embed_size,
                output_dim=self.model.config.hidden_size,
                hidden_dim=(args.item_embed_size + self.model.config.hidden_size) // 2,
            )
        # freeze parameters in llama
        for param in self.model.parameters():
            param.requires_grad = False

    def wrsrelation_sup(self, quant_user, quant_item, all_text_features, mask_padding):

        _, _, D = all_text_features.shape

        quant_user = self.user_embedding_converter(quant_user).half()                       # [B, G², D]
        quant_item = self.item_embedding_converter(quant_item).half()   

        quant_tokens = torch.cat([quant_user, quant_item], dim=1)  # [B, 2G², D]

        all_text_features = all_text_features / torch.norm(all_text_features, dim=-1, keepdim=True)
        graph_token_norm = quant_tokens / torch.norm(quant_tokens, dim=-1, keepdim=True)

        token_text_sim = torch.matmul(all_text_features, graph_token_norm.half().transpose(1, 2))  # [B, S, G²]
        value, indices = token_text_sim.max(dim=-1)  # value: [B, S], indices: [B, S]

        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)  # [B, S, D]
        text_to_quant = quant_tokens.gather(dim=1, index=indices_exp)  # [B, S, D]

        all_text_features = all_text_features * mask_padding.unsqueeze(-1)
        q = torch.matmul(all_text_features, all_text_features.transpose(1, 2))  # [B, S, S]

        text_to_quant = text_to_quant * value.unsqueeze(-1)           # [B, S, D]
        text_to_quant = text_to_quant / (torch.norm(text_to_quant, dim=-1, keepdim=True) + 1e-6)
        text_to_quant = text_to_quant * mask_padding.unsqueeze(-1)    
        p = torch.matmul(text_to_quant, text_to_quant.transpose(1, 2))  # [B, S, S]

        relation_loss = nn.MSELoss()(p, q.detach())

        return relation_loss, quant_user.half(), quant_item.half()

    def _init_indices_(self):
        self.vocab_emb = self.model.get_input_embeddings().weight.mean(0)
        token_num = 512
       
        self.user_prototype_local = torch.load(f"../data/{self.args.dataset}/repre/{self.args.load_name}/codebook_user_local.pt", map_location="cuda")
        self.user_prototype_global = torch.load(f"../data/{self.args.dataset}/repre/{self.args.load_name}/codebook_user_global.pt", map_location="cuda")
        self.item_prototype_local = torch.load(f"../data/{self.args.dataset}/repre/{self.args.load_name}/codebook_item_local.pt", map_location="cuda")
        self.item_prototype_global = torch.load(f"../data/{self.args.dataset}/repre/{self.args.load_name}/codebook_item_global.pt", map_location="cuda")
        
        self.vocab_user_local = torch.nn.Parameter(self.user_prototype_local, requires_grad=True)
        self.vocab_user_global = torch.nn.Parameter(self.user_prototype_global, requires_grad=True)

        self.vocab_item_local = torch.nn.Parameter(self.item_prototype_local, requires_grad=True)
        self.vocab_item_global = torch.nn.Parameter(self.item_prototype_global, requires_grad=True)
            
    def forward(self, user_embedding, item_embedding, user_indices, item_indices, input_text):
        # Convert embeddings
        global_user_indices = user_indices[:, :1]
        global_item_indices = item_indices[:, :1]
        local_user_indices = user_indices[:, 1:]
        local_item_indices = item_indices[:, 1:]

        # user_token_emb = torch.cat([self.vocab_user_global[global_user_indices], self.vocab_user_local[local_user_indices]], dim=1)
        # item_token_emb = torch.cat([self.vocab_item_global[global_item_indices], self.vocab_item_local[local_item_indices]], dim=1)
        global_user_emb = self.vocab_user_global[global_user_indices]   # [B, 1, D]
        local_user_emb = self.vocab_user_local[local_user_indices]      # [B, N1, D]
        user_token_emb = torch.cat([global_user_emb, local_user_emb], dim=1)

        global_item_emb = self.vocab_item_global[global_item_indices]   # [B, 1, D]
        local_item_emb = self.vocab_item_local[local_item_indices]      # [B, N2, D]
        item_token_emb = torch.cat([global_item_emb, local_item_emb], dim=1)
        user_token_num = user_token_emb.shape[1]
        item_token_num = item_token_emb.shape[1]
        # shape of tokenized_inputs['input_ids']: [batch_size, input_length]
        tokenized_inputs = self.tokenizer(
            input_text, padding=True, return_tensors="pt"
        )
        # Convert tokenized input IDs to model's embeddings
        
        # Get the token ID for the <USER_EMBED> <ITEM_EMBED> token
        
        user_embed_token_id = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        item_embed_token_id = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")
        # Find the position of the <USER_EMBED> <ITEM_EMBED> <EXPLAIN_POS> token in the input embeddings
        # shape of explain_pos_position: [batch_size]
        mask_user = (tokenized_inputs['input_ids'] == user_embed_token_id)
        user_token_enhanced_inputs = self.repeat_tokens(tokenized_inputs['input_ids'], mask_user, user_token_num)
        # print(user_token_enhanced_inputs.shape)
        mask_item = (user_token_enhanced_inputs == item_embed_token_id)
        tokenized_inputs_ids = self.repeat_tokens(user_token_enhanced_inputs, mask_item, item_token_num)

        user_token_enhanced_attn_mask = self.repeat_tokens(tokenized_inputs['attention_mask'], mask_user, user_token_num)
        # print(user_token_enhanced_inputs.shape)
        tokenized_attn_mask = self.repeat_tokens(user_token_enhanced_attn_mask, mask_item, item_token_num)
        # print(self.tokenizer.decode(tokenized_inputs_ids[0]))
        # print(tokenized_inputs_ids.shape, "tokenized_inputs_ids")
        self.user_embed_position = (tokenized_inputs_ids == user_embed_token_id).nonzero()[:, 1:].reshape(-1, user_token_num)
        self.item_embed_position = (tokenized_inputs_ids == item_embed_token_id).nonzero()[:, 1:].reshape(-1, item_token_num)
        explain_pos_position = (tokenized_inputs_ids == explain_pos_token_id).nonzero()[:, 1:]
        inputs_embeds = self.model.get_input_embeddings()(tokenized_inputs_ids).cuda()
        interval = torch.arange(inputs_embeds.shape[1])
        mask = (interval[None, :] > explain_pos_position[:, None]).squeeze() & tokenized_attn_mask
        relation_loss, quant_user, quant_item = self.wrsrelation_sup(quant_user=user_token_emb, quant_item=item_token_emb, all_text_features=inputs_embeds, mask_padding=mask.cuda())
        batch_size, num_pos = self.user_embed_position.shape  # (8, 5)
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, num_pos)  # [8, 5]
        inputs_embeds[batch_idx, self.user_embed_position] = quant_user
        inputs_embeds[batch_idx, self.item_embed_position] = quant_item
        # shape of outputs.logits: [batch_size, input_length, vocab_size]
        outputs = self.model(inputs_embeds=inputs_embeds)
        #self.kl_loss = compute_kl_divergence(self.learnable_vocab_user, standard_distri=self.vocab_emb.unsqueeze(0)) + compute_kl_divergence(self.learnable_vocab_item, standard_distri=self.vocab_emb.unsqueeze(0))
        return tokenized_inputs_ids, outputs, explain_pos_position.flatten(), relation_loss
        # return tokenized_inputs['input_ids'], outputs, explain_pos_position.flatten(), torch.tensor(0).cuda()

    def repeat_tokens(self, input_ids, mask, k):    

        cat_input_ids = []
        for i in range(input_ids.size(0)):  
            repeat_position = torch.where(mask[i])[0]
            token_id = input_ids[i][repeat_position]

            cat_input_id = torch.cat([input_ids[i][:repeat_position], token_id.repeat(k), input_ids[i][repeat_position+1:]], dim=0)
            cat_input_ids.append(
                cat_input_id
            )

        return torch.stack(cat_input_ids)
    
    def repeat_attention_mask(self, input_ids, mask, k):    

        cat_input_ids = []
        for i in range(input_ids.size(0)):  
            repeat_position = torch.where(mask[i])[0]
            token_id = input_ids[i][repeat_position]

            cat_input_id = torch.cat([input_ids[i][:repeat_position], token_id.repeat(k), input_ids[i][repeat_position+1:]], dim=0)
            cat_input_ids.append(
                cat_input_id
            )

        return torch.stack(cat_input_ids)

    def loss(self, input_ids, outputs, explain_pos_position):
        '''
        input_ids: [batch_size, input_length]
        outputs.logits: [batch_size, input_length, vocab_size]
        explain_pos_position: [batch_size]
        '''
        # freeze the information

        logits = outputs.logits
        # pad_token_id=self.tokenizer.pad_token_id
        interval = torch.arange(input_ids.shape[1])
        mask = interval[None, :] < explain_pos_position[:, None]

        input_ids[mask] = -100
        # input_ids[input_ids == pad_token_id] = -100
        # Shift target_ids to the right to create labels; the last token is ignored in the targets.
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = nn.CrossEntropyLoss()(shift_logits, shift_labels.to(shift_logits.device))
        # print(loss, self.kl_loss)
        return loss 

    def generate(self, user_embedding, item_embedding,  user_indices, item_indices, input_text):
        # Convert embeddings
        global_user_indices = user_indices[:, :1]
        global_item_indices = item_indices[:, :1]
        local_user_indices = user_indices[:, 1:]
        local_item_indices = item_indices[:, 1:]
        user_token_emb = torch.cat([self.vocab_user_global[global_user_indices], self.vocab_user_local[local_user_indices]], dim=1)
        item_token_emb = torch.cat([self.vocab_item_global[global_item_indices], self.vocab_item_local[local_item_indices]], dim=1)
        user_token_num = user_token_emb.shape[1]
        item_token_num = item_token_emb.shape[1]
        if not self.args.indice:
            converted_user_embedding = self.user_embedding_converter(user_token_emb).half()#.reshape(-1, self.model.config.hidden_size)
            converted_item_embedding = self.item_embedding_converter(item_token_emb).half()#.reshape(-1, self.model.config.hidden_size)
        else:
            user_token_indices = user_indices.reshape(user_embedding.shape[0], -1)
            item_token_indices = item_indices.reshape(item_embedding.shape[0], -1)
            user_prototype = self.learnable_vocab_user[user_token_indices].to(user_embedding.device)
            item_prototype = self.learnable_vocab_item[item_token_indices].to(item_embedding.device)
            # print("user_emb shape", user_embedding.shape, "user_prototype shape", user_prototype.shape)
            converted_user_embedding = self.user_prototype_fusion(user_token_emb.float(), user_prototype.float()).half()
            converted_item_embedding = self.item_prototype_fusion(item_token_emb.float(), item_prototype.float()).half()
        # shape of tokenized_inputs['input_ids']: [batch_size, input_length]
        tokenized_inputs = self.tokenizer(
            input_text, padding=True, return_tensors="pt"
        )
        # Convert tokenized input IDs to model's embeddings
        # Get the token ID for the <USER_EMBED> <ITEM_EMBED> token
        user_embed_token_id = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        item_embed_token_id = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")
        # Find the position of the <USER_EMBED> <ITEM_EMBED> <EXPLAIN_POS> token in the input embeddings
        # shape of explain_pos_position: [batch_size]
        mask_user = (tokenized_inputs['input_ids'] == user_embed_token_id)
        user_token_enhanced_inputs = self.repeat_tokens(tokenized_inputs['input_ids'], mask_user, user_token_num)
        # print(user_token_enhanced_inputs.shape)
        mask_item = (user_token_enhanced_inputs == item_embed_token_id)
        tokenized_inputs_ids = self.repeat_tokens(user_token_enhanced_inputs, mask_item, item_token_num)

        user_token_enhanced_attn_mask = self.repeat_tokens(tokenized_inputs['attention_mask'], mask_user, user_token_num)
        # print(user_token_enhanced_inputs.shape)
        tokenized_attn_mask = self.repeat_tokens(user_token_enhanced_attn_mask, mask_item, item_token_num)
        # print(tokenized_inputs_ids.shape, "tokenized_inputs_ids")
        user_embed_position = (tokenized_inputs_ids == user_embed_token_id).nonzero()[:, 1:].reshape(-1, user_token_num)
        item_embed_position = (tokenized_inputs_ids == item_embed_token_id).nonzero()[:, 1:].reshape(-1, item_token_num)
        explain_pos_position = (tokenized_inputs_ids == explain_pos_token_id).nonzero()[:, 1:]
        inputs_embeds = self.model.get_input_embeddings()(tokenized_inputs_ids).to(
            converted_user_embedding.device)
        batch_size, num_pos = user_embed_position.shape  # (8, 5)
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, num_pos)  # [8, 5]
        inputs_embeds[batch_idx, user_embed_position] = converted_user_embedding
        inputs_embeds[batch_idx, item_embed_position] = converted_item_embedding
        # shape of outputs.logits: [batch_size, input_length, vocab_size]
        outputs = self.model.generate(inputs_embeds=inputs_embeds, attention_mask=tokenized_attn_mask, max_new_tokens=128)
        output_text = self.tokenizer.batch_decode(outputs , skip_special_tokens=True)
        return output_text


if __name__ == "__main__":
    model = Explainer()
    for name, param in model.model.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    for name, param in model.user_embedding_converter.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    for name, param in model.item_embedding_converter.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
