import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from arg import args
from typing import List
import random
import torch.nn.functional as F
import os 
class TextDataset(Dataset):
    def __init__(self, input_text: List[str]):
        self.input_text = input_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        return self.input_text[idx]


class DataHandler:
    def __init__(self, args):
        if args.dataset == "amazon":
            self.system_prompt = "Explain why the user would buy with the book within 50 words."
            self.item = "book"
        elif args.dataset == "yelp" or args.dataset == "google":
            self.system_prompt = "Explain why the user would enjoy the business within 50 words."
            self.item = "business"

        user_emb_path = f"./{args.dataset}/repre/{args.load_name}/user_emb_token.pt"
        item_emb_path = f"./{args.dataset}/repre/{args.load_name}/item_emb_token.pt"
        user_indice_path = f"./{args.dataset}/repre/{args.load_name}/u_indices.pt"
        item_indice_path = f"./{args.dataset}/repre/{args.load_name}/i_indices.pt"

        self.user_indices = torch.load(user_indice_path, map_location="cuda")
        self.item_indices = torch.load(item_indice_path, map_location="cuda")

        self.user_emb = torch.load(user_emb_path, map_location="cuda")
        self.item_emb = torch.load(item_emb_path, map_location="cuda")
        self.if_profile = args.if_profile
        self.user_num = self.user_emb.shape[0]
        self.item_num = self.item_emb.shape[0] 

        self.user_embed_size = args.token_dim
        self.item_embed_size = args.token_dim
        self.args = args

    def load_data(self):
        # load data from data_loaders in data

        with open(f"./{args.dataset}/zeroshot_train.pkl", "rb") as file:
            trn_data = pickle.load(file)
            trn_dict = trn_data.to_dict("list")

        with open(f"./{args.dataset}/val.pkl", "rb") as file:
            val_data = pickle.load(file)
        with open(f"./{args.dataset}/tst.pkl", "rb") as file:
            tst_data = pickle.load(file)

        # convert data into dictionary
        
        val_dict = val_data.to_dict("list")
        tst_dict = tst_data.to_dict("list")

        # combine all information input input string
        trn_input = []
        val_input = []
        tst_input = []
        for i in range(len(trn_dict["uid"])):
            if self.if_profile:
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {trn_dict['title'][i]} user profile: {trn_dict['user_summary'][i]} {self.item} profile: {trn_dict['item_summary'][i]} <EXPLAIN_POS> {trn_dict['explanation'][i]}"
            else:
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {trn_dict['title'][i]} <EXPLAIN_POS> {trn_dict['explanation'][i]}"
            trn_input.append(
                (
                    self.user_emb[trn_dict["uid"][i]],
                    self.item_emb[trn_dict["iid"][i]],
                    self.user_indices[trn_dict["uid"][i]],
                    self.item_indices[trn_dict["iid"][i]],
                    i,
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
                )
            )
        for i in range(len(val_dict["uid"])):
            if self.if_profile:
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {val_dict['title'][i]} user profile: {val_dict['user_summary'][i]} {self.item} profile: {val_dict['item_summary'][i]} <EXPLAIN_POS>"
            else:
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {val_dict['title'][i]} <EXPLAIN_POS>"
            val_input.append(
                (
                    self.user_emb[val_dict["uid"][i]],
                    self.item_emb[val_dict["iid"][i]],
                    self.user_indices[val_dict["uid"][i]],
                    self.item_indices[val_dict["iid"][i]],
                    i,
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                    val_dict['explanation'][i],
                )
            )
        for i in range(len(tst_dict["uid"])):
            if self.if_profile:
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {tst_dict['title'][i]} user profile: {tst_dict['user_summary'][i]} {self.item} profile: {tst_dict['item_summary'][i]} <EXPLAIN_POS>"
            else:
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {tst_dict['title'][i]} <EXPLAIN_POS>"
            tst_input.append(
                (
                    self.user_emb[tst_dict["uid"][i]],
                    self.item_emb[tst_dict["iid"][i]],
                    self.user_indices[tst_dict["uid"][i]],
                    self.item_indices[tst_dict["iid"][i]],
                    (i, tst_dict["uid"][i], tst_dict["iid"][i]),
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                    tst_dict["explanation"][i],
                )
            )

        # load training batch
        trn_dataset = TextDataset(trn_input)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)

        # load validation batch
        val_dataset = TextDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # load testing batch
        tst_dataset = TextDataset(tst_input)
        tst_loader = DataLoader(tst_dataset, batch_size=args.eval_batch_size, shuffle=True)

        return trn_loader, val_loader, tst_loader


if __name__ == "__main__":
    for dataset in ["amazon", "yelp", "google"]:
        with open(f"./{dataset}/zeroshot_train.pkl", "rb") as file:
            trn_data = pickle.load(file)
            trn_dict = trn_data.to_dict("list")
            print(len(trn_data['uid']))