import json
import torch
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation'))

from Token_LLM_ras import Explainer
from data_loader import DataHandler, DataHandler_Onehot
from arg import args
import time
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from tqdm import tqdm
import re
from evaluation import evaluate_metrics
import wandb



class TrainingManager:
    def __init__(self, args, enable_wandb=True):
        self.enable_wandb = enable_wandb
        if self.enable_wandb != None:
            # Use WANDB_API_KEY environment variable for authentication
            # Set it before running: export WANDB_API_KEY=your_api_key
            wandb_api_key = os.environ.get('WANDB_API_KEY', None)
            if wandb_api_key:
                wandb.login(key=wandb_api_key, relogin=True)

            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'BEAT'),
                name=args.enable_wandb,
                config={
                    "batch_size": args.batch_size,
                    "model": args.model_name,
                    "lr": args.lr,
                    "random": args.random,
                    "if_profile": args.if_profile,
                    "weight_decay": args.weight_decay
                }
            )

    def log_metrics(self, metrics, commit=True):
        """
        根据标志决定是否记录指标

        参数:
        metrics (dict): 要记录的指标
        """
        if self.enable_wandb:
            wandb.log(metrics, commit=commit)

    def finish(self):
        """
        结束 wandb 会话
        """
        if self.enable_wandb:
            wandb.finish()


class TextProcessor:
    def __init__(self):
        pass
    def extract_inst_text(self, text):
        match = re.search(r'\[\\INST\](.*?)([\"\[\]\\/;:!])', text)
        if match:
            return match.group(1).strip()  # 返回提取的文本，并去掉两端空白
        else:
            return None  # 如果没有匹配，返回None

    def clean_text(self, text):

        cleaned = re.sub(r'[^\w\s]', '', text)  # 保留字母、数字和空格
        words = cleaned.split()
        seen = set()
        deduped_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                deduped_words.append(word)
        cleaned_final = ' '.join(deduped_words)
        return cleaned_final

    def process_texts(self, texts):
        extracted_texts = []
        for text in texts:
            result = self.extract_inst_text(text)
            extracted_texts.append(result)
        cleaned_texts = [self.clean_text(text) if text else None for text in extracted_texts]
        return cleaned_texts


class TokenRec:
    def __init__(self, model_path, wanb_logger):
        print(f"dataset: {args.dataset}")
        # Get project root directory (github_relese/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        deepspeed_config = os.path.join(project_root, 'ds_config.json')
        self.accelerator = Accelerator(
            deepspeed_plugin=DeepSpeedPlugin(deepspeed_config)
        )
        self.data_handler = DataHandler(args=args)
        # self.data_handler = DataHandler_Onehot(args=args)
        args.user_num = self.data_handler.user_num
        args.item_num = self.data_handler.item_num 
        args.user_embed_size = self.data_handler.user_embed_size
        args.item_embed_size = self.data_handler.item_embed_size
        self.model = Explainer(model_name=model_path, args=args)
        '''
        DataHandler 需要输入
        
        if_random=False  ---> 是否需要用随机生成的random代替原先的用户表征
        user_emb_path=None  ---> 用户表征文件位置
        item_emb_path=None  ---> 商品表征文件位置
        '''
        self.model_path = model_path

        self.wanb_logger = wanb_logger
        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()

        # Use relative paths based on project root
        data_dir = os.path.join(project_root, 'data', args.dataset)
        convert_params_dir = os.path.join(data_dir, 'convert_params', args.task_name)
        output_text_dir = os.path.join(data_dir, 'output_text')

        if args.indice:
            self.user_embedding_converter_path = os.path.join(convert_params_dir, '')
            self.item_embedding_converter_path = os.path.join(convert_params_dir, '')
        else:
            self.user_embedding_converter_path = os.path.join(convert_params_dir, '')
            self.item_embedding_converter_path = os.path.join(convert_params_dir, '')

        self.tst_predictions_path = os.path.join(output_text_dir, f"tst_predictions_{args.task_name}.pkl")
        self.tst_references_path = os.path.join(output_text_dir, f"tst_references_{args.task_name}.pkl")

    def _format_args(self):
        """
        格式化参数，便于 TensorBoard 显示

        Returns:
            str: 格式化后的参数字符串
        """
        args_dict = vars(args)
        formatted_args = "```json\n" + json.dumps(args_dict, indent=2) + "\n```"
        return formatted_args

    def _ensure_dir(self, file_path):
        """确保文件路径的目录存在，如果不存在则创建它。"""
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def train(self):
        if args.clr == 0:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam([p for p in self.model.parameters() if (p is not self.model.vocab_user_global) and (p is not self.model.vocab_item_global) and (p is not self.model.vocab_user_local) and (p is not self.model.vocab_item_local) ], lr=args.lr) 
            optimizer_codebook = torch.optim.Adam([self.model.vocab_item_local, self.model.vocab_user_local, self.model.vocab_user_global ,self.model.vocab_item_global], lr=args.clr) 
        self.model, optimizer, self.trn_loader = self.accelerator.prepare(
            self.model, optimizer, self.trn_loader
        )
        global_step = 0

        for epoch in range(args.epochs):
            total_loss = 0
            self.model.train()
            start_time = time.time()
            trn_loader = tqdm(self.trn_loader, desc=f"Epoch {epoch + 1}", total=len(self.trn_loader))
            
            for i, batch in enumerate(trn_loader):
                user_embed, item_embed, user_indices, item_indice, exp_index, input_text = batch
                input_ids, outputs, explain_pos_position, relation_loss = self.model.forward(
                    user_embed, item_embed, user_indices, item_indice, input_text
                )
                optimizer.zero_grad()
                # print(args.clr, "Here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if args.clr != 0:
                    optimizer_codebook.zero_grad()
                loss = self.model.loss(input_ids, outputs, explain_pos_position) + args.beta * relation_loss
                self.accelerator.backward(loss)
                optimizer.step()
                if args.clr != 0:
                    optimizer_codebook.step()
                total_loss += loss.item()
                global_step += 1
                trn_loader.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'batch': i
                })
                self.wanb_logger.log_metrics({"Loss/train": loss.item(), "Epoch": epoch})
                if i % 100 == 0 and i != 0 and self.accelerator.is_main_process:
                    elapsed_time = time.time() - start_time
                    print(
                        f"Epoch [{epoch + 1}/{args.epochs}], Step [{i}/{len(self.trn_loader)}], Loss: {loss.item()}"
                    )
                    print(f"Generated Explanation: {outputs[0]}")
                    print(f"Time taken: {elapsed_time / 60:.2f} minutes")
                    # 记录每个 epoch 的平均 loss 到 TensorBoard
            if self.accelerator.is_main_process:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Total Loss: {total_loss}")
                # 使用 unwrap_model 解包模型
                self._ensure_dir(self.user_embedding_converter_path)
                self._ensure_dir(self.item_embedding_converter_path)
                torch.save(
                    self.accelerator.unwrap_model(self.model).user_embedding_converter.state_dict(),
                    self.user_embedding_converter_path + f"user_converter_{epoch}.pkl",
                )
                torch.save(
                    self.accelerator.unwrap_model(self.model).item_embedding_converter.state_dict(),
                    self.item_embedding_converter_path + f"item_converter_{epoch}.pkl",
                )
                torch.save(
                    self.accelerator.unwrap_model(self.model).vocab_user_local,
                    self.user_embedding_converter_path + f"vocab_user_local_{epoch}.pt",  # 推荐使用 .pt 或 .pth 后缀
                )

                # 保存 user_global vocab
                torch.save(
                    self.accelerator.unwrap_model(self.model).vocab_user_global,
                    self.user_embedding_converter_path + f"vocab_user_global_{epoch}.pt",
                )

                # 保存 item_local vocab
                torch.save(
                    self.accelerator.unwrap_model(self.model).vocab_item_local,
                    self.user_embedding_converter_path + f"vocab_item_local_{epoch}.pt",
                )

                # 保存 item_global vocab
                torch.save(
                    self.accelerator.unwrap_model(self.model).vocab_item_global,
                    self.user_embedding_converter_path + f"vocab_item_global_{epoch}.pt",
                )
                print(f"Saved model to {self.user_embedding_converter_path}")
                print(f"Saved model to {self.item_embedding_converter_path}")


    def evaluate(self):
        import json
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        deepspeed_config = os.path.join(project_root, 'ds_config.json')
        accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(deepspeed_config))

        model = Explainer(model_name=self.model_path, args=args)
        for epoch in range(0, args.epochs):
            model.user_embedding_converter.load_state_dict(
                torch.load(self.user_embedding_converter_path + f"user_converter_{epoch}.pkl", map_location="cuda")
            )
            model.item_embedding_converter.load_state_dict(
                torch.load(self.item_embedding_converter_path + f"item_converter_{epoch}.pkl", map_location="cuda")
            )
            with torch.no_grad():
 
                user_local_data = torch.load(self.user_embedding_converter_path + f"vocab_user_local_{epoch}.pt", map_location="cuda")
                model.vocab_user_local.data = user_local_data

                user_global_data = torch.load(self.user_embedding_converter_path + f"vocab_user_global_{epoch}.pt", map_location="cuda")
                model.vocab_user_global.data = user_global_data

                item_local_data = torch.load(self.user_embedding_converter_path + f"vocab_item_local_{epoch}.pt", map_location="cuda")
                model.vocab_item_local.data = item_local_data

                item_global_data = torch.load(self.user_embedding_converter_path + f"vocab_item_global_{epoch}.pt", map_location="cuda")
                model.vocab_item_global.data = item_global_data
            model.eval()
            test_dataloader = self.tst_loader
            model, test_dataloader = accelerator.prepare(model, test_dataloader)

            generated_txt = []
            references = []

            with torch.inference_mode():
                print("Start Evaluating")
                for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    user_embed, item_embed, user_indices, item_indice, exp_index, input_text, explain = batch

                    outputs = model.generate(user_embed, item_embed, user_indices, item_indice, input_text)

                    generated_txt.extend(outputs)
                    references.extend(explain)

            # Save generated text to relative path
            generated_text_dir = os.path.join(project_root, 'outputs', 'generated_text', args.dataset, args.task_name)
            self._ensure_dir(generated_text_dir)
            output_file = os.path.join(generated_text_dir, f"epoch_{epoch}.json")
            with open(output_file, "w") as json_file:
                for i in range(len(generated_txt)):
                    record = {
                        "generated_txt": generated_txt[i],
                        "reference": references[i]
                    }
                    json.dump(record, json_file, indent=4)
                    json_file.write("\n\n")
            

def get_model_path(model_name: str) -> str:
    """
    Get model path based on model name.
    Uses environment variable MODEL_BASE_PATH if set, otherwise uses Hugging Face model names.

    Set custom model path via environment variable:
        export MODEL_BASE_PATH=/path/to/your/models

    Or use Hugging Face model names directly (will be downloaded automatically)
    """
    # Check if custom model base path is set
    base_path = os.environ.get('MODEL_BASE_PATH', None)

    # If custom path is set, use it
    if base_path and os.path.exists(base_path):
        model_map = {
            "llama_8b": os.path.join(base_path, "Meta-Llama-3-8B-Instruct"),
            "qwen_7b": os.path.join(base_path, "Qwen2.5-7B-Instruct"),
            "deepseek_8b": os.path.join(base_path, "DeepSeek-R1-Distill-Llama-8B"),
            "llama_3.2_3b": os.path.join(base_path, "Llama-3.2-3B-Instruct"),
            "skywork_8b": os.path.join(base_path, "Skywork-o1-Open-Llama-3.1-8B"),
            "llama_3.1_8b": os.path.join(base_path, "Llama-3.1-8B-Instruct"),
        }
        return model_map.get(model_name, None)

    # Otherwise, use Hugging Face model names (will auto-download)
    huggingface_models = {
        "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
        "deepseek_8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "llama_3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
        "skywork_8b": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
        "llama_3.1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    }

    return huggingface_models.get(model_name, model_name)




def main():
    import debugpy
    wandb_logger = TrainingManager(args, enable_wandb=args.enable_wandb)
    model_path = get_model_path(args.model_name)
    print(args.model_name)
    sample = TokenRec(model_path, wandb_logger)

    sample.train()
    print("Generating explanations...")
    sample.evaluate()


if __name__ == "__main__":

    main()