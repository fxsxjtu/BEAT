import numpy as np
import json
import concurrent.futures
import re
import subprocess
import unicodedata
from typing import List, Dict  
from collections import Counter 


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        # seq is a list of words
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def BERT_score(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
    )
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1),
        np.std(precision),
        np.std(recall),
        np.std(f1),
    )


class TextEvaluator:
    def __init__(self):
        warnings.filterwarnings('ignore')
        self.language = 'english'
        bleurt_checkpoint = "/your_path/bleurt-large-512"
        self.bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

        print("✅ Evaluator initialized.")
    def evaluate(self, ground_truth_texts, generated_texts):
        assert len(ground_truth_texts) == len(generated_texts), "真值文本和生成文本的数量必须相同。"

        bert_score = evaluation_bert(ground_truth_texts, generated_texts)

        bart_score = evaluation_bart(ground_truth_texts, generated_texts)
        print("processing bleurt_results", flush=True)

        results = {

            'BARTScore': bart_score,
            'BertScore': bert_score,

        }
        return results


def evaluate_metrics(ground_truth_texts, generated_texts):
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import warnings
    from tqdm import tqdm
    from evaluate import load

    warnings.filterwarnings('ignore')

    print("downloaded")
    # 确保输入列表长度一致
    assert len(ground_truth_texts) == len(generated_texts), "真值文本和生成文本的数量必须相同。"
    bert_score = evaluation_bert(ground_truth_texts, generated_texts)
    # 指定语言参数，避免 LookupError
    language = 'english'  # 根据您的文本语言进行调整

    tokenized_ground_truth_texts = [
        [nltk.word_tokenize(text.lower(), language=language)] for text in ground_truth_texts
    ]
    tokenized_generated_texts = [
        nltk.word_tokenize(text.lower(), language=language) for text in generated_texts
    ]

    # 定义平滑函数
    smooth_fn = SmoothingFunction().method1

    # 计算 BLEU-1 分数
    bleu1_score = corpus_bleu(
        tokenized_ground_truth_texts,
        tokenized_generated_texts,
        weights=(1, 0, 0, 0),
        smoothing_function=smooth_fn
    )

    # 计算 BLEU-4 分数
    bleu4_score = corpus_bleu(
        tokenized_ground_truth_texts,
        tokenized_generated_texts,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth_fn
    )

    # 初始化 ROUGE 评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    # 存储 ROUGE 分数
    rouge1_scores = []
    rougeL_scores = []

    for ref, gen in tqdm(zip(ground_truth_texts, generated_texts)):
        scores = scorer.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    # 计算平均 ROUGE 分数
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    # 定义计算 Distinct 指标的函数
    def calculate_distinct_ngrams(sentences, n):
        total_ngrams = 0
        unique_ngrams = set()
        for sentence in sentences:
            tokens = sentence.lower().split()
            ngrams = zip(*[tokens[i:] for i in range(n)])
            ngrams = list(ngrams)
            total_ngrams += len(ngrams)
            unique_ngrams.update(ngrams)
        if total_ngrams == 0:
            return 0.0
        return len(unique_ngrams) / total_ngrams

    # 计算 Distinct-1 和 Distinct-2
    distinct_1 = calculate_distinct_ngrams(generated_texts, 1)
    distinct_2 = calculate_distinct_ngrams(generated_texts, 2)

    bart_score = evaluation_bart(ground_truth_texts, generated_texts)


    results = {
        'BLEU-1': bleu1_score,
        'BLEU-4': bleu4_score,
        'ROUGE-1': avg_rouge1,
        'ROUGE-L': avg_rougeL,
        'Distinct-1': distinct_1,
        'Distinct-2': distinct_2,
        'BARTScore': bart_score,
        'BertScore': bert_score,

    }
    return results


def extract_metrics(text):
    """
    提取 P, R, F1 值（支持负数）。如果任一项缺失，则返回原始文本。
    """
    metrics = {}
    pattern = r'(P|R|F1):\s*(-?[\d.]+)'
    
    for match in re.findall(pattern, text):
        metric_type, value = match
        metrics[metric_type] = float(value)

    required_keys = {'P', 'R', 'F1'}
    if not required_keys.issubset(metrics.keys()):
        return {'raw_text': text}  # 任意一个缺失，返回原文本
    else:
        return metrics

def evaluation_bert(ground_truth_texts, generated_texts):
    import subprocess
    with open('generated.txt', 'w', encoding='utf-8') as f:
        for item1 in generated_texts:
            item1 = item1.replace('\n', ' ').replace('\r', ' ').strip()
            f.write(item1 + '\n')
    with open('groundtruth.txt', 'w', encoding='utf-8') as f:
        for item2 in ground_truth_texts:
            item2 = item2.replace('\n', ' ').replace('\r', ' ').strip()
            f.write(item2 + '\n')
    print(len(generated_texts), len(generated_texts))
    # 执行命令行命令，并等待命令执行结束
    result = subprocess.run(
        ['bert-score', '-r', 'groundtruth.txt', '-c', 'generated.txt', '--lang', 'en', '--rescale_with_baseline', '-b 64'],
        capture_output=True, text=True)
    full_output = result.stdout + "\n" + result.stderr
    return extract_metrics(full_output)


def evaluation_bart(ground_truth_texts, generated_texts):
    import sys
    sys.path.append("/your_path/BARTScore/")
    from bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')#, cache_dir='/mnt/petrelfs/fengxinshun/MM_llama/llama_model/bart_score/BART')
    bart_scorer.load(path='/your_path/bart_score/bart_score.pth')
    bart_score = bart_scorer.score(generated_texts, ground_truth_texts, batch_size=128)
    return np.mean(bart_score)


def post_process_output(output_text):
    # 如果输出包含了 [/INST]，则只保留第一个部分
    if '[/INST]' in output_text:
        output_text = output_text.split('[/INST]', 1)[0]
    # 去除首尾多余的空白字符
    output_text = output_text.strip()
    return output_text


def post_search_content(txt):
    import re
    match = re.search(r"The user would enjoy the business because\.\.\.(.*)", txt, re.DOTALL)
    if match:
        clean_content = match.group(1).strip()
        return clean_content
    else:
        return -1

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


    