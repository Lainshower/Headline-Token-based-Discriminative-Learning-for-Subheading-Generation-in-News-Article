import torch
import numpy as np
from tqdm import *
import pandas as pd
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast,  BartTokenizer, MT5ForConditionalGeneration, T5Tokenizer
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", default='yonhapnews', type=str)
parser.add_argument("--headline_col", default='headline', type=str)
parser.add_argument("--subheading_col", default='subheading', type=str)
parser.add_argument("--body_col", default='body', type=str)
parser.add_argument("--encoder_max_len", default=1024, type=int)
parser.add_argument("--generate_max_len", default=95, type=int)
parser.add_argument("--device", default='cuda', type=str)
args = parser.parse_args()


tokenizer = BartTokenizer.from_pretrained(tokenizer_path) # your tokenizer path 
model = BartForConditionalGeneration.from_pretrained(model_path) # yout trained model path
model.cuda()

# Load Infered Data
infer_data  = pd.read_csv(os.path.join('dataset', args.data_type, 'infer.csv'))

# Infering subheading 
infer_data['generated_'+args.subheading_col] = np.nan

for i in tqdm(range(len(infer_data))):
    input_ids = tokenizer.encode(infer_data[args.body_col].iloc[i])
    input_ids = input_ids[:args.encoder_max_len]
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids.to('cuda'), eos_token_id=1, max_length=args.generate_max_len, num_beams=5, no_repeat_ngram_size=5)
    output.detach().cpu
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    infer_data['generated_'+args.subheading_col][i] = output

    print('headline', infer_data[args.headline_col][i])
    print('subheading', infer_data[args.subheading_col][i])
    print('generated', infer_data['generated_'+args.subheading_col][i])

infer_data.to_csv(os.path.join('dataset', args.data_type, 'infer_done.csv'), index=False)
model.cpu()
del model