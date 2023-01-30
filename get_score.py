from cmath import nan
from tqdm import *
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast, T5Tokenizer
from scorer import Score_Calculator
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", default='yonhap_news', type=str)
parser.add_argument("--headline_col", default='headline', type=str)
parser.add_argument("--subheading_col", default='subheading', type=str)
parser.add_argument("--body_col", default='body', type=str)
parser.add_argument("--device", default='cuda', type=str)
args = parser.parse_args()

if args.data_type in ['yonhapnews', 'xlsum_kor']:
    language = 'kor'
else:
    language = 'eng'

if language == 'kor':
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
else:
    tokenizer = PreTrainedTokenizerFast.from_pretrained('facebook/bart-base')

# Load Infered_Done
infer_data  = pd.read_csv(os.path.join('dataset', args.data_type, 'infer_done.csv'))
infer_data.reset_index(drop=True, inplace=True)

score_list = ['ref_gen_r1', 'ref_gen_r2', 'ref_gen_rl', 'ref_gen_bertscore',
                'gen_body_bleu_list', 'gen_body_bleu_max', 'gen_body_bleu_avg',
                'gen_body_r1_list', 'gen_body_r2_list', 'gen_body_rl_list',
                'gen_body_r1_max', 'gen_body_r1_avg', 'gen_body_r2_max', 'gen_body_r2_avg', 'gen_body_rl_max', 'gen_body_rl_avg' 
            ]

for score in score_list:
    infer_data[score] = None

scorer = Score_Calculator(tokenizer=tokenizer, lang=language, device=args.device)
nan_list = list()

for i in tqdm(range(len(infer_data))):
    try:
        instance = infer_data.iloc[i]
        headline = instance[args.headline_col]
        subheading = instance[args.subheading_col]
        body = instance[args.body_col]
        generated_subheading = instance['generated_'+args.subheading_col]
        result_dict = scorer.compute(headline, subheading, body, generated_subheading)

        ref_gen = result_dict['ref_gen']
        gen_body_bleu = result_dict['gen_body_bleu']
        gen_body_rouge = result_dict['gen_body_rouge']

        infer_data['ref_gen_r1'].iloc[i], infer_data['ref_gen_r2'].iloc[i], infer_data['ref_gen_rl'].iloc[i], infer_data['ref_gen_bertscore'].iloc[i] = ref_gen['r1'], ref_gen['r2'], ref_gen['rl'], ref_gen['bert_score']
        infer_data['gen_body_bleu_list'].iloc[i], infer_data['gen_body_bleu_max'].iloc[i], infer_data['gen_body_bleu_avg'].iloc[i] = gen_body_bleu['bleu_list'], gen_body_bleu['bleu_max'], gen_body_bleu['bleu_avg']
        infer_data['gen_body_r1_list'].iloc[i], infer_data['gen_body_r2_list'].iloc[i], infer_data['gen_body_rl_list'].iloc[i] = gen_body_rouge['r1_list'], gen_body_rouge['r2_list'], gen_body_rouge['rl_list']
        infer_data['gen_body_r1_max'].iloc[i], infer_data['gen_body_r1_avg'].iloc[i] = gen_body_rouge['r1_max'], gen_body_rouge['r1_avg']
        infer_data['gen_body_r2_max'].iloc[i], infer_data['gen_body_r2_avg'].iloc[i] = gen_body_rouge['r2_max'], gen_body_rouge['r2_avg']
        infer_data['gen_body_rl_max'].iloc[i], infer_data['gen_body_rl_avg'].iloc[i] = gen_body_rouge['rl_max'], gen_body_rouge['rl_avg']
    except:
        print(i)
        nan_list.append(i)

    if i<=3:
        print(infer_data.iloc[i])

print('nan_list - ', nan_list)
infer_data.to_csv(os.path.join('dataset', args.data_type, 'infer_done', 'score_calculated.csv'), index=False)