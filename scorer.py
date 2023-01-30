import datasets
from metrics.rouge import Rouge
import numpy as np
from nltk.tokenize import sent_tokenize

class Score_Calculator:
    def __init__(self, tokenizer, sent_tokenizer=sent_tokenize, lang='kor', smooth=False, epsilon = 1e-7, device='cuda;0'):
        self.tokenizer = tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.lang = lang 
        self.smooth = smooth # smoothing argument for bleu score calculation
        self.epsilon = epsilon  # epsilon for rouge score calculation
        self.bert_scorer = datasets.load_metric("bertscore")
        self.bleu_scorer = datasets.load_metric('bleu')
        self.rouge_scorer = Rouge(rouge_types = ["rouge1", "rouge2", "rougeL"])
        self.device = device

    def wrap_bleu_calculate(self, reference_token, prediction_token_list):
        bleu_list = list()

        for sent_token in prediction_token_list:
            bleu = self.bleu_scorer.compute(predictions=[sent_token], references=[[reference_token]], smooth=self.smooth)['bleu']
            bleu_list.append(bleu)

        assert len(prediction_token_list) == len(bleu_list), "BLEU Scorer Calculation Error"

        bleu_max, bleu_avg = np.max(bleu_list), np.mean(bleu_list)
        
        return bleu_list, bleu_max, bleu_avg

    def rouge_calculate(self, reference_token, prediction_token):
        rouge_score_dict = self.rouge_scorer.score(reference_token, prediction_token)
        return rouge_score_dict['rouge1']['f1_score'], rouge_score_dict['rouge2']['f1_score'], rouge_score_dict['rougeL']['f1_score'] 

    # get rouge score with each element in prediction_token_list 
    def wrap_rouge_calculate(self, reference_token, prediction_token_list):
        r1_list = list()
        r2_list = list()
        rl_list = list()

        for sent_token in prediction_token_list:
            r1, r2, rl = self.rouge_calculate(reference_token, sent_token)
            r1_list.append(r1)
            r2_list.append(r2)
            rl_list.append(rl)
        
        assert len(r1_list) == len(r2_list) == len(rl_list) == len(prediction_token_list), "Rouge Scorer Calculation Error"

        r1_max, r1_avg = np.max(r1_list), np.mean(r1_list)
        r2_max, r2_avg = np.max(r2_list), np.mean(r2_list)
        rl_max, rl_avg = np.max(rl_list), np.mean(rl_list)

        return r1_list, r2_list, rl_list, r1_max, r1_avg, r2_max, r2_avg, rl_max, rl_avg

    def compute(self, title, summary, text, generated_summary) :

        """
        input : 
            title : str
            summary : str
            text : str
            generated_summary : str

        return : dictionary where each value is scalar
            ref_gen R1 | ref_gen R2 | ref_gen R-L | ref_gen BERTSCORE | 
            gen_body_list BLEU | gen_body BLEU MAX | gen_body BLEU AVG
            gen_body_list R1 | gen_body_list R2 | gen_body_list RL | 
            gen_body R1 MAX | gen_body R1 AVG | gen_body R2 MAX | gen_body R2 AVG | gen_body RL MAX | gen_body RL AVG | 
        """

        tokenized_title = self.tokenizer.tokenize(title)
        
        tokenized_ref_sum = self.tokenizer.tokenize(summary)
        tokenized_gen_sum = self.tokenizer.tokenize(generated_summary)
        
        text_list = self.sent_tokenizer(text)
        tokenized_text_list = [self.tokenizer.tokenize(sent) for sent in text_list]

        ref_gen_r1, ref_gen_r2, ref_gen_rl = self.rouge_calculate(tokenized_ref_sum, tokenized_gen_sum)
        ref_gen_bs = self.bert_scorer.compute(predictions=[generated_summary], references=[summary], lang=self.lang, device=self.device)['f1'][0]

        r1_text_list, r2_text_list, rl_text_list, r1_max, r1_avg, r2_max, r2_avg, rl_max, rl_avg = self.wrap_rouge_calculate(tokenized_gen_sum, tokenized_text_list)
        bleu_text_list, bleu_max, bleu_avg = self.wrap_bleu_calculate(tokenized_gen_sum, tokenized_text_list)

        result_dict = dict(ref_gen=dict(r1=ref_gen_r1, r2=ref_gen_r2, rl=ref_gen_rl, bert_score=ref_gen_bs),
                            gen_text_bleu=dict(bleu_list=bleu_text_list, bleu_max=bleu_max, bleu_avg=bleu_avg),
                            gen_text_rouge=dict(r1_list=r1_text_list, r2_list=r2_text_list, rl_list=rl_text_list,
                            r1_max=r1_max, r1_avg=r1_avg, r2_max=r2_max, r2_avg=r2_avg, rl_max=rl_max, rl_avg=rl_avg)         
        )

        return result_dict