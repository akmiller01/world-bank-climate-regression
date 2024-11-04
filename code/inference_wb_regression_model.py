from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset

global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL = AutoModelForSequenceClassification.from_pretrained('alex-miller/wb-climate-regression')
MODEL = MODEL.to(DEVICE)

def inference(model, inputs):
    predictions = model(**inputs)
    logits = predictions.logits.cpu().detach().numpy()[0]
    return logits

def map_columns(example):
    inputs = TOKENIZER(example['text'], return_tensors='pt', truncation=True).to(DEVICE)
    logits = inference(MODEL, inputs)
    example['pred'] = logits[0]
    example['pred_a'] = logits[1]
    example['pred_m'] = logits[2]
    return example

def main():
    dataset = load_dataset('alex-miller/wb-climate-percentage', split='test')
    dataset = dataset.map(map_columns)
    dataset.to_csv('output/wb_regression_inference.csv')


if __name__ == '__main__':
    main()


