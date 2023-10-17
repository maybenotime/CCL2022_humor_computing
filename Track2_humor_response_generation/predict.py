from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import argparse
import csv
import json


test_path = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/T5_humor/data/task2_test.json'
save_path = './test_result/Tal2022_任务2_largev3.csv'

def generate_sentence(input, model, tokenizer):
    model.eval()                 
    input_ids = input.unsqueeze(0).to('cuda')
    print(input_ids.shape)  
    outputs = model.generate(input_ids=input_ids,num_beams=20, max_length=128, repetition_penalty=10.0)
    output_str = tokenizer.decode(outputs.reshape(-1), skip_special_tokens=True)
    return output_str

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/aliyun-06/share_v2/yangshiping/projects/humor_computation/T5_humor/true_large_checkpoints/checkpoint-39440')
    parser.add_argument("--model_name", type=str, default='t5')
    args = parser.parse_args()
    return args

def load_test_data(path):
    with open(path,'r') as f:
        valid_data = json.load(f)
    return  valid_data


if __name__ == '__main__':
    args = args()
    if args.model_name == 't5':
        model = T5ForConditionalGeneration.from_pretrained(args.model_path).to('cuda')
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        
    valid_data = load_test_data(test_path)
    csv_data = []
    for entry in valid_data:
        context_input_ids = []
        for sen in entry['context']:
            sen += '</s>'           #分割utter
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen))
            context_input_ids.extend(input_ids)
        input_tensor = torch.tensor(context_input_ids)
        response = generate_sentence(input_tensor,model,tokenizer)
        dialogue_id = entry['Dialogue_id']
        speaker = entry['Speaker']
        csv_data.append([dialogue_id,speaker,response])
    
    print(csv_data)
    header = ['Dialogue_id', 'Speaker', 'Sentence']
    with open(save_path,"w",encoding='utf-8',newline='') as w:
        writer = csv.writer(w)
        writer.writerow(header)
        writer.writerows(csv_data)

