import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
import os
os.environ['http_proxy'] = 'http://202.117.43.244:10007'
os.environ['https_proxy'] = 'http://202.117.43.244:10007'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.utils import disable_torch_init
from PIL import Image
import math
from lavis.models import load_model_and_preprocess
from vcd_sample import evolve_vcd_sampling
import numpy as np 
evolve_vcd_sampling()
from pathlib import Path

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    tokenizer = model.llm_tokenizer
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / 'data'
    if args.dataset == 'infoseek':
        question_file = DATA_DIR / 'eval_data' / 'mc' / 'infoseek_mc.json'
        indices = np.load(DATA_DIR / 'retrieval_result' / 'infoseek_mc_indices_50_17k.npy', allow_pickle=True)
        retrieval_sim = np.load(DATA_DIR / 'retrieval_result' / 'infoseek_mc_distance_50_17k.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map_17k.npy', allow_pickle=True).item()
    elif args.dataset == 'viquae':
        question_file = DATA_DIR / 'eval_data' / 'mc' / 'viquae_mc.json'
        indices = np.load(DATA_DIR / 'retrieval_result' / 'viquae_indices_50.npy', allow_pickle=True)
        retrieval_sim = np.load(DATA_DIR / 'retrieval_result' / 'viquae_distance_50.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map.npy', allow_pickle=True).item()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    

    with open(os.path.expanduser(question_file), "r") as f :
        questions = json.load(f)
    knowledge_base = np.load(DATA_DIR / 'wiki' / 'wiki_with_image.npy', allow_pickle=True).item()
    
    i = 0
    for line in tqdm(questions):
        ret_sim = retrieval_sim[i, 0]
        i = i + 1
        context = ''
        idx = line['id']
        choices = line['multiple_choices']
        question = line["question"]
    
        indice = indices[i-1]
        know_index = indice_map[indice[0]]
        context = knowledge_base[know_index]['wikipedia_summary']
        if context == '':
            context = knowledge_base[know_index]['wikipedia_content']
        
        
        temp = '\n Option: \n'
        for c_name, c_content in choices.items():
            temp += f"{c_name}: {c_content}\n"

        question_len = len(tokenizer('You are an expert at question answering. Given the question, please output the answer. No explanation or further questions. Question: ' + question, return_tensors="pt", add_special_tokens=False).input_ids[0])
        prompt_len = len(tokenizer('\nContext: ', return_tensors="pt", add_special_tokens=False).input_ids[0])
        context_len = len(tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids[0])


        prompt = 'You are an expert at question answering. Given the question, please output the answer. No explanation or further questions. Question: ' + question +  '\nContext: ' + context + temp + 'Short answer: '

        prompt1 = 'You are an expert at question answering. Given the question, please output the answer. No explanation or further questions. Question: ' + question + temp + 'Short answer: '

        if args.dataset == 'infoseek':
            image_file = line["image"].split('.')[0]
            try:
                raw_image = Image.open(os.path.join(args.image_folder, image_file + '.jpg')).convert('RGB')
            except:
                raw_image = Image.open(os.path.join(args.image_folder, image_file+ '.JPEG')).convert('RGB')
        else:
            image_file = line["image"]
            raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to('cuda:0')





        with torch.inference_mode():
            outputs = model.generate({"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=1,
                top_p = args.top_p, repetition_penalty=1,
                images_cd=prompt1, cd_beta = args.cd_beta, temperature=args.temperature, 
                max_length=10, img_start_idx=0, img_end_idx=32, 
                question_len = question_len, prompt_len = prompt_len, 
                context_len = context_len, ret_sim = ret_sim, att_alpha = args.att_alpha)


        outputs = outputs[0]
        outputs = ' '.join(outputs.split(' ')[1:])
        ans_file.write(json.dumps({"data_id": idx,
                                   "prediction": outputs,
                                   "model_id": "instruct_blip",
                                   "image": image_file}) + "\n")
        ans_file.flush()
    ans_file.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/model/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/data/infoseek_val_images")
    parser.add_argument("--dataset", type=str, default="infoseek")
    parser.add_argument("--answers-file", type=str, default="../result/mc.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--cd_beta", type=float, default=0.7)
    parser.add_argument("--att_alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
