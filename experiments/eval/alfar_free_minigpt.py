import argparse
import json
from tqdm import tqdm
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
sys.path.append(str(Path(__file__).parent.parent.parent))
from minigpt4.common.eval_utils import init_model   
from minigpt4.common.config import Config
from PIL import Image
from transformers import set_seed
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()


class MiniGPT4Config:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.options = None

def eval_model(args):
    cfg_path = "../minigpt4/eval_config/minigpt4_llama2_eval.yaml"
    cfg = MiniGPT4Config(cfg_path)
    model, vis_processor = init_model(cfg)
    tokenizer = model.llama_tokenizer
    template = "[INST] <Img><ImageHere></Img> <question> [/INST] "



    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / 'data'
    knowledge_base = np.load(DATA_DIR / 'wiki' / 'wiki_with_image.npy', allow_pickle=True).item()
    if args.dataset == 'human':
        question_file = DATA_DIR / 'eval_data' / 'free' / 'infoseek_human.jsonl'
        indices = np.load(DATA_DIR / 'retrieval_result' / 'human_indices_50.npy', allow_pickle=True)
        retrieval_sim = np.load(DATA_DIR / 'retrieval_result' / 'human_distance_50.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map.npy', allow_pickle=True).item()
    elif args.dataset == 'validation':
        question_file = DATA_DIR / 'eval_data' / 'free' / 'infoseek_val.jsonl'
        indices = np.load(DATA_DIR / 'retrieval_result' / 'val_indices_50_17k.npy', allow_pickle=True)
        retrieval_sim = np.load(DATA_DIR / 'retrieval_result' / 'val_distance_50_17k.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map_17k.npy', allow_pickle=True).item()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    

    
    model.eval()
    i = 0
    for line in tqdm(questions):
        ret_sim = retrieval_sim[i, 0]
        i = i + 1
        
        context = ''
        idx = line['data_id']

        
        indice = indices[i-1]
        know_index = indice_map[indice[0]]
        context = knowledge_base[know_index]['wikipedia_summary']
        if context == '':
            context = knowledge_base[know_index]['wikipedia_content']
        
        image_file = line["image_id"]
        question = line["question"]


        context_len = len(tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids[0])
        
    

        cur_prompt = 'You are an expert at question answering. Given the question, please output the answer. No explanation or further questions.\n Question: ' + question + '\nContext: ' + context + '\nShort answer with one word: '
        cur_prompt1 = 'You are an expert at question answering. Given the question, please output the answer. No explanation or further questions.\n Question: ' + question + '\nShort answer with one word: '

        text = [template.replace("<question>", cur_prompt)]
        text1 = [template.replace("<question>", cur_prompt1)]
        img_start_idx = model.llama_tokenizer(text[0].split("<ImageHere>")[0], return_tensors="pt", add_special_tokens=False).input_ids.shape[-1] + 1
        img_end_idx = img_start_idx + 32
        question_len = len(tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0])
        prompt_len = len(tokenizer('You are an expert at question answering. Given the question, please output the answer. No explanation or further questions.\n Question: ', return_tensors="pt", add_special_tokens=False).input_ids[0]) + 2
        
        try:
            raw_image = Image.open(os.path.join(args.image_folder, image_file + '.jpg')).convert('RGB')
        except:
            raw_image = Image.open(os.path.join(args.image_folder, image_file + '.JPEG')).convert('RGB')
        
        image = vis_processor(raw_image).unsqueeze(0).to('cuda')


        

        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                outputs = model.generate(images=image, texts=text, img_start_idx=img_start_idx, img_end_idx=img_end_idx, question_len = question_len,
                prompt_len = prompt_len, context_len=context_len, ret_sim=ret_sim, images_cd=text1, att_alpha=args.att_alpha, do_sample=True, temperature=args.temperature, top_p=args.top_p,cd_beta=args.cd_beta, max_new_tokens=10)
        

        outputs = outputs[0].strip()
        ans_file.write(json.dumps({"data_id": idx,
                                   "prediction": outputs,
                                   "model_id": 'Shikra',
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/model/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/data/infoseek_human_images")
    parser.add_argument("--dataset", type=str, default="human")
    parser.add_argument("--answers-file", type=str, default="../result/free.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--cd_beta", type=float, default=0.7)
    parser.add_argument("--att_alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args) 
