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
from mmengine import Config
import numpy as np
sys.path.append(str(Path(__file__).parent.parent.parent))
from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.models.builder.build_shikra import load_pretrained_shikra
from mllm.dataset.utils.transform import expand2square, box_xyxy_expand2square    
from PIL import Image
from transformers import set_seed
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

def eval_model(args):
    training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
    )) 
    quantization_kwargs = dict()
    model_args = Config(dict(
    type='shikra',
    version='v1',
    # checkpoint config
    cache_dir=None,
    model_name_or_path='/workspace/model/shikra-7b',
    vision_tower=r'openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,

    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,

    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=None),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)) 
    
    model, preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)
    model.to(dtype=torch.float16, device=torch.device('cuda'))
    tokenizer = preprocessor['text']
    
    preprocessor['target'] = {'boxes': PlainBoxFormatter()}

    gen_kwargs = dict(
    use_cache=True,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=10,
    top_p=1.0,
    temperature=1,
    )

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
    
    i = 0
    model.eval()
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
        
        question_len = len(tokenizer('You are an expert at question answering. Given the question, please output the answer. No explanation or further questions.\n Question: ' + question, return_tensors="pt", add_special_tokens=False).input_ids[0])
        prompt_len = len(tokenizer('\nContext: ', return_tensors="pt", add_special_tokens=False).input_ids[0])
        context_len = len(tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids[0])
        

        cur_prompt = 'You are an expert at question answering. Given the question, please output the answer. No explanation or further questions.\n Question: ' + question +  '\nContext: ' + context + '\nShort answer: '
        cur_prompt1 = 'You are an expert at question answering. Given the question, please output the answer. No explanation or further questions.\n Question: ' + question + '\nShort answer: '

        ds = prepare_interactive(model_args, preprocessor)
        ds1 = prepare_interactive(model_args, preprocessor)
        
        try:
            raw_image = Image.open(os.path.join(args.image_folder, image_file + '.jpg')).convert('RGB')
        except:
            raw_image = Image.open(os.path.join(args.image_folder, image_file + '.JPEG')).convert('RGB')
        
        image = expand2square(raw_image)
        boxes_value = [box_xyxy_expand2square(box, w=raw_image.width, h=raw_image.height) for box in []]

        ds.set_image(image)
        ds.append_message(role=ds.roles[0], message=cur_prompt, boxes=boxes_value, boxes_seq=[])
        model_inputs = ds.to_model_input()
        model_inputs['images'] = model_inputs['images'].to(torch.float16)
        input_ids = model_inputs['input_ids']
        input_token_len = input_ids.shape[-1]
        
        model_inputs['question_len'] = question_len
        model_inputs['prompt_len'] = prompt_len
        model_inputs['context_len'] = context_len
        model_inputs['ret_sim'] = ret_sim
        model_inputs['att_alpha'] = args.att_alpha
        model_inputs['img_start_idx'] = torch.nonzero(input_ids[0] == 32000, as_tuple=True)[0][0].item()
        model_inputs['img_end_idx'] = 256 + model_inputs['img_start_idx']
        model_inputs['cd_beta'] = args.cd_beta 

        ds1.set_image(image)
        ds1.append_message(role=ds.roles[0], message=cur_prompt1, boxes=boxes_value, boxes_seq=[])
        model_inputs1 = ds1.to_model_input()
        model_inputs1['images'] = model_inputs1['images'].to(torch.float16)
        model_inputs['images_cd'] = model_inputs1['input_ids']

        
        gen_kwargs['temperature'] = 1

        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                output_ids = model.generate(**model_inputs, **gen_kwargs)
        response = tokenizer.batch_decode(output_ids[:, input_token_len:])[0]


        outputs = response.replace('.</s>', '').replace('The answer is', '').strip()
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
