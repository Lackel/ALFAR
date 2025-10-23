import argparse
import json
from tqdm import tqdm
import sys
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria    
from PIL import Image
from transformers import set_seed
from vcd_sample import evolve_vcd_sampling
from pathlib import Path
evolve_vcd_sampling()


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / 'data' / 'eval_data' / 'okvqa'
    
    if args.dataset == 'aokvqa':
        questions = pd.read_csv(DATA_DIR / 'a_ok_vqa_val_fixed_annots.csv')
        knowledge = json.load(open(DATA_DIR / 'aokvqa_val_dcaption.json'))
    elif args.dataset == 'okvqa':
        questions = pd.read_csv(DATA_DIR / 'ok_vqa_val_annots_fixed.csv')
        knowledge = json.load(open(DATA_DIR / 'okvqa_val_dcaption.json'))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_id_list = []
    image_id_list = []
    ans_list = []
    
    for i in tqdm(range(questions.shape[0])):
        test_sample = questions.iloc[i]
        
        context = knowledge[str(test_sample.question_id)]
        idx = test_sample.question_id
        image_file = test_sample.image_path
        question = test_sample.question

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        

        question_len = len(tokenizer('\n'+ question, return_tensors="pt", add_special_tokens=False).input_ids[0])
        prompt_len = len(tokenizer('Answer the question using a single word or phrase based on the given context. Context: ', return_tensors="pt", add_special_tokens=False).input_ids[0])
        context_len = len(tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids[0])
        
        conv = conv_templates[args.conv_mode].copy()
        conv1 = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0],  qs  + 'Answer the question using a single word or phrase based on the given context. Context: ' + context)
        
        conv.append_message(conv.roles[1], None)
        
        conv1.append_message(conv.roles[0],  qs + 'Answer the question using a single word or phrase based on your knowledge.')
        conv1.append_message(conv.roles[1], None)


        

        prompt = conv.get_prompt()
        prompt1 = conv1.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_ids1 = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0).half().cuda(),
                images_cd = input_ids1,
                cd_beta = args.cd_beta,
                question_len = question_len,
                prompt_len = prompt_len,
                context_len = context_len,
                ret_sim = 0.9,
                att_alpha = args.att_alpha,
                img_start_idx = 35,
                img_end_idx = 611,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=10,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]

        

        outputs = outputs.strip()

                # if i%10==0 and i>0: #save preds every 10 samples
        #     llama_preds_df = pd.DataFrame({'question_id':question_id_list,'image_id':image_id_list,'llama_answer':ans_list})
        #     llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=","").strip())
        #     llama_preds_df.to_csv(ans_file, index=False)
    
        question_id_list.append(test_sample.question_id)
        image_id_list.append(test_sample.image_id)
        ans_list.append(outputs)

    llama_preds_df = pd.DataFrame({'question_id':question_id_list,'image_id':image_id_list,'llama_answer':ans_list})
    llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=","").strip())
    llama_preds_df.to_csv(ans_file,index=False)
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/model/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/data/val2017")
    parser.add_argument("--dataset", type=str, default="aokvqa")
    parser.add_argument("--answers-file", type=str, default="../result/aokvqa.csv")
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
