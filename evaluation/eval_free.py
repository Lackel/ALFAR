""" Infoseek Validation Set Evaluation script."""
from infoseek_eval import evaluate

if __name__ == "__main__":
    for split in ["val"]:
        pred_path = f"/workspace/12_WVQA/experiments/eval/result/test.jsonl"
        reference_path = f"/workspace/data/infoseek_val/infoseek_val.jsonl"
        reference_qtype_path = f"/workspace/data/infoseek_val/infoseek_val_qtype.jsonl"
        # reference_path = f"/workspace/data/infoseek_val/infoseek_human.jsonl"
        # reference_qtype_path = f"/workspace/data/infoseek_val/infoseek_human_qtype.jsonl"

        result = evaluate(pred_path, reference_path, reference_qtype_path)
        final_score = result["final_score"]
        unseen_question_score = result["unseen_question_score"]["score"]
        unseen_entity_score = result["unseen_entity_score"]["score"]
        print(f"{split} final score: {final_score}")
        print(f"{split} unseen question score: {unseen_question_score}")
        print(f"{split} unseen entity score: {unseen_entity_score}")
