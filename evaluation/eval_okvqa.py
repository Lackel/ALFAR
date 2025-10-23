
import pandas as pd
from ast import literal_eval
import okvqa_evaluation as evaluation
import numpy as np

val_annotations_df = pd.read_csv('/workspace/12_WVQA/data/eval_data/okvqa/a_ok_vqa_val_fixed_annots.csv')
val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)
results_df = pd.read_csv('/workspace/12_WVQA/experiments/result/aokvqa.csv')
results_df = pd.merge(val_annotations_df, results_df, on = 'question_id')
results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['answers'], row['question_id']),axis = 1)
print("VQA acc: ", np.round(results_df.acc.mean(),3))