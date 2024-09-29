# gsm8k-eval-batch-v1
reference: https://github.com/tianlwang/eval_gsm8k. This is an implementation of batch evaluation for GSM8K. 

## few-shot
### 8-shot
The 8-shot prompt is from the [lm-evaluation-harness gsm8k-cot](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml)

`python eval_gsm8k.py --model <model_name>`

| Model           | Accuracy | Harness Accuracy |
|-----------------|----------|------------------|
| Mistral-7B-v0.1 | 
| Llama-2-7b-hf   |      

### 8-shot maj1@8

`python eval_gsm8k.py --model <model_name> --use_majority_vote --temp 0.2 --n_votes 8`
| Model           | Accuracy | Harness Accuracy |
|-----------------|----------|------------------|
| Mistral-7B-v0.1 |

`python eval_gsm8k.py --model <model_name> --use_majority_vote --temp 0.4 --n_votes 8`
| Model           | Accuracy |
|-----------------|----------|
| Mistral-7B-v0.1 | 

# zero-shot
## cot zero-shot
use the Chain of Thought prompt "Let's think step by step." before answering the question.

`python eval_gsm8k.py --model <model_name> --cot`

| Model           | Accuracy | Harness Accuracy |
|-----------------|----------|------------------|
| Mistral-7B-v0.1 | 

## zero-shot

`python eval_gsmk_zero_shot.py --model <model_name> --zero-shot`

| Model           | Accuracy |
|-----------------|----------|
| Mistral-7B-v0.1 | 
