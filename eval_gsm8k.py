import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList
)
from utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from datasets import load_dataset
from collections import Counter
import json


FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A:"""




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/bingxing2/home/scx6d1j/zgy/llama31')
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--padding_side', type=str, default='left')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--type', choices=['few-shot', 'zero-shot', 'cot'],type=str, default='few-shot')
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16) 
    
    print('\nLoading dataset...')
    # dataset = load_dataset('gsm8k', "main", split='test')
    dataset = load_dataset('/home/bingxing2/home/scx6d1j/zgy/eval_gsm8k-main/dataset', split='test')
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
    ]
    
    # 批量生成few shot 提示
    def prompt_format(question):
        if args.type == 'few-shot':
            return FEW_SHOT_PROMPT.format(question=question)
        elif args.type == 'zero-shot':
            return 'Q: ' + question + '\nA:'
        elif args.type == 'cot':
            return "Q: {question}\nA: Let's think step by step.".format(question)

    def prompt_and_ans(examples):
        input_texts = []
        ground_truth_answers = []
        questions = examples['question']
        answers = examples['answer']
        for question, answer in zip(questions, answers):
            input_texts.append(prompt_format(question))
            ground_truth_answers.append( extract_ground_truth(answer))
        return input_texts, ground_truth_answers
    def extract_predicted_answers(output_texts):
        model_answers = []
        for output_text in output_texts:
            model_answers.append(extract_predicted_answer(output_text))
        return model_answers
    def post_process_ans(output_texts, model_answers):
        vote_model_answers = []
        for output_text, model_answer in zip(output_texts, model_answers):
            vote_model_answers.append({'text': output_text, 'numeric': model_answer})
        return vote_model_answers
    def compare_answer(examples, vote_model_answers, ground_truth_answers):
        numeric_answers = [[] for _ in range(len(ground_truth_answers))]
        filtered_answers = [[] for _ in range(len(ground_truth_answers))]
        text_answers = [[] for _ in range(len(ground_truth_answers))]
        for model_answers in vote_model_answers:
            for i, model_answer in enumerate(model_answers):
                if model_answer['numeric'] is not None:
                    filtered_answers[i].append(model_answer['numeric'])
                numeric_answers[i].append(model_answer['numeric'])
                text_answers[i].append(model_answer['text'])
        results = []
        # 一个批次的大小
        assert len(model_answers) == len(ground_truth_answers), "数据出现了不对齐的现象"
        orgin_questions = examples['question']
        orgin_answers = examples['answer']
        for i, ( filtered_answer, ground_truth_answer) in enumerate(zip(filtered_answers, ground_truth_answers)):
            majority_answer = Counter(filtered_answer).most_common(1)[0][0] if filtered_answer else None
            correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
            results.append({
                'question': orgin_questions[i],
                'gold_answer_text': orgin_answers[i],
                'model_answers_text': text_answers[i],
                'extracted_model_answers': numeric_answers[i],
                'extracted_gold_answer': ground_truth_answer,
                'majority_answer': majority_answer,
                'correct': correct
            })
        return results


    results = []

    batch_size = args.batch_size
    batchs = datasize // batch_size

    def one_batch_results(batch_example):
        input_texts, ground_truth_answers = prompt_and_ans(batch_example)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(input_texts, return_tensors='pt', padding=True).to(model.device)
        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, inputs)
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])
        prompt_len = inputs['input_ids'].size(1)
        vote_model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                    outputs = model.generate(**inputs, temperature=args.temp, max_new_tokens=args.max_new_tokens, do_sample=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
                output_texts = tokenizer.batch_decode(outputs['sequences'][:, prompt_len:], skip_special_tokens=True)
                # Extract the final answer from the model's output
                model_answers = extract_predicted_answers(output_texts)
                vote_model_answers.append(post_process_ans(output_texts, model_answers))
        else:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
            output_texts = tokenizer.batch_decode(outputs['sequences'][:, prompt_len:], skip_special_tokens=True)
            model_answers = extract_predicted_answers(output_texts)
            vote_model_answers.append(post_process_ans(output_texts, model_answers))

        return compare_answer(batch_example, vote_model_answers, ground_truth_answers)
    
    for i in tqdm(range(0, batchs), desc="evaluate"):
        start = batch_size * i
        batch_example = dataset[start: start + batch_size]
        results += one_batch_results(batch_example)
    if datasize % batch_size != 0:
        batch_example = dataset[batchs * batch_size:]
        results += one_batch_results(batch_example)

    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")

    results.append({'accuracy': cnt / total})

    os.makedirs(f'eval_results/{args.type}', exist_ok=True)
    
    model_name = args.model.split('/')[-1]
    result_file = f"eval_results/{args.type}/{model_name}"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")
                

if __name__ == '__main__':
    main()


