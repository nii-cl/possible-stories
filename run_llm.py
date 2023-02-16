import json
import random

import tqdm
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

random.seed(1234)

model_name = "flan-t5-xxl"  # or 'race'
dataset_name = "possible_stories"  # or 'race'
shot_num = 0

tokenizer = T5Tokenizer.from_pretrained(f"google/{model_name}")
model = T5ForConditionalGeneration.from_pretrained(
    f"google/{model_name}", device_map="auto"
)


def get_prediction(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


def multiple_choice_to_prompt(passage, question, options, answer=""):
    ret = f"Passage: {passage}\n"
    ret += f"Question: {question}\n"
    opt_str = " ".join([f"({i}) {j}" for i, j in zip("ABCD", options)])
    ret += f"Options: {opt_str}\n"
    if answer:
        ret += f"Answer: {answer}\n"
    else:
        ret += "Answer:"
    return ret


if dataset_name == "race":
    dataset = load_dataset("race", "all")
    key_set = ["article", "question", "options", "answer", "example_id"]
elif dataset_name == "possible_stories":
    dataset = load_dataset(
        "../possible-stories/dataset",
        data_files={
            "train": "train.jsonl",
            "validation": "dev.jsonl",
            "test": "test.jsonl",
        },
        streaming=True,
    )
    key_set = ["document", "question", "options", "gold_label", "question_id"]

few_shot_prompt = ""
if shot_num > 0:
    few_shot_examples = random.sample(list(dataset["train"]), shot_num)
    few_shot_prompt = "\n".join(
        [
            multiple_choice_to_prompt(
                ex[key_set[0]], ex["question"], ex["options"], ex[key_set[3]]
            )
            for ex in few_shot_examples
        ]
    )

instruction = (
    "Read the given passage and answer the multiple choice question about it.\n"
)

results = []
predictions = []
for ex in tqdm.tqdm(dataset["test"]):
    ex_prompt = multiple_choice_to_prompt(ex[key_set[0]], ex["question"], ex["options"])
    prompt = instruction + few_shot_prompt + "\n" + ex_prompt
    pred = get_prediction(prompt)
    predictions.append((ex[key_set[4]], ex[key_set[3]], pred))
    pred_answer = "E"
    for pred_char in pred:
        if pred_char in "ABCD":
            pred_answer = pred_char
            break
    if ex[key_set[3]] in range(0, 4):
        gold_answer = chr(ord("A") + ex[key_set[3]])
    else:
        gold_answer = ex[key_set[3]]
    results.append(pred_answer == gold_answer)
print(f"Accuracy: {sum(results)/len(results)}")

with open(
    f"results/{model_name}-shot{shot_num}-{dataset_name}-predictions.jsonl", "w"
) as f:
    for prediction in predictions:
        f.write(
            json.dumps(
                {
                    "id": prediction[0],
                    "label": prediction[1],
                    "prediction": prediction[2],
                }
            )
            + "\n"
        )

if dataset_name == "possible_stories":
    from collections import defaultdict

    passage_wise = defaultdict(list)
    for p, r in zip(predictions, results):
        passage_wise[p[0].split("_")[0]].append(r)
    consist = [1 if sum(l) == len(l) else 0 for l in passage_wise.values()]
    print(f"Consistency: {sum(consist)/len(consist)}")
    # [three shot results of Flan-T5 XXL]
    # seed 1234: acc = 82.0, consist = 51.0
    # seed 2345: acc = 81.2, consist = 51.5
    # seed 3456: acc = 81.1, consist = 50.0
    # seed 4567: acc = 81.7, consist = 50.5
    # seed 5678: acc = 81.2, consist = 50.5
    # average: acc = 81.4, consist = 50.7
