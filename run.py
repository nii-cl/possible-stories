import argparse
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset, set_caching_enabled
from model.deberta_v2_multiple_choice import DebertaV2ForMultipleChoice
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_util import DataCollatorForMultipleChoice

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument(
    "--input_mode", default="full"
)  # no_context, no_question, or option_only
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

os.environ["WANDB_DISABLED"] = "true"
set_caching_enabled(False)

set_seed = 2022
random.seed(set_seed)
os.environ["PYTHONHASHSEED"] = str(set_seed)
np.random.seed(set_seed)
torch.manual_seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

dataset_dir = "dataset"
result_dir = "results"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

datasets = load_dataset(
    "dataset",
    data_files={
        "train": "train.jsonl",
        "validation": "dev.jsonl",
        "test": "test.jsonl",
    },
    streaming=True,
)

model_checkpoint, batch_size, learning_rate = {
    "bert-base": ("bert-base-uncased", 24, 3e-5),
    "bert-large": ("bert-large-uncased", 12, 1e-5),
    "roberta-base": ("roberta-base", 16, 3e-5),
    "roberta-large": ("roberta-large", 8, 1e-5),
    "roberta-large-race": ("LIAMF-USP/roberta-large-finetuned-race", 8, 1e-5),
    "deberta-base": ("microsoft/deberta-v3-base", 16, 3e-5),
    "deberta-large": ("microsoft/deberta-v3-large", 8, 1e-5),
}[args.model_name]

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


def preprocess(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    if args.input_mode in ["no_context", "option_only"]:
        first_sentences = [[""] * 4 for context in examples["document"]]
    else:
        first_sentences = [[context] * 4 for context in examples["document"]]

    # Grab all second sentences possible for each context.
    question_headers = examples["question"]
    if args.input_mode in ["no_question", "option_only"]:
        second_sentences = examples["options"]
    else:
        second_sentences = [
            [f"{q} {opt}" for opt in opts]
            for q, opts in zip(examples["question"], examples["options"])
        ]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)

    # Un-flatten
    encoded_data = {
        k: [v[i : i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }
    encoded_data["labels"] = examples["gold_label"]
    return encoded_data


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


datasets = datasets.with_format("torch")
encoded_datasets = datasets.map(preprocess, batched=True)
train_size = len([x for x in datasets["train"]])
test_size = len([x for x in datasets["test"]])

model_name = model_checkpoint.split("/")[-1]
folder_name = os.path.join(
    f"{result_dir}", f"{args.model_name}_{set_seed}_{args.input_mode}"
)

train_args = TrainingArguments(
    folder_name,
    evaluation_strategy="steps",  # or "epoch"
    eval_steps=200,
    save_steps=200,
    learning_rate=learning_rate or 1e-5,
    seed=set_seed,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=3,
    # num_train_epochs=10,  # or 5
    max_steps=math.ceil(train_size * 0.1 / batch_size),
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


if "deberta" in model_checkpoint:
    model_module = DebertaV2ForMultipleChoice
else:
    model_module = AutoModelForMultipleChoice

trainer = Trainer(
    model=model_module.from_pretrained(model_checkpoint),
    args=train_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

if args.train:
    trainer.train()

res = trainer.predict(encoded_datasets["test"])
print(f"accuracy:\t{res[2]['test_accuracy']}")

# consistency
passage_ids = defaultdict(list)
for pred, d in zip(res[0], datasets["test"]):
    passage_ids[d["question_id"].split("_")[0]].append(
        d["gold_label"] == np.argmax(pred)
    )
passage_wise_results = [1 if len(v) == sum(v) else 0 for v in passage_ids.values()]
print(f"consistency:\t{sum(passage_wise_results)/len(passage_wise_results)}")
