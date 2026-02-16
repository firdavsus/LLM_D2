import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer
from LLM import Transformer, Config

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# dataset
# -------------------------
ds = load_dataset("yahma/alpaca-cleaned", split="train")

def format_sample(x):
    if x["input"]:
        prompt = f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{x['instruction']}\n\n### Response:\n"

    full_text = prompt + x["output"] + tokenizer.eos_token
    tok = tokenizer(full_text)

    input_ids = tok["input_ids"]

    labels = input_ids[1:] + [-100]

    prompt_len = len(tokenizer(prompt)["input_ids"])
    for i in range(prompt_len - 1):
        labels[i] = -100

    return {"input_ids": input_ids, "labels": labels}


dataset = ds.map(format_sample, remove_columns=ds.column_names)

# -------------------------
# filter too-short answers
# -------------------------
MAX_LENGTH = 1024

def keep_valid(example):
    total_len = len(example["input_ids"])
    answer_len = len(example["labels"]) - example["labels"].count(-100)
    return total_len <= MAX_LENGTH and answer_len > 5

dataset = dataset.filter(keep_valid)

# -------------------------
# train / eval split
# -------------------------
eval_dataset = dataset.select(range(500))
train_dataset = dataset.select(range(500, len(dataset)))

# -------------------------
# collator: pad to batch max length
# -------------------------
def causal_lm_collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    labels    = [torch.tensor(f["labels"],    dtype=torch.long) for f in features]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels    = pad_sequence(labels,    batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# -------------------------
# model
# -------------------------
config = Config()
model = Transformer(config).to(device)

# load pretrained checkpoint
state = torch.load("model_save/checkpoint-218000/pytorch_model.bin", map_location=device)
clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
model.load_state_dict(clean, strict=True)


def to_json_string(self):
    import json
    return json.dumps(self.__dict__, indent=2)

Config.to_json_string = to_json_string

# -------------------------
# training args (NOTE: name = training_args)
# -------------------------
training_args = TrainingArguments(
    output_dir="./ft_out",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=10,
    warmup_ratio=0.10,
    lr_scheduler_type="cosine", 
    logging_steps=25,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy="steps",
    save_total_limit=3,
    bf16=torch.cuda.is_bf16_supported(),
    weight_decay=0.01,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    save_safetensors=False,
    max_grad_norm=1.0
)

# -------------------------
# trainer
# -------------------------
class MyTrainer(Trainer):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator
        )

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=causal_lm_collator
)

# -------------------------
# train
# -------------------------
trainer.train()