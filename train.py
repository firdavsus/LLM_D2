# dataset -> NeelNanda/pile-tokenized-10b used tokenizer-> EleutherAI/gpt-neox-20b

import torch
torch.cuda.empty_cache()      # frees cached memory
torch.cuda.reset_peak_memory_stats()  # optional, resets peak stats
import gc
gc.collect()
torch.cuda.empty_cache()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

### DATASET ###
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

# --- Load train and validation splits ---
full_data = load_dataset("NeelNanda/pile-tokenized-10b", split="train", cache_dir="../cache_dataset/")

eval_data = full_data.select(range(1_000))  
train_data = full_data.select(range(1_000, len(full_data)))

# --- Prepare function (same as training) ---
def prepare_dataset(batch):
    seq = batch["tokens"]
    batch["input_ids"] = seq[:-1]
    batch["labels"] = seq[1:]
    return batch

train_dataset = train_data.map(prepare_dataset, remove_columns=["tokens"], num_proc=32)
eval_dataset  = eval_data.map(prepare_dataset, remove_columns=["tokens"], num_proc=16)

print("total number of samples: ", len(train_dataset))
print("total number of val slice: ", len(eval_dataset))

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# --- Data collator ---
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

# --- Load model ---
from LLM import Transformer, Config
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(config).to(device) 
# # load checkpoint
# state_dict = torch.load(
#     "model_save/checkpoint-68000/pytorch_model.bin",
#     map_location=device
# )

# # remove torch.compile wrapper
# clean_state_dict = {
#     k.replace("_orig_mod.", ""): v
#     for k, v in state_dict.items()
# }

# # load correctly
# model.load_state_dict(clean_state_dict, strict=True)
model = torch.compile(model)

def to_json_string(self):
    import json
    return json.dumps(self.__dict__, indent=2)

Config.to_json_string = to_json_string

print("Model will train on bf16: ", torch.cuda.is_bf16_supported())
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# --- Training args ---
training_args = TrainingArguments(
    output_dir="./model_save",
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=32,
    evaluation_strategy="steps",
    logging_strategy="steps",
    eval_steps=1_000, 
    num_train_epochs=30,
    learning_rate=3e-4,
    max_grad_norm=1.0, 
    lr_scheduler_type="cosine", 
    warmup_steps=10_000, 
    bf16=torch.cuda.is_bf16_supported(), 
    # group_by_length=True,
    logging_steps=25,
    save_steps=1_000,
    save_total_limit=3,
    weight_decay=0.01,
    dataloader_num_workers=16,
    remove_unused_columns=False,
    save_safetensors=False,
    gradient_checkpointing = False # train faster, higher vram (I have plenty hehehehe)
)

# --- Trainer ---
from torch.utils.data import DataLoader

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
    data_collator=data_collator
)

# --- Train ---
# trainer.train()
trainer.train(resume_from_checkpoint="model_save/checkpoint-89000")