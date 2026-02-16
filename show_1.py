import json
import matplotlib.pyplot as plt

LOG_PATH = "ft_out/checkpoint-4500/trainer_state.json"

# -------------------------
# load log file
# -------------------------
with open(LOG_PATH, "r") as f:
    data = json.load(f)

log_history = data["log_history"]

# -------------------------
# collect metrics
# -------------------------
train_steps = []
train_loss  = []

eval_steps = []
eval_loss  = []

for entry in log_history:
    # training loss entries usually have "loss"
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry.get("step"))
        train_loss.append(entry["loss"])

    # eval entries usually have "eval_loss"
    if "eval_loss" in entry:
        eval_steps.append(entry.get("step"))
        eval_loss.append(entry["eval_loss"])

# -------------------------
# plot
# -------------------------
plt.figure(figsize=(10, 6))

if train_loss:
    plt.plot(train_steps, train_loss, label="train_loss")

if eval_loss:
    plt.plot(eval_steps, eval_loss, label="eval_loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss.png", dpi=300) 
plt.close()