import json
import matplotlib.pyplot as plt
import os

# Load trainer state
with open("model_save/checkpoint-218000/trainer_state.json", "r") as f:
    data = json.load(f)

logs = data.get("log_history", [])

steps = [entry["step"] for entry in logs if "step" in entry]
losses = [entry["loss"] for entry in logs if "loss" in entry]
lrs = [entry["learning_rate"] for entry in logs if "learning_rate" in entry]
grad_norms = [entry["grad_norm"] for entry in logs if "grad_norm" in entry]

# Create output folder
output_folder = "training_plots"
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, "training_curves.png")

# Plot
plt.figure(figsize=(16,5))

# Loss
plt.subplot(1,3,1)
plt.plot(steps[:len(losses)], losses, color="red")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

# Learning rate
plt.subplot(1,3,2)
plt.plot(steps[:len(lrs)], lrs, color="blue")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)

# Gradient norm
plt.subplot(1,3,3)
plt.plot(steps[:len(grad_norms)], grad_norms, color="green")
plt.xlabel("Step")
plt.ylabel("Grad Norm")
plt.title("Gradient Norm")
plt.grid(True)

plt.tight_layout()
plt.savefig(file_path, dpi=300) 
plt.close()

# 36_000 -> 26%
