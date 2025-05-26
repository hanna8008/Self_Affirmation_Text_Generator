# ----------------------------------------------------------------------------
# plot_loss.py
# ----------------------------------------------------------------------------



# --- Imports ---
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



# --- Paths ---
event_dir = "outputs/logs"
event_file = [f for f in os.listdir(event_dir) if f.startswith("events.out")][0]
event_path = os.path.join(event_dir, event_file)

event_acc = EventAccumulator(event_path)
event_acc.Reload()



# --- Check available scalar tags ---
print("Available tags:", event_acc.Tags()['scalars'])



# --- Extract Loss Metrics ---
train_loss_events = event_acc.Scalars("train/loss")
eval_loss_events = event_acc.Scalars("eval/loss")



# --- Parse Step & Value ---
train_steps = [e.step for e in train_loss_events]
train_values = [e.value for e in train_loss_events]

eval_steps = [e.step for e in eval_loss_events]
eval_values = [e.value for e in eval_loss_events]



# --- Plot and Save ---
plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_values, label="Training Loss", marker="o", color='orange')
plt.plot(eval_steps, eval_values, label="Validation Loss", marker="x", color='blue')
plt.title("Training and Evaluation Loss Over Time")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/training_loss_curve.png")
plt.show()