# ----------------------------------------------------------------------------
# plot_loss.py
# ----------------------------------------------------------------------------
# This script reads TensorBoard event logs from model training and evaluation,
# extracts scalar loss metrics, and visualizes training vs. validation loss over
# time. The resulting line plot helps diagnose overfitting, convergence, and 
# training quality.



# --- Imports ---
import os
import matplotlib.pyplot as plt
#TensorBoard utility to load and parse event log files
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



# --- Paths ---
#directory where TensorBoard logs are saved (defined in config.yaml)
event_dir = "outputs/logs"
#find the first event file starting with 'events.out'
event_file = [f for f in os.listdir(event_dir) if f.startswith("events.out")][0]
#construct full path to the selected event log file
event_path = os.path.join(event_dir, event_file)

#load the event log using EventAccumulator
event_acc = EventAccumulator(event_path)
#parses and prepares scalar metrics from the TensorBoard log
event_acc.Reload()



# --- Check available scalar tags ---
#prints all scalar tags recorded during training (useful for debugging or custom tags)
print("Available tags:", event_acc.Tags()['scalars'])



# --- Extract Loss Metrics ---
#load training and validation loss values as lists of scalar events
#training loss per step
train_loss_events = event_acc.Scalars("train/loss")
#validation loss per step
eval_loss_events = event_acc.Scalars("eval/loss")



# --- Parse Step & Value ---
#extract x-axis (step) and y-axis (loss value) for training_loss
train_steps = [e.step for e in train_loss_events]
train_values = [e.value for e in train_loss_events]

#extract step and value for validation loss
eval_steps = [e.step for e in eval_loss_events]
eval_values = [e.value for e in eval_loss_events]



# --- Plot and Save ---
#set the size of the plot canvas
plt.figure(figsize=(10, 5))
#plot training loss with orange markers
plt.plot(train_steps, train_values, label="Training Loss", marker="o", color='orange')
#plot evaluation (validation) loss with blue markers
plt.plot(eval_steps, eval_values, label="Validation Loss", marker="x", color='blue')
#title and axis labels
plt.title("Training and Evaluation Loss Over Time")
plt.xlabel("Step")
plt.ylabel("Loss")
#add gridlines and legend
plt.grid(True)
plt.legend()
#adjust layout to avoid clipping
plt.tight_layout()
#save the plot to a file
plt.savefig("results/training_loss_curve.png")
#display the plot
plt.show()