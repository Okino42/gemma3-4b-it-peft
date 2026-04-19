from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# TensorBoard event file path and the scalar series to plot.
# Update event_file and scalar_tag when switching to another training run.
event_file = Path("/projects/_hdd/checkpoint/gemma-3-4b-med-lora-s1/runs/Apr05_09-50-51_gpu-6000ada-2.cluster02.eee.ntu.edu.sg/events.out.tfevents.1775382679.gpu-6000ada-2.cluster02.eee.ntu.edu.sg.884615.0")
scalar_tag = "train/loss"

# EventAccumulator parses scalar, histogram, and other records from TensorBoard event files.
ea = event_accumulator.EventAccumulator(str(event_file))
ea.Reload()

print("Available scalar tags:", ea.Tags()["scalars"])

# Extract the step/value sequence for training loss and save it as an offline image.
events = ea.Scalars(scalar_tag)
steps = [e.step for e in events]
values = [e.value for e in events]

plt.figure(figsize=(8, 5))
plt.plot(steps, values, label=scalar_tag)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("train loss.png", dpi=300)
