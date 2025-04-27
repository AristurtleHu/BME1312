import matplotlib
import platform
from pathlib import Path
import numpy as np

if platform.system() == "Linux":
    matplotlib.use("Agg")  # Use a non-interactive backend only on Linux
import matplotlib.pyplot as plt


train_losses = []
val_losses = []

# Construct the path in an OS-independent way
output_file_path = Path("./output/output.txt")
output_file_path_L1 = Path("./output/L1_Loss_output.txt")
output_file_path_No_Opt = Path("./output/No_Opt_output.txt")
output_file_path_unrolled = Path("./output/unrolled_output.txt")
output_file_path_with_attention = Path("./output/with_attention.txt")
output_file_path_One_unet = Path("./output/one_unet.txt")

with open(output_file_path, "r") as f:
    for line in f:
        if "Train Loss" in line and "Val Loss" in line:
            parts = line.strip().split(",")
            train_part = parts[1].strip()
            train_loss = float(train_part.split(":")[1].strip())
            val_part = parts[2].strip()
            val_loss = float(val_part.split(":")[1].strip())

            train_losses.append(train_loss)
            val_losses.append(val_loss)

plt.figure(figsize=(8, 5))
plt.plot(np.log(train_losses), label="Log Train Loss")
plt.plot(np.log(val_losses), label="Log Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(f"assets/Training Loss and Validation Loss.png")
plt.close()

# Reset lists for L1 plot
train_losses_L1 = []
val_losses_L1 = []
with open(output_file_path_L1, "r") as f:
    for line in f:
        # Example line: "Epoch [1/800] Train Loss: 0.367360, Val Loss: 0.348975"
        if "Train Loss" in line and "Val Loss" in line:
            try:
                parts = line.strip().split(",")
                # Find the part containing "Train Loss" and extract the value
                train_part = [p for p in parts if "Train Loss" in p][0]
                train_loss = float(train_part.split(":")[1].strip())
                # Find the part containing "Val Loss" and extract the value
                val_part = [p for p in parts if "Val Loss" in p][0]
                val_loss = float(val_part.split(":")[1].strip())

                train_losses_L1.append(train_loss)
                val_losses_L1.append(val_loss)
            except (IndexError, ValueError) as e:
                print(f"Skipping line due to parsing error: {line.strip()} - {e}")


# Plot L1 losses
plt.figure(figsize=(8, 5))
plt.plot(np.log(train_losses_L1), label="Log Train Loss (L1)")
plt.plot(np.log(val_losses_L1), label="Log Validation Loss (L1)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve with L1")
plt.legend()
plt.grid(True)
plt.savefig(f"assets/Training Loss and Validation Loss L1.png")
plt.close()

# Reset lists for No_Opt plot
train_losses_No_Opt = []
val_losses_No_Opt = []
with open(output_file_path_No_Opt, "r") as f:
    for line in f:
        # Example line: "Epoch [1/800] Train Loss: 0.367360, Val Loss: 0.348975"
        if "Train Loss" in line and "Val Loss" in line:
            try:
                parts = line.strip().split(",")
                # Find the part containing "Train Loss" and extract the value
                train_part = [p for p in parts if "Train Loss" in p][0]
                train_loss = float(train_part.split(":")[1].strip())
                # Find the part containing "Val Loss" and extract the value
                val_part = [p for p in parts if "Val Loss" in p][0]
                val_loss = float(val_part.split(":")[1].strip())

                train_losses_No_Opt.append(train_loss)
                val_losses_No_Opt.append(val_loss)
            except (IndexError, ValueError) as e:
                print(f"Skipping line due to parsing error: {line.strip()} - {e}")


# Plot No_Opt losses
plt.figure(figsize=(8, 5))
plt.plot(np.log(train_losses_No_Opt), label="Log Train Loss (No Opt)")
plt.plot(np.log(val_losses_No_Opt), label="Log Validation Loss (No Opt)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve without Optimization")
plt.legend()
plt.grid(True)
plt.savefig(f"assets/Training Loss and Validation Loss No opt.png")
plt.close()

# Reset lists for unrolled plot
train_losses_unrolled = []
val_losses_unrolled = []

with open(output_file_path_unrolled, "r") as f:
    for line in f:
        # Example line: Epoch [1/300], Train Loss: 0.088762, Val Loss: 0.068758, Epoch Time: 274.75s
        if "Train Loss" in line and "Val Loss" in line:
            try:
                parts = line.strip().split(",")
                # Find the part containing "Train Loss" and extract the value
                train_part = [p for p in parts if "Train Loss" in p][0]
                train_loss = float(train_part.split(":")[1].strip())
                # Find the part containing "Val Loss" and extract the value
                val_part = [p for p in parts if "Val Loss" in p][0]
                val_loss = float(val_part.split(":")[1].strip())

                train_losses_unrolled.append(train_loss)
                val_losses_unrolled.append(val_loss)
            except (IndexError, ValueError) as e:
                print(f"Skipping line due to parsing error: {line.strip()} - {e}")


# Plot unrolled losses
plt.figure(figsize=(8, 5))
plt.plot(np.log(train_losses_unrolled), label="Log Train Loss (Unrolled)")
plt.plot(np.log(val_losses_unrolled), label="Log Validation Loss (Unrolled)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve (Unrolled)")
plt.legend()
plt.grid(True)
plt.savefig(f"assets/Training Loss and Validation Loss Unrolled.png")
plt.close()

train_losses = []
val_losses = []
with open(output_file_path_with_attention, "r") as f:
    for line in f:
        if "Train Loss" in line and "Val Loss" in line:
            parts = line.strip().split(",")
            train_part = parts[1].strip()
            train_loss = float(train_part.split(":")[1].strip())
            val_part = parts[2].strip()
            val_loss = float(val_part.split(":")[1].strip())

            train_losses.append(train_loss)
            val_losses.append(val_loss)

plt.figure(figsize=(8, 5))
plt.plot(np.log(train_losses), label="Log Train Loss")
plt.plot(np.log(val_losses), label="Log Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve with attention")
plt.legend()
plt.grid(True)
plt.savefig(f"assets/Training Loss and Validation Loss with attention.png")
plt.close()

train_losses = []
val_losses = []
with open(output_file_path_One_unet, "r") as f:
    for line in f:
        if "Train Loss" in line and "Val Loss" in line:
            parts = line.strip().split(",")
            train_part = parts[1].strip()
            train_loss = float(train_part.split(":")[1].strip())
            val_part = parts[2].strip()
            val_loss = float(val_part.split(":")[1].strip())

            train_losses.append(train_loss)
            val_losses.append(val_loss)

plt.figure(figsize=(8, 5))
plt.plot(np.log(train_losses), label="Log Train Loss")
plt.plot(np.log(val_losses), label="Log Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve one unet")
plt.legend()
plt.grid(True)
plt.savefig(f"assets/Training Loss and Validation Loss one unet.png")
plt.close()
