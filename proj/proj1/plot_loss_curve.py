import matplotlib
import platform
from pathlib import Path

if platform.system() == "Linux":
    matplotlib.use("Agg")  # Use a non-interactive backend only on Linux
import matplotlib.pyplot as plt


train_losses = []
val_losses = []

# Construct the path in an OS-independent way
output_file_path = Path("./output/output.txt")
output_file_path_L1 = Path("./output/L1_Loss_output.txt")
output_file_path_No_Opt = Path("./output/No_Opt_output.txt")

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
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
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
plt.plot(train_losses_L1, label="Train Loss (L1)")
plt.plot(val_losses_L1, label="Validation Loss (L1)")
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
plt.plot(train_losses_No_Opt, label="Train Loss (No Opt)")
plt.plot(val_losses_No_Opt, label="Validation Loss (No Opt)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve without Optimization")
plt.legend()
plt.grid(True)
plt.savefig(f"assets/Training Loss and Validation Loss No opt.png")
plt.close()
