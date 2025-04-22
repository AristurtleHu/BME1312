import matplotlib.pyplot as plt

train_losses = []
val_losses = []

with open('.\output\output.txt', 'r') as f:
    for line in f:
        if 'Train Loss' in line and 'Val Loss' in line:
            parts = line.strip().split(',')
            train_part = parts[1].strip()
            train_loss = float(train_part.split(':')[1].strip())
            val_part = parts[2].strip()
            val_loss = float(val_part.split(':')[1].strip())
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(f'Training Loss and Validation Loss.png')
plt.close()