---
marp: true
theme: default
paginate: true
---

# MRI Reconstruction with Deep Learning

**BME1312 Project 1**

---

## Introduction: MRI Undersampling

*   **Goal:** Accelerate MRI acquisition by acquiring fewer k-space samples.
*   **Challenge:** Undersampling leads to aliasing artifacts in the reconstructed image.
*   **Solution:** Use deep learning to reconstruct high-quality images from undersampled data.

---

## Data Processing

1.  **Load Data:** Load the `cine.npz` dataset containing fully sampled MRI frames.
2.  **Generate Mask:** Create an undersampling mask.
    *   Variable density pattern.
    *   Preserves central k-space lines.
    *   Random sampling in outer regions.
3.  **Apply Mask:** Multiply the k-space representation of the fully sampled data by the mask.
4.  **Convert to Image Space:** Apply inverse Fourier Transform (iFFT) to get the undersampled, aliased images.
5.  **Format Data:** Convert complex images to pseudo-real format for model input.

---

## Variable Density Mask Generation

```python
def variable_density_mask(shape, acceleration=5, center_lines=11, seed=42):
    # ... (Implementation details) ...
    # Ensures central lines are sampled
    # Samples outer lines with decreasing probability
    # ...
    return mask
```

*   Generates a mask to simulate undersampling.
*   `acceleration`: Controls the degree of undersampling.
*   `center_lines`: Number of fully sampled central lines.
*   Visualized mask shows sampling pattern in k-space and over time

---

![Undersampling Mask](../assets/undersampling_mask.png)

---

## Model Architecture 1: UNet

*   Encoder-Decoder structure with skip connections.
*   Captures multi-scale features.
*   **Modifications:**
    *   LeakyReLU activation.
    *   Dropout for regularization.
    *   Batch Normalization.
    *   Attention mechanism in the bottleneck.
    *   Kaiming weight initialization.

---

## UNet: Attention Mechanism

*   Combines Channel Attention and Spatial Attention.
*   **Channel Attention:** Focuses on 'what' features are important. Uses AvgPool and MaxPool along channels.
*   **Spatial Attention:** Focuses on 'where' features are important. Uses AvgPool and MaxPool along spatial dimensions.
*   Helps the model focus on relevant parts of the input.

```python
    def _attention_block(self, features):
        # ... Channel and Spatial Attention implementation ...
```

---

## Model Architecture 2: 3D ResNet

*   Processes the temporal dimension along with spatial dimensions.
*   Uses 3D convolutions and residual blocks.
*   Designed to capture spatio-temporal correlations.
*   `BasicBlock`: Standard residual block with 3x3x3 convolutions.

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, block_inplanes, ...):
        # ... 3D Conv layers, Residual blocks ...
    def forward(self, x):
        # ... Forward pass through ResNet layers ...

def resnet18(**kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], ...) # Example configuration
    return model
```

---

## Combined Model Approach

1.  **Input:** Pseudo-real undersampled images (real and imaginary parts as separate channels).
2.  **UNet 1 (`model`):** Processes the 'real' part channels.
3.  **UNet 2 (`model2`):** Processes the 'imaginary' part channels.
4.  **Stack Outputs:** Combine the outputs of the two UNets.
5.  **ResNet (`model3`):** Takes the combined UNet outputs (now treated as spatio-temporal data) and performs final reconstruction.

```python
# Forward pass logic in train/evaluate/test
outputs1 = model(x[:, :, 0]) # Process real part
outputs2 = model2(x[:, :, 1]) # Process imaginary part
tmp = torch.stack((outputs1, outputs2), dim=2) # Combine
outputs = model3(lab.pseudo2real(tmp).unsqueeze(2)).squeeze(2) # ResNet processing
```

---

## Training Setup

*   **Data Splitting:** Train (114), Validation (29), Test (57) sets.
*   **Loss Function:** Mean Squared Error (MSE) or L1 Loss (`criterion`).
*   **Optimizer:** Adam optimizer with weight decay.
*   **Learning Rate Scheduler:** Warmup phase followed by cosine decay.
    ```python
    lr = lr_scheduler(epoch, warmup_epochs, warmup_lr, initial_lr, num_epochs)
    ```
*   **Device:** Uses CUDA (GPU) if available.
*   **Logging:** TensorBoard for tracking loss curves. Output logs saved to `output/output.txt`.

---

## Training Loop

```python
for epoch in range(num_epochs):
    # Adjust learning rate
    lr = lr_scheduler(...)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Training Phase
    train_loss = train_epoch(model, model2, model3, dataloader_train, optimizer, criterion)

    # Validation Phase
    val_loss = evaluate(model, model2, model3, dataloader_val, criterion)

    # Logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
    # Save log to file...
```

---

## Evaluation and Testing

*   **Evaluation (`evaluate`):** Calculates loss on the validation set during training (no backpropagation).
*   **Testing (`test_models`):**
    *   Calculates final performance on the unseen test set.
    *   Metrics:
        *   Loss (MSE or L1)
        *   Peak Signal-to-Noise Ratio (PSNR)
        *   Structural Similarity Index Measure (SSIM)
    *   Saves reconstructed images for visual comparison.

---

## Visualization (`imsshow`)

*   Utility function to display multiple images in a grid.
*   Used to save: Undersampled, Fully sampled, reconstructedimages.

![Comparison Example](../assets/comparison_image_0.png) <!-- Add one of the comparison images generated by process_data -->

---

## Execution & Parameters

*   The `train` function orchestrates the entire process.
*   Key hyperparameters used in the final run:
    *   `in_channels`: 20 (10 frames x 2 for pseudo-real)
    *   `out_channels`: 20
    *   `init_features` (UNet): 64
    *   `num_epochs`: 800
    *   `weight_decay`: 1e-4
    *   `batch_size`: 10
    *   `initial_lr`: 1e-4
    *   `loss_tpe`: "L2" (MSE Loss)

---

## Results & Conclusion

*   The combined UNet-ResNet model was trained to reconstruct MRI images from undersampled k-space data.
*   Training progress and validation loss were monitored.
*   Final performance was evaluated using Loss, PSNR, and SSIM on the test set.
*   Visualizations confirm the model's ability to reduce aliasing artifacts.
   
```txt
Loss: mean = 0.00134656, std = 0.00054760
PSNR: mean = 29.08446121, std = 1.93235576
SSIM: mean = 0.84434632, std = 0.03711063
```