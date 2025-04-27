import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from bme1312 import lab1 as lab
from bme1312.utils import (
    image2kspace,
    kspace2image,
    pseudo2real,
    pseudo2complex,
    complex2pseudo,
)
from torch.utils.tensorboard import SummaryWriter
from torch.fft import ifft2, fft2
from CS_mask import cartesian_mask

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import argparse
import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import math
from functools import partial

from bme1312.utils import lr_scheduler, compute_psnr, compute_ssim


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features

        # Improved dropout strategy with slightly lower rate
        self.dropout = nn.Dropout(p=0.3)

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add attention mechanism to bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.attention = self._attention_block(features * 16)

        # Upsampling path with skip connections
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # Add a deeper output layer
        self.pre_conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # Initialize weights for better training
        self._initialize_weights()

    def forward(self, x):
        batch_size = x.shape[0]

        # Encoder path
        enc1 = self.encoder1(x)
        enc1_shape = enc1.shape  # Debug shape
        enc1 = self.dropout(enc1)

        # Apply pooling and get encoder features
        pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        enc2_shape = enc2.shape  # Debug shape
        enc2 = self.dropout(enc2)

        pool2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        enc3_shape = enc3.shape  # Debug shape

        pool3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        enc4_shape = enc4.shape  # Debug shape
        enc4 = self.dropout(enc4)

        # Bottleneck
        pool4 = self.pool4(enc4)
        bottleneck = self.bottleneck(pool4)
        bottleneck_shape = bottleneck.shape  # Debug shape
        bottleneck = self.attention(bottleneck)

        # Decoder path - make sure shapes are compatible for skip connections
        dec4 = self.upconv4(bottleneck)
        dec4_shape = dec4.shape  # Debug shape

        # Check for dimension mismatch and fix if necessary
        if dec4.shape[2:] != enc4.shape[2:]:
            # Resize dec4 to match spatial dimensions of enc4
            dec4 = F.interpolate(
                dec4, size=enc4.shape[2:], mode="bilinear", align_corners=True
            )

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.shape[2:] != enc3.shape[2:]:
            # Resize dec3 to match spatial dimensions of enc3
            dec3 = F.interpolate(
                dec3, size=enc3.shape[2:], mode="bilinear", align_corners=True
            )

        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            # Resize dec2 to match spatial dimensions of enc2
            dec2 = F.interpolate(
                dec2, size=enc2.shape[2:], mode="bilinear", align_corners=True
            )

        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            # Resize dec1 to match spatial dimensions of enc1
            dec1 = F.interpolate(
                dec1, size=enc1.shape[2:], mode="bilinear", align_corners=True
            )

        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Apply final convolutions
        out = self.activation(self.pre_conv(dec1))
        return torch.sigmoid(self.conv(out))

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
                ]
            )
        )

    def _attention_block(self, features):
        # Create an actual Module instead of just a function
        class AttentionModule(nn.Module):
            def __init__(self, features):
                super().__init__()
                # Channel Attention Module (SE-like)
                self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
                self.channel_max_pool = nn.AdaptiveMaxPool2d(1)

                self.channel_shared_mlp = nn.Sequential(
                    nn.Conv2d(features, features // 4, kernel_size=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Conv2d(features // 4, features, kernel_size=1, bias=False),
                )

                # Spatial Attention Module
                self.spatial_attention = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                # Channel attention
                avg_pool = self.channel_avg_pool(x)
                max_pool = self.channel_max_pool(x)

                avg_out = self.channel_shared_mlp(avg_pool)
                max_out = self.channel_shared_mlp(max_pool)

                channel_attention = torch.sigmoid(avg_out + max_out)
                x_channel = x * channel_attention

                # Spatial attention
                avg_out = torch.mean(x_channel, dim=1, keepdim=True)
                max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
                spatial_in = torch.cat([avg_out, max_out], dim=1)
                spatial_weight = self.spatial_attention(spatial_in)

                # Final output with both attention mechanisms applied
                return x_channel * spatial_weight

        # Create and return an instance of the attention module
        return AttentionModule(features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 3D ResNet implementation


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=20,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
    ):
        super().__init__()

        if isinstance(block_inplanes, int):
            block_inplanes = [
                block_inplanes,
                block_inplanes * 2,
                block_inplanes * 4,
                block_inplanes * 8,
            ]

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # Full 3D convolution
        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.conv2 = nn.Conv3d(self.in_planes, 20, kernel_size=1)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = F.interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=True
        )  # Upsample spatial dims
        x = self.conv2(x)
        return torch.sigmoid(x)


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], [32, 64, 128, 256], **kwargs)
    return model


def imsshow(
    imgs,
    flag,
    titles=None,
    num_col=5,
    dpi=100,
    cmap=None,
    is_colorbar=False,
    is_ticks=False,
    save_path=None,
):
    num_imgs = len(imgs)
    num_row = math.ceil(num_imgs / num_col)
    fig_width = num_col * 3
    if is_colorbar:
        fig_width += num_col * 1.5
    fig_height = num_row * 3
    fig = plt.figure(dpi=dpi, figsize=(fig_width, fig_height))
    for i in range(num_imgs):
        ax = plt.subplot(num_row, num_col, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap)
        if titles:
            plt.title(titles[i])
        if is_colorbar:
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.01,
                    ax.get_position().y0,
                    0.01,
                    ax.get_position().height,
                ]
            )
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    # plt.show()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.close("all")


def process_data():
    dataset = np.load("./cine.npz")["dataset"]
    labels = torch.Tensor(dataset) # Shape (N, 20, H, W)

    # Create variable density mask with acceleration factor 5
    mask = cartesian_mask(shape=(1, 20, 192, 192), acc=5)
    mask = torch.Tensor(mask)

    # Apply mask to k-space data
    inputs_k = lab.image2kspace(labels) # Shape (N, 20, H, W), complex
    inputs_k = inputs_k * mask
    inputs = lab.kspace2image(inputs_k) # Shape (N, 20, H, W), complex
    inputs = lab.complex2pseudo(inputs) # Shape (N, 20, 2, H, W)

    inputs2 = lab.pseudo2real(inputs).unsqueeze(2) # Shape (N, 20, 1, H, W) - Magnitude
    inputs = torch.cat((inputs, inputs2), dim=2) # Shape (N, 20, 3, H, W) - Real, Imag, Mag

    # Visualize original and undersampled images
    for i in range(min(3, labels.shape[0])):  # Show first 3 images
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(labels[i, 0].numpy(), cmap="gray")
        plt.title("Fully Sampled Image (Frame 0)")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        # Visualize magnitude of undersampled image
        plt.imshow(inputs[i, 0, 2].numpy(), cmap="gray")
        plt.title("Undersampled Image (Mag, Frame 0)")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(mask[0, 0].numpy()), cmap="gray")
        plt.title("Sampling Mask (Frame 0)")
        plt.colorbar()

        plt.savefig(f"assets/comparison_image_{i}.png")
        plt.close()

    return (inputs, labels, mask)


def train(
    in_channels, # Base channels (e.g., 20 for time frames)
    out_channels, # Base channels
    init_features,
    num_epochs,
    weight_decay,
    batch_size,
    initial_lr,
    loss_tpe,
):
    # Setup directories for saving results
    parser = argparse.ArgumentParser(
        description="Save images to specified folder structure."
    )
    parser.add_argument(
        "folder_name", type=str, help="Name of the folder to save images."
    )
    args = parser.parse_args()

    # Create directory structure
    base_dir = os.path.join("images", args.folder_name)
    sub_dirs = ["under_sampling", "full_sampling", "reconstruction"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

    # Create output directory for logging
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "one_unet.txt")

    # Load and prepare data
    inputs, labels, mask = process_data() # inputs shape (N, 20, 3, H, W), labels shape (N, 20, H, W)

    # Initialize models
    # Single UNet takes concatenated real+imag (2*in_channels) and outputs concatenated real+imag (2*out_channels)
    model = UNet(
        in_channels=in_channels * 2, out_channels=out_channels * 2, init_features=init_features
    )
    model3 = resnet18(n_input_channels=20) # ResNet takes magnitude derived from UNet output

    # Move data and models to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    labels = labels.to(device)
    model = model.to(device)
    model3 = model3.to(device)

    # Setup tensorboard writer
    writer = SummaryWriter()

    # Define loss function
    criterion = nn.MSELoss() if loss_tpe == "L2" else nn.L1Loss()

    # Setup optimizer (parameters from both models)
    param_list = (
        list(model.parameters()) + list(model3.parameters())
    )
    optimizer = optim.Adam(param_list, lr=initial_lr, weight_decay=weight_decay)

    # Prepare dataset splits
    dataset = TensorDataset(inputs, labels)
    train_size, val_size, test_size = 114, 29, 57
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False) # No shuffle for validation
    dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False) # No shuffle for test

    # Learning rate scheduling parameters
    warmup_epochs = int(0.05 * num_epochs)
    warmup_lr = 1e-3 * initial_lr

    # Training loop
    for epoch in range(num_epochs):
        # Learning rate scheduling
        lr = lr_scheduler(epoch, warmup_epochs, warmup_lr, initial_lr, num_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Training phase
        train_loss = train_epoch(
            model, model3, dataloader_train, optimizer, criterion
        )

        # Validation phase
        val_loss = evaluate(model, model3, dataloader_val, criterion)

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        log_message = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        print(log_message)

        # Save log to file
        with open(output_path, "a") as file:
            file.write(log_message + "\n")

    # Testing phase
    test_results = test_models(
        model, model3, dataloader_test, criterion, base_dir
    )

    # Log test results
    log_test_results(test_results, output_path)

    # Save models (saving UNet and ResNet separately might be better)
    save_model(model, "saved_unet_model")
    save_model(model3, "saved_resnet_model")


    writer.close()


def train_epoch(model, model3, dataloader, optimizer, criterion):
    """Run one training epoch"""
    model.train()
    model3.train()

    total_loss = 0
    for x, y in dataloader: # x shape: (B, 20, 3, H, W), y shape: (B, 20, H, W)
        # Prepare input for UNet (concatenate real and imaginary parts)
        # x[:, :, 0] -> Real part, shape (B, 20, H, W)
        # x[:, :, 1] -> Imaginary part, shape (B, 20, H, W)
        unet_input = torch.cat((x[:, :, 0], x[:, :, 1]), dim=1) # Shape: (B, 40, H, W)

        # Forward pass through the UNet
        unet_output = model(unet_input) # Shape: (B, 40, H, W)

        # Reshape UNet output to pseudo-complex format for ResNet
        # Target shape: (B, 20, 2, H, W)
        # B=x.shape[0], C=x.shape[1]=20, H=x.shape[3], W=x.shape[4]
        tmp = unet_output.view(x.shape[0], x.shape[1], 2, x.shape[3], x.shape[4])

        # Convert pseudo-complex output to magnitude and prepare for ResNet
        # lab.pseudo2real(tmp) -> Magnitude, shape (B, 20, H, W)
        # unsqueeze(2) -> Shape (B, 20, 1, H, W) - Adds depth dimension for Conv3d
        resnet_input = lab.pseudo2real(tmp).unsqueeze(2)

        # Forward pass through ResNet
        outputs = model3(resnet_input).squeeze(2) # Shape: (B, 20, H, W)

        # Calculate loss and backpropagate
        loss = criterion(outputs, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def evaluate(model, model3, dataloader, criterion):
    """Evaluate models on validation or test data"""
    model.eval()
    model3.eval()

    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader: # x shape: (B, 20, 3, H, W), y shape: (B, 20, H, W)
            # Prepare input for UNet
            unet_input = torch.cat((x[:, :, 0], x[:, :, 1]), dim=1) # Shape: (B, 40, H, W)

            # Forward pass through UNet
            unet_output = model(unet_input) # Shape: (B, 40, H, W)

            # Reshape UNet output to pseudo-complex format
            tmp = unet_output.view(x.shape[0], x.shape[1], 2, x.shape[3], x.shape[4]) # Shape: (B, 20, 2, H, W)

            # Prepare input for ResNet
            resnet_input = lab.pseudo2real(tmp).unsqueeze(2) # Shape: (B, 20, 1, H, W)

            # Forward pass through ResNet
            outputs = model3(resnet_input).squeeze(2) # Shape: (B, 20, H, W)

            total_loss += criterion(outputs, y).item()

    return total_loss / len(dataloader)


def test_models(model, model3, dataloader, criterion, base_dir):
    """Test models and save visualization results"""
    model.eval()
    model3.eval()

    losses, psnrs, ssims = [], [], []
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(dataloader, desc="Testing")): # x shape: (B, 20, 3, H, W), y shape: (B, 20, H, W)
            # Prepare input for UNet
            unet_input = torch.cat((x[:, :, 0], x[:, :, 1]), dim=1) # Shape: (B, 40, H, W)

            # Forward pass through UNet
            unet_output = model(unet_input) # Shape: (B, 40, H, W)

            # Reshape UNet output to pseudo-complex format
            tmp = unet_output.view(x.shape[0], x.shape[1], 2, x.shape[3], x.shape[4]) # Shape: (B, 20, 2, H, W)

            # Prepare input for ResNet
            resnet_input = lab.pseudo2real(tmp).unsqueeze(2) # Shape: (B, 20, 1, H, W)

            # Forward pass through ResNet
            outputs = model3(resnet_input).squeeze(2) # Shape: (B, 20, H, W)

            # Save sample visualizations (only first item in batch)
            if idx < 5: # Save visualizations for the first 5 batches
                 save_visualizations(x[0:1], y[0:1], outputs[0:1], idx, base_dir) # Pass batch dim 1

            # Calculate metrics for each item in the batch
            for i in range(x.size(0)):
                output_i = outputs[i:i+1] # Keep batch dimension for consistency if needed
                y_i = y[i:i+1]
                losses.append(criterion(output_i, y_i).item())
                # compute_psnr/ssim expect numpy arrays without batch dim
                psnrs.append(compute_psnr(output_i.squeeze(0).cpu().numpy(), y_i.squeeze(0).cpu().numpy()))
                ssims.append(compute_ssim(output_i.squeeze(0).cpu().numpy(), y_i.squeeze(0).cpu().numpy()))

    # Calculate statistics
    metrics = {
        "loss": {
            "mean": torch.mean(torch.tensor(losses)),
            "std": torch.std(torch.tensor(losses)),
        },
        "psnr": {
            "mean": torch.mean(torch.tensor(psnrs)),
            "std": torch.std(torch.tensor(psnrs)),
        },
        "ssim": {
            "mean": torch.mean(torch.tensor(ssims)),
            "std": torch.std(torch.tensor(ssims)),
        },
    }

    return metrics


def save_visualizations(x, y, outputs, idx, base_dir):
    """Save visualization images for the first item in the batch"""
    # x shape: (1, 20, 3, H, W), y shape: (1, 20, H, W), outputs shape: (1, 20, H, W)
    under_sampling_filename = os.path.join(
        base_dir, "under_sampling", f"under_sampling_batch{idx}.png"
    )
    full_sampling_filename = os.path.join(
        base_dir, "full_sampling", f"full_sampling_batch{idx}.png"
    )
    reconstruction_filename = os.path.join(
        base_dir, "reconstruction", f"reconstruction_batch{idx}.png"
    )

    # Use magnitude (channel 2) from input x for visualization
    imsshow(
        x[0, :, 2].cpu().numpy(), # Shape (20, H, W)
        num_col=5,
        cmap="gray",
        flag="under_sampling",
        is_colorbar=True,
        save_path=under_sampling_filename,
        titles=[f'Frame {i}' for i in range(x.shape[1])]
    )
    imsshow(
        y[0].cpu().numpy(), # Shape (20, H, W)
        num_col=5,
        cmap="gray",
        flag="full_sampling",
        is_colorbar=True,
        save_path=full_sampling_filename,
        titles=[f'Frame {i}' for i in range(y.shape[1])]
    )
    imsshow(
        outputs[0].cpu().numpy(), # Shape (20, H, W)
        num_col=5,
        cmap="gray",
        flag="reconstruction",
        is_colorbar=True,
        save_path=reconstruction_filename,
        titles=[f'Frame {i}' for i in range(outputs.shape[1])]
    )


def log_test_results(results, output_path):
    """Log test results to file"""
    with open(output_path, "a") as file:
        file.write("\n--- Test Results ---\n")
        loss_output = f'Loss: mean = {results["loss"]["mean"]:.8f}, std = {results["loss"]["std"]:.8f}'
        psnr_output = f'PSNR: mean = {results["psnr"]["mean"]:.8f}, std = {results["psnr"]["std"]:.8f}'
        ssim_output = f'SSIM: mean = {results["ssim"]["mean"]:.8f}, std = {results["ssim"]["std"]:.8f}'

        print(loss_output)
        print(psnr_output)
        print(ssim_output)

        file.write(loss_output + "\n")
        file.write(psnr_output + "\n")
        file.write(ssim_output + "\n")


def save_model(model, base_name):
    """Save model state dict with unique filename"""
    filename = f"./{base_name}.pth" # Use .pth extension
    i = 1
    # Avoid overwriting existing files by appending a number
    while os.path.exists(filename):
        filename = f"./{base_name}_{i}.pth"
        i += 1
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    # Example call to the train function
    # in_channels and out_channels refer to the base number of channels (e.g., time frames)
    # The UNet will internally use 2*in_channels and 2*out_channels
    train(
        in_channels=20,      # Number of time frames / base channels
        out_channels=20,     # Number of time frames / base channels
        init_features=64,    # Initial features for UNet
        num_epochs=800,      # Total training epochs
        weight_decay=1e-4,   # Optimizer weight decay
        batch_size=10,       # Batch size for training/validation/testing
        initial_lr=1e-4,     # Initial learning rate for Adam
        loss_tpe="L2",       # Type of loss function ('L1' or 'L2')
    )
