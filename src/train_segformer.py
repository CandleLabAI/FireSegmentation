import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import SegformerFeatureExtractor, SegformerConfig, SegformerModel, SegformerPreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import evaluate
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import numpy as np
import math
import cv2
from PIL import Image


# Define custom model components
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.lateral = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        skip = self.lateral(skip)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class BuildFPN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_depths = [32, 64]
        self.e1 = encoder_block(4, self.attention_depths[0])
        self.e2 = encoder_block(self.attention_depths[0], self.attention_depths[1])
        self.d2 = decoder_block(config.decoder_hidden_size, self.attention_depths[1])
        self.d1 = decoder_block(self.attention_depths[1], self.attention_depths[0])
        self.outputs = nn.Conv2d(self.attention_depths[0], config.num_labels, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(64, 768, kernel_size=1, padding=0)

    def forward(self, pixel_values: torch.FloatTensor, segformer_output: torch.FloatTensor) -> torch.Tensor:
        s1, p1 = self.e1(pixel_values)
        s2, p2 = self.e2(p1)
        unet_output = self.conv(p2)
        segformer_output = segformer_output + unet_output  # Skip Connection
        d2 = self.d2(segformer_output, s2)
        d1 = self.d1(d2, s1)
        outputs = self.outputs(d1)
        return outputs


class CustomSegformerDecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # linear layers which will unify the channel dimension
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = nn.Linear(config.hidden_sizes[i], config.decoder_hidden_size)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.config = config
        self.build_fpn = BuildFPN(self.config)

    def forward(self, encoder_hidden_states, pixel_values):
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                    encoder_hidden_state.shape[1])
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        output = self.build_fpn(pixel_values, hidden_states)
        return output


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


class CustomSegformerForSemanticSegmentation(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = CustomSegformerDecodeHead(config)
        self.post_init()

    def forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        pixel_values = pixel_values.permute(0, 3, 1, 2)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True  # We need the intermediate hidden states

        outputs = self.segformer(
            pixel_values[:, 0:4, :, :],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        logits = self.decode_head(encoder_hidden_states, pixel_values[:, 4:8, :, :])

        loss = None
        if labels is not None:
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            if self.config.num_labels == 1:
                loss_fct = FocalTverskyLoss()
                loss = loss_fct(upsampled_logits, labels)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


# Dataset class remains the same
class FireSegmentationDataset(Dataset):
    """Fire segmentation dataset for 8-channel images."""

    def __init__(self, root_dir, feature_extractor, split='train'):
        self.root_dir = Path(root_dir) / split
        self.feature_extractor = feature_extractor

        self.multichannel_files = sorted(list(self.root_dir.glob('multichannel/*.npy')))
        self.mask_files = sorted(list(self.root_dir.glob('masks/*.png')))

    def __len__(self):
        return len(self.multichannel_files)

    def __getitem__(self, idx):
        multichannel = np.load(str(self.multichannel_files[idx]))
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.int64)

        multichannel = torch.from_numpy(multichannel).float()
        mask = torch.from_numpy(mask).long()

        return {
            'pixel_values': multichannel,
            'labels': mask
        }


class FireSegformerModel(pl.LightningModule):
    def __init__(self, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        super().__init__()

        config = SegformerConfig(
            num_channels=4,
            num_encoder_blocks=4,
            num_labels=1,
            reshape_last_stage=False,
            decoder_hidden_size=768,
            semantic_loss_ignore_index=255
        )

        self.model = CustomSegformerForSemanticSegmentation(config)

        self.train_iou = evaluate.load("mean_iou")
        self.val_iou = evaluate.load("mean_iou")
        self.test_iou = evaluate.load("mean_iou")

        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['pixel_values'], batch['labels'])

        loss = outputs.loss
        logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=batch['labels'].shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = (torch.sigmoid(upsampled_logits) > 0.5).long()

        # Convert to numpy and ensure correct shape
        predictions = predicted.detach().cpu().numpy().squeeze()
        references = batch['labels'].detach().cpu().numpy().squeeze()

        # Ensure we have 2D arrays
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if references.ndim == 3 and references.shape[0] == 1:
            references = references[0]

        self.train_iou.add_batch(
            predictions=predictions,
            references=references
        )

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Compute metrics with required arguments
        metrics = self.train_iou.compute(
            num_labels=2,  # binary classification: background (0) and fire (1)
            ignore_index=255,  # ignore_index from config
            reduce_labels=False  # don't reduce labels to consecutive integers
        )

        # Log metrics
        self.log('train_mean_iou', metrics['mean_iou'], prog_bar=True)
        self.log('train_mean_accuracy', metrics['mean_accuracy'], prog_bar=True)

        # Reset metric for next epoch
        self.train_iou = evaluate.load("mean_iou")

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['pixel_values'], batch['labels'])

        loss = outputs.loss
        logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=batch['labels'].shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Get predictions
        predicted = (torch.sigmoid(upsampled_logits) > 0.5).long()

        # Convert to numpy and ensure correct shape
        predictions = predicted.detach().cpu().numpy().squeeze()  # Remove batch and channel dimensions
        references = batch['labels'].detach().cpu().numpy().squeeze()

        # Ensure we have 2D arrays
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if references.ndim == 3 and references.shape[0] == 1:
            references = references[0]

        # Add batch to metric
        self.val_iou.add_batch(
            predictions=predictions,
            references=references
        )

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute metrics with required arguments
        metrics = self.val_iou.compute(
            num_labels=2,  # binary classification: background (0) and fire (1)
            ignore_index=255,  # ignore_index from config
            reduce_labels=False  # don't reduce labels to consecutive integers
        )

        # Log metrics
        self.log('val_mean_iou', metrics['mean_iou'], prog_bar=True)
        self.log('val_mean_accuracy', metrics['mean_accuracy'], prog_bar=True)

        # Reset metric for next epoch
        self.val_iou = evaluate.load("mean_iou")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


def main():
    dataset_path = '../fire_detection_dataset'

    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        reduce_labels=False,
        size=512,
        do_normalize=False
    )

    # Create datasets and dataloaders
    train_dataset = FireSegmentationDataset(dataset_path, feature_extractor, 'train')
    val_dataset = FireSegmentationDataset(dataset_path, feature_extractor, 'val')
    test_dataset = FireSegmentationDataset(dataset_path, feature_extractor, 'test')

    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    model = FireSegformerModel(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='fire-segformer-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=100,
        callbacks=callbacks,
        precision=16,  # Use mixed precision training
        gradient_clip_val=0.5,  # Add gradient clipping
        accumulate_grad_batches=2,  # Gradient accumulation for larger effective batch size
    )

    # Train model
    trainer.fit(model)

    # Test model
    trainer.test(model)


if __name__ == "__main__":
    main()