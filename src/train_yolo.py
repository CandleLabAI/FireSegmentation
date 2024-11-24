from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import logging
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FireDetectionTrainer:
    def __init__(self, dataset_path='../fire_detection_dataset'):
        self.dataset_path = Path(dataset_path)
        self.config_path = self.dataset_path / 'dataset.yaml'
        self.runs_dir = Path('../runs')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Training hyperparameters
        self.hp = {
            'epochs': 100,
            'imgsz': 512,
            'batch_size': 16,
            'patience': 20,  # Early stopping patience
            'workers': 8,  # Number of worker threads
            'pretrained': True,  # Use pretrained weights
            'optimizer': 'Adam',  # Optimizer
            'lr0': 0.001,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate fraction
            'momentum': 0.937,  # SGD momentum/Adam beta1
            'weight_decay': 0.0005,  # Weight decay coefficient
            'warmup_epochs': 3,  # Warmup epochs
            'warmup_momentum': 0.8,  # Warmup initial momentum
            'warmup_bias_lr': 0.1,  # Warmup initial bias lr
            'close_mosaic': 10,  # Disable mosaic augmentation last N epochs
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Classification loss gain
        }

    def create_dataset_config(self):
        """Create YAML configuration file for the dataset."""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {
                0: 'fire'
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logging.info(f"Created dataset config at {self.config_path}")
        return self.config_path

    def train_model(self, model_size='n', resume=False):
        """
        Train YOLOv8 model.

        Args:
            model_size (str): Model size ('n', 's', 'm', 'l', 'x')
            resume (bool): Resume training from last checkpoint
        """
        try:
            # Create timestamp for run identification
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure config exists
            if not self.config_path.exists():
                self.create_dataset_config()

            # Initialize model
            model_name = f"yolov8{model_size}.pt"
            if self.hp['pretrained']:
                logging.info(f"Loading pretrained model: {model_name}")
                model = YOLO(model_name)
            else:
                logging.info("Training from scratch")
                model = YOLO(model_name).load_config()

            # Training arguments
            train_args = {
                'data': str(self.config_path),
                'epochs': self.hp['epochs'],
                'imgsz': self.hp['imgsz'],
                'batch': self.hp['batch_size'],
                'device': self.device,
                'workers': self.hp['workers'],
                'patience': self.hp['patience'],
                'optimizer': self.hp['optimizer'],
                'lr0': self.hp['lr0'],
                'lrf': self.hp['lrf'],
                'momentum': self.hp['momentum'],
                'weight_decay': self.hp['weight_decay'],
                'warmup_epochs': self.hp['warmup_epochs'],
                'warmup_momentum': self.hp['warmup_momentum'],
                'warmup_bias_lr': self.hp['warmup_bias_lr'],
                'close_mosaic': self.hp['close_mosaic'],
                'box': self.hp['box'],
                'cls': self.hp['cls'],
                'name': f'fire_detection_{timestamp}',
                'exist_ok': False,  # Increment run name if exists
                'pretrained': self.hp['pretrained'],
                'resume': resume
            }

            # Start training
            logging.info("Starting training with parameters:")
            for k, v in train_args.items():
                logging.info(f"{k}: {v}")

            results = model.train(**train_args)

            # Log training completion
            logging.info("Training completed successfully")
            logging.info(f"Results saved to {self.runs_dir}")

            return results

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def validate_model(self, weights_path):
        """
        Validate trained model on the test set.

        Args:
            weights_path (str): Path to trained weights
        """
        try:
            logging.info(f"Loading model from {weights_path}")
            model = YOLO(weights_path)

            # Run validation
            logging.info("Starting validation on test set")
            results = model.val(
                data=str(self.config_path),
                split='test',
                imgsz=self.hp['imgsz'],
                batch=self.hp['batch_size'],
                device=self.device,
            )

            logging.info("Validation completed")
            return results

        except Exception as e:
            logging.error(f"Error during validation: {str(e)}")
            raise


def main():
    # Create trainer instance
    trainer = FireDetectionTrainer()

    # Training configuration options
    training_configs = [
        {'model_size': 'n', 'epochs': 100, 'imgsz': 512},  # Fast training with small model
        # {'model_size': 's', 'epochs': 150, 'imgsz': 640},  # Medium training
        # {'model_size': 'm', 'epochs': 200, 'imgsz': 640},  # Full training
    ]

    # Run training for each configuration
    for config in training_configs:
        logging.info(f"\nStarting training with configuration: {config}")

        # Update hyperparameters
        trainer.hp.update({
            'epochs': config['epochs'],
            'imgsz': config['imgsz']
        })

        # Train model
        results = trainer.train_model(model_size=config['model_size'])

        # Get best weights path
        best_weights = trainer.runs_dir / 'detect' / results.save_dir.stem / 'weights' / 'best.pt'

        # Validate model
        if best_weights.exists():
            val_results = trainer.validate_model(str(best_weights))
            logging.info(f"Validation results: {val_results}")


if __name__ == "__main__":
    main()