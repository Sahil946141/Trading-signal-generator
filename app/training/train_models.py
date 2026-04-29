#!/usr/bin/env python3
"""
Training script for both CNN pattern detection and LSTM price prediction models.

This script provides comprehensive training with:
- Data loading and preprocessing
- Model training with advanced techniques
- Validation and evaluation
- Model saving and checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from pathlib import Path
import logging
from typing import Tuple, List
import argparse
import matplotlib.pyplot as plt

# Import our models
from app.models.cnn_pattern_model import ImprovedPatternDetectionModel
from app.models.lstm_signal_model import StockLSTMModel
from app.services.image_preprocess import ImagePreprocessor
from app.config import config
from app.training.dataset_preparation import load_prepared_dataset, prepare_lstm_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternDataset(Dataset):
    """Dataset for pattern recognition training."""
    
    def __init__(self, image_paths: List[str], labels: List[int], preprocessor: ImagePreprocessor):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_tensor = self.preprocessor.preprocess(self.image_paths[idx]).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label

class ModelTrainer:
    """Comprehensive model trainer for both CNN and LSTM models."""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def train_cnn_model(
        self,
        train_data_path: str,
        val_data_path: str = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_path: str = "models/best_cnn_model.pth"
    ):
        """
        Train the CNN pattern detection model.
        
        Args:
            train_data_path: Path to training data directory
            val_data_path: Path to validation data directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            save_path: Path to save the best model
        """
        logger.info("Starting CNN model training...")
        
        # Initialize model
        model = ImprovedPatternDetectionModel().to(self.device)
        
        # Load data
        train_loader, val_loader = self._load_pattern_data(
            train_data_path, val_data_path, batch_size
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training loop
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 50 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation phase
            val_acc = self._validate_cnn(model, val_loader)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Record metrics
            train_losses.append(train_loss / len(train_loader))
            val_accuracies.append(val_acc)
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            train_acc = 100. * train_correct / train_total
            logger.info(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_accuracies, "CNN Training")
        
        logger.info(f"CNN training completed. Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def train_lstm_model(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_path: str = "models/stock_lstm.h5"
    ):
        """Train the stock LSTM classifier and save it as a `.h5` model."""
        logger.info("Starting stock LSTM model training...")

        X_train, y_train = train_data
        X_val, y_val = val_data if val_data else (X_train, y_train)

        model = StockLSTMModel(
            sequence_length=X_train.shape[1],
            input_features=X_train.shape[2],
            model_path=save_path,
        )
        artifacts_dir = Path(save_path).with_suffix("")
        results = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path=save_path,
            artifacts_dir=str(artifacts_dir),
        )
        logger.info(
            "Stock LSTM training completed. Best validation accuracy: %.2f%%",
            results["best_validation_accuracy"],
        )
        return results
    
    def _load_pattern_data(self, train_path: str, val_path: str, batch_size: int):
        """Load pattern recognition data from a standard folder layout."""
        from torchvision.datasets import ImageFolder

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data path not found: {train_path}")

        train_dataset = ImageFolder(train_path, transform=ImagePreprocessor(target_size=(224, 224), is_training=True).transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(self.device == 'cuda'),
        )

        if val_path:
            if not os.path.exists(val_path):
                raise FileNotFoundError(f"Validation data path not found: {val_path}")
            val_dataset = ImageFolder(val_path, transform=ImagePreprocessor(target_size=(224, 224), is_training=False).transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=(self.device == 'cuda'),
            )
        else:
            val_loader = train_loader

        return train_loader, val_loader
    
    def _validate_cnn(self, model, val_loader):
        """Validate CNN model."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total if total > 0 else 0.0

    def _plot_training_curves(self, losses: List[float], accuracies: List[float], title: str):
        """Plot basic training curves for the CNN trainer."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(losses)
        ax1.set_title(f'{title} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(accuracies)
        ax2.set_title(f'{title} - Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.savefig(f'training_curves_{title.lower().replace(" ", "_")}.png')
        plt.close(fig)
    
def load_or_prepare_lstm_training_data(
    dataset_dir: str,
    symbol: str = None,
    symbols: str = None,
    start_date: str = None,
    end_date: str = None,
    timeframe: str = None,
    sequence_length: int = None,
    lookahead: int = None,
    threshold: float = None,
    train_split: float = None,
    feature_column: str = None,
    normalization: str = None,
):
    """Load a prepared dataset or build one from stock history."""
    dataset_path = Path(dataset_dir)

    expected_files = [
        dataset_path / "X_train.npy",
        dataset_path / "y_train.npy",
        dataset_path / "X_val.npy",
        dataset_path / "y_val.npy",
    ]

    if not all(file_path.exists() for file_path in expected_files):
        summary = prepare_lstm_dataset(
            output_dir=str(dataset_path),
            symbol=symbol or config.DEFAULT_STOCK_SYMBOL,
            symbols=symbols,
            start_date=start_date or config.STOCK_TRAINING_START,
            end_date=end_date,
            timeframe=timeframe or config.DEFAULT_TRAINING_TIMEFRAME,
            sequence_length=sequence_length or config.LSTM_SEQUENCE_LENGTH,
            lookahead=lookahead or config.LABEL_LOOKAHEAD,
            threshold=threshold if threshold is not None else config.LABEL_THRESHOLD,
            train_split=train_split or config.TRAIN_SPLIT,
            feature_column=feature_column or config.LSTM_FEATURE_COLUMN,
            normalization=normalization or config.LSTM_SEQUENCE_NORMALIZATION,
        )
    else:
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        else:
            summary = {"output_dir": str(dataset_path)}

    train_data, val_data = load_prepared_dataset(str(dataset_path))
    return train_data, val_data, summary

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CNN and LSTM models')
    parser.add_argument('--model', choices=['cnn', 'lstm', 'both'], default='both',
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_data', type=str, default=os.path.join(config.DATA_DIR, "lstm_dataset"),
                        help='Path to training data or prepared LSTM dataset directory')
    parser.add_argument('--val_data', type=str,
                        help='Path to validation data')
    parser.add_argument('--symbol', type=str, default=config.DEFAULT_STOCK_SYMBOL,
                        help='Primary stock symbol for training, e.g. RELIANCE.NS')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated stock symbols for multi-stock training')
    parser.add_argument('--start_date', type=str, default=config.STOCK_TRAINING_START,
                        help='Start date for stock dataset preparation')
    parser.add_argument('--end_date', type=str, default=None,
                        help='Optional end date for stock dataset preparation')
    parser.add_argument('--timeframe', type=str, default=config.DEFAULT_TRAINING_TIMEFRAME,
                        help='Candle timeframe used for training data')
    parser.add_argument('--sequence_length', type=int, default=config.LSTM_SEQUENCE_LENGTH,
                        help='Sequence length for the LSTM window')
    parser.add_argument('--lookahead', type=int, default=config.LABEL_LOOKAHEAD,
                        help='Forward candles used for BUY/SELL/HOLD labeling')
    parser.add_argument('--threshold', type=float, default=config.LABEL_THRESHOLD,
                        help='Future-move threshold used for BUY/SELL/HOLD labels')
    parser.add_argument('--train_split', type=float, default=config.TRAIN_SPLIT,
                        help='Chronological train split fraction')
    parser.add_argument('--save_path', type=str, default=config.LSTM_MODEL_PATH,
                        help='Where to save the trained LSTM model')
    parser.add_argument('--feature_column', type=str, default=config.LSTM_FEATURE_COLUMN,
                        help='Feature column for the LSTM: return, log_return, or zscore_20')
    parser.add_argument('--normalization', type=str, default=config.LSTM_SEQUENCE_NORMALIZATION,
                        help='Sequence normalization: none, zscore, or minmax')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.model in ['cnn', 'both']:
        logger.info("Training CNN model...")
        if not args.train_data:
            raise ValueError("--train_data is required for CNN training")
        
        trainer.train_cnn_model(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    
    if args.model in ['lstm', 'both']:
        logger.info("Training LSTM model...")
        train_data, val_data, summary = load_or_prepare_lstm_training_data(
            dataset_dir=args.train_data,
            symbol=args.symbol,
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            sequence_length=args.sequence_length,
            lookahead=args.lookahead,
            threshold=args.threshold,
            train_split=args.train_split,
            feature_column=args.feature_column,
            normalization=args.normalization,
        )

        logger.info(
            "Prepared LSTM dataset with %s train sequences and %s validation sequences",
            len(train_data[1]),
            len(val_data[1]),
        )

        trainer.train_lstm_model(
            train_data=train_data,
            val_data=val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_path=args.save_path,
        )

if __name__ == "__main__":
    main()
