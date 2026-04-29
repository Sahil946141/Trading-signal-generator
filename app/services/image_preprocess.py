import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Enhanced preprocessor for chart images with data augmentation.
    
    Handles:
    - Loading images (JPG, PNG, etc.)
    - Resizing to 224x224 (EfficientNet standard)
    - Advanced data augmentation for training
    - Normalization using ImageNet statistics
    - Tensor conversion for PyTorch models
    """
    
    def __init__(self, target_size: tuple = (224, 224), is_training: bool = False):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            is_training: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.is_training = is_training
        
        # Base transforms
        base_transforms = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean (R, G, B)
                std=[0.229, 0.224, 0.225]    # ImageNet std (R, G, B)
            )
        ]
        
        if is_training:
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Slightly larger for random crop
                transforms.RandomCrop(target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                # Add noise for robustness
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
            ])
        else:
            # Inference transforms (no augmentation)
            self.transform = transforms.Compose(base_transforms)
        
        logger.info(f"Initialized ImagePreprocessor with target size {target_size}, training: {is_training}")
    
    def set_training_mode(self, is_training: bool):
        """Switch between training and inference modes."""
        self.is_training = is_training
        self.__init__(self.target_size, is_training)
    
    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor (1, 3, 224, 224)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            PIL.UnidentifiedImageError: If image format is invalid
        """
        try:
            # Load image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Convert to RGB (handles RGBA, grayscale, etc.)
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
            batch = tensor.unsqueeze(0)
            
            logger.debug(f"Preprocessed image shape: {batch.shape}")
            return batch
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def preprocess_batch(self, image_paths: list) -> torch.Tensor:
        """
        Preprocess multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            torch.Tensor: Batch of images (batch_size, 3, 224, 224)
        """
        batch = []
        for path in image_paths:
            tensor = self.preprocess(path)
            batch.append(tensor)
        
        return torch.cat(batch, dim=0)

    def preprocess_tensorflow(self, image_path: str, target_size: tuple = (128, 128)) -> np.ndarray:
        """
        Preprocess a single image for TensorFlow/Keras chart models.

        Returns:
            np.ndarray: Image batch shaped (1, height, width, 4) with float32 pixel values.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = image.resize(target_size)
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[:, :, :3]
        return np.expand_dims(array, axis=0)
    
    def create_test_time_augmentation(self, image_path: str, num_augmentations: int = 5) -> torch.Tensor:
        """
        Create multiple augmented versions for test-time augmentation.
        
        Args:
            image_path: Path to image file
            num_augmentations: Number of augmented versions to create
        
        Returns:
            torch.Tensor: Batch of augmented images (num_augmentations, 3, 224, 224)
        """
        # Temporarily switch to training mode for augmentation
        original_mode = self.is_training
        self.set_training_mode(True)
        
        augmented_batch = []
        for _ in range(num_augmentations):
            tensor = self.preprocess(image_path)
            augmented_batch.append(tensor)
        
        # Restore original mode
        self.set_training_mode(original_mode)
        
        return torch.cat(augmented_batch, dim=0)
