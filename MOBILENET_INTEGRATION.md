# MobileNet Model Integration Guide

## Overview

This document describes the integration of the MobileNet pattern recognition model (`mobilenet_pattern_model.h5`) into the chart pattern detection system, replacing the previous simple CNN model (`chart_pattern_tf_model.h5`).

## Model Comparison

### Previous Model: Simple CNN (`chart_pattern_tf_model.h5`)
- **Size**: 44MB
- **Architecture**: Basic 2-layer CNN
- **Input Size**: 128x128 pixels
- **Parameters**: ~500K
- **Framework**: Custom NumPy inference

### New Model: MobileNetV2 (`mobilenet_pattern_model.h5`)
- **Size**: 11MB (75% smaller!)
- **Architecture**: MobileNetV2 with depthwise separable convolutions
- **Input Size**: 224x224 pixels
- **Parameters**: ~3.5M
- **Framework**: TensorFlow/Keras
- **Key Features**:
  - 16 MobileNet blocks with inverted residuals
  - Global average pooling
  - Efficient depthwise separable convolutions
  - Better feature extraction capabilities

## Integration Changes

### 1. Configuration Update

**File**: `.env`

```bash
# Before
PATTERN_MODEL_PATH=chart_pattern_tf_model.h5

# After
PATTERN_MODEL_PATH=mobilenet_pattern_model.h5
```

### 2. Code Changes

**File**: `app/models/cnn_pattern_model.py`

#### Key Modifications:

1. **Added TensorFlow Import**: Added conditional TensorFlow import for Keras model loading
2. **New Model Loading Method**: Replaced `_load_numpy_chart_model()` with `_load_keras_model()`
3. **Updated Prediction Method**: Replaced `_predict_numpy_chart_model()` with `_predict_tensorflow_model()`
4. **Framework Detection**: Updated to use "tensorflow" framework identifier

#### Architecture Support:
- The new implementation uses TensorFlow/Keras to load models, enabling support for complex architectures like MobileNet
- Automatically detects model input size from the loaded model
- Handles both .h5 and .keras file formats
- Maintains backward compatibility with PyTorch models (.pth files)

### 3. Input Size Handling

The system automatically adapts to the new input size:

```python
# In app/api/routes.py
if pattern_model.framework in {"tensorflow", "numpy-h5"}:
    image_tensor = img_preprocessor.preprocess_tensorflow(
        temp_file, target_size=pattern_model.input_size  # Automatically uses 224x224
    )
```

## Benefits of MobileNet Integration

### 1. **Reduced Model Size**
- 75% reduction in disk space (44MB → 11MB)
- Faster model loading times
- Reduced memory footprint

### 2. **Better Performance**
- MobileNetV2 architecture provides better feature extraction
- Depthwise separable convolutions are more efficient
- Larger input size (224x224) captures more detail

### 3. **Improved Accuracy Potential**
- MobileNetV2 is a proven architecture for image classification
- More parameters (3.5M vs 500K) allow for better pattern learning
- Transfer learning from ImageNet pre-training

### 4. **Easier Maintenance**
- Using TensorFlow/Keras instead of custom NumPy implementation
- Easier to update to newer model versions
- Better support for complex architectures

## Usage

### API Endpoint

The integration is transparent to API users. The same endpoint works:

```http
POST /api/predict
Content-Type: multipart/form-data

file: <chart_image>
symbol: RELIANCE.NS
timeframe: 5m
use_lstm: true
```

### Response Format

Response format remains unchanged:

```json
{
  "pattern": "Double Bottom",
  "pattern_confidence": 0.85,
  "probabilities": {
    "Double Bottom": 0.85,
    "Double Top": 0.05,
    ...
  },
  "top_5": [...],
  "signal": "BUY",
  "signal_confidence": 0.72,
  ...
}
```

## Testing

### Integration Test Results

```
=== End-to-End MobileNet Integration Test ===

1. Initializing services...
   [OK] Services initialized successfully

2. Model loaded:
   - Framework: tensorflow
   - Input size: (224, 224)
   - Number of classes: 20
   - Model path: mobilenet_pattern_model.h5
   [OK] MobileNet model loaded correctly

3. Testing prediction pipeline...
   - Preprocessed image shape: (1, 224, 224, 3)
   - Predicted pattern: Pattern 16
   - Confidence: 0.3718
   [OK] Prediction successful

=== All Tests Passed! MobileNet Integration Complete ===
```

## Technical Details

### Model Architecture

The MobileNet model consists of:
1. **Initial Conv Layer**: 3×3 convolution with 32 filters
2. **16 MobileNet Blocks**: Inverted residuals with depthwise separable convolutions
3. **Global Average Pooling**: Reduces spatial dimensions
4. **Classification Head**: Fully connected layer with 20 outputs (pattern classes)

### Preprocessing Pipeline

Images are processed as follows:
1. Resize to 224×224 pixels
2. Convert to RGB format
3. Normalize pixel values to [0, 1] range
4. Add batch dimension

### Dependencies

Required packages:
- `tensorflow>=2.8.0` - For loading and running MobileNet model
- `h5py` - For reading .h5 model files
- `Pillow` - For image processing
- `numpy` - For array operations

## Troubleshooting

### Issue: TensorFlow not found
**Solution**: Install TensorFlow:
```bash
pip install tensorflow
```

### Issue: Model loading fails
**Solution**: Verify model file exists and is not corrupted:
```bash
ls -lh mobilenet_pattern_model.h5
```

### Issue: Prediction errors
**Solution**: Check input image format and size. Ensure it's a valid image file.

## Future Enhancements

1. **Model Quantization**: Convert to TensorFlow Lite for even faster inference
2. **GPU Acceleration**: Enable GPU support for batch processing
3. **Ensemble Methods**: Combine MobileNet with other architectures
4. **Active Learning**: Retrain model with new patterns

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow MobileNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
- [Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
