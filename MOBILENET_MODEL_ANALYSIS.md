# MobileNet Model Issue Analysis

## Problem Identified

The MobileNet model (`mobilenet_pattern_model.h5`) is **severely biased** towards Class 17, predicting it for 94% of inputs regardless of the actual pattern.

## Statistical Evidence

### Class Distribution Analysis (100 random test images)
```
Class 17: 48.41% average probability (94/100 predictions)
Class 12: 22.05% average probability (6/100 predictions)
All other classes: < 8% average probability each
```

### Bias Indicators
- **Class 17** appears in 94% of random predictions
- **Only 2 classes** account for 99% of predictions
- **18 out of 20 classes** have < 3% average probability
- **Standard deviations are low**, indicating consistent bias

## Root Causes

### 1. Training Data Imbalance
The model was likely trained on highly imbalanced data where:
- Class 17 dominated the training set
- Other pattern classes were underrepresented
- The model learned to predict the majority class

### 2. Insufficient Training
- Model may not have converged properly
- Training was stopped too early
- Learning rate issues prevented proper learning

### 3. Poor Generalization
- Model memorized training patterns instead of learning features
- Overfitting to Class 17 patterns
- Lack of data augmentation or regularization

## Impact

### Current Behavior
- ❌ Model predicts "Pattern 17" for most chart images
- ❌ Confidence scores are unreliable
- ❌ Pattern detection is effectively non-functional
- ❌ Users receive the same prediction regardless of input

### Consequences
- Chart pattern analysis is not working
- Trading signals based on patterns are invalid
- User trust in the system is compromised

## Solutions

### Solution 1: Retrain the Model (Recommended)

#### Steps:
1. **Collect balanced dataset** with equal representation of all 20 pattern classes
2. **Ensure minimum samples per class**: At least 100-200 images per pattern
3. **Use data augmentation**: Rotate, flip, adjust brightness/contrast
4. **Implement class weights**: To handle any remaining imbalance
5. **Train for sufficient epochs**: Monitor validation loss for convergence
6. **Use proper evaluation**: 80/20 train/validation split

#### Training Script Example:
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Balance dataset
from sklearn.utils.class_weight import compute_class_weight
train_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Train with callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    class_weight=dict(enumerate(class_weights)),
    callbacks=[early_stop, reduce_lr]
)
```

### Solution 2: Fine-Tune Existing Model

If retraining is not possible, fine-tune the biased model:

```python
# Freeze early layers
for layer in model.layers[:-10]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train on balanced subset
model.fit(balanced_dataset, epochs=50)
```

### Solution 3: Use Alternative Model

Switch back to the previous model or use a different architecture:

```bash
# Option A: Use the original CNN model
PATTERN_MODEL_PATH=chart_pattern_tf_model.h5

# Option B: Use PyTorch EfficientNet
# Train a new model using app/training/train_models.py
```

### Solution 4: Implement Quick Fix

Add post-processing to detect and handle bias:

```python
def detect_model_bias(predictions, threshold=0.8):
    """Detect if model is showing bias towards specific classes."""
    if predictions['pattern'] == 'Pattern 17' and predictions['pattern_confidence'] > threshold:
        return True
    return False

# In prediction endpoint
if detect_model_bias(pattern_result):
    logger.warning("Model bias detected - consider retraining")
    pattern_result['warning'] = "Model may be biased - prediction confidence is low"
```

## Recommended Action Plan

### Immediate (Today)
1. ✅ Switch back to `chart_pattern_tf_model.h5` in `.env`
2. ✅ Document the issue for users
3. ✅ Add bias detection to the API

### Short-term (This Week)
1. 📊 Collect and analyze training data distribution
2. 📈 Create balanced dataset with all pattern classes
3. 🔧 Set up proper training pipeline

### Medium-term (This Month)
1. 🎯 Retrain MobileNet model with balanced data
2. ✅ Evaluate new model on validation set
3. 🚀 Deploy retrained model

## Verification Steps

### Test Retrained Model
After retraining, verify the model with:

```python
def verify_model_unbiased(model, test_images_per_class=10):
    """Verify model predictions are well-distributed."""
    predictions = []
    for _ in range(test_images_per_class):
        for class_id in range(20):
            # Generate or load test image for each class
            img = load_test_image(class_id)
            pred = model.predict(img)
            predictions.append(np.argmax(pred[0]))

    # Check distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print("Prediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} predictions")

    # Verify no class dominates
    max_count = max(counts)
    total = len(predictions)
    if max_count / total > 0.3:  # No class > 30%
        return False, "Model still shows bias"

    return True, "Model appears unbiased"
```

## Conclusion

The MobileNet model is **not usable in its current state** due to severe class imbalance bias. The model must be retrained on a balanced dataset before deployment. Until then, use the original `chart_pattern_tf_model.h5` model.

**Estimated time to fix**: 2-3 days for data collection and retraining
**Priority**: High - Pattern detection is a core feature
