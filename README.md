# ResNet-18 Visual Inspection Classifier

Image classification pipeline using a fine-tuned **ResNet-18** model to classify industrial component images as **OK** or **NOK**.

The model was developed for visual inspection, where the orientation or side of a component needs to be classified before further processing.

## Overview

The pipeline covers:

- Dataset preparation
- Data augmentation
- ResNet-18 transfer learning and fine-tuning
- Stratified K-Fold cross-validation
- Early stopping
- Learning rate scheduling
- GPU acceleration with CUDA when available
- Image classification with confidence scores
- OpenCV result visualization
- Live model testing
- ONNX model export

## Model

The classifier is based on a ResNet-18 pretrained on ImageNet.

The earlier layers of the network are frozen, while `layer4` and a custom classification head are fine-tuned for the OK/NOK classification task.

The original fully connected layer is replaced with:

```text
ResNet-18 Features
        ↓
Dropout (0.2)
        ↓
Linear (512 → 128)
        ↓
ReLU
        ↓
Dropout (0.1)
        ↓
Linear (128 → 2)
        ↓
     OK / NOK
```

## Training Pipeline

Images are loaded from the dataset and assigned labels based on their filenames.

```text
Images
   ↓
Dataset Loading
   ↓
Data Augmentation
   ↓
Stratified K-Fold Split
   ↓
ResNet-18 Fine-Tuning
   ↓
Validation
   ↓
Best Model Checkpoint
```

During training, the model is evaluated on the validation fold after every epoch.

The checkpoint with the lowest validation loss is saved for each fold. Training can stop early when validation loss stops improving.

`ReduceLROnPlateau` is used to reduce the learning rate when validation performance stops improving.

## Data Augmentation

Training images are randomly transformed using:

- Rotation
- Random resized cropping
- Brightness adjustment
- Contrast adjustment
- ImageNet normalization

Validation and inference images are resized and normalized without random augmentation.

## Project Structure

```text
resnet18-visual-inspection-classifier/
├── data/
│   ├── OK_01.jpg
│   ├── OK_02.jpg
│   ├── NOK_01.jpg
│   └── NOK_02.jpg
│
├── Helping Tools/
│   ├── capture.py
│   ├── convert_onnx.py
│   ├── model_testing_live.py
│   └── rename.py
│
├── train.py
├── requirements.txt
└── README.md
```

### Helping Tools

`capture.py`  
Utility for image acquisition and dataset collection.

`rename.py`  
Utility for preparing and renaming dataset images.

`model_testing_live.py`  
Runs the trained classifier on live images for testing and visual inspection.

`convert_onnx.py`  
Exports the trained PyTorch model to ONNX for use with other inference runtimes and deployment environments.

## Dataset

Training images are stored directly inside the `data/` directory.

The class is determined from the beginning of the filename:

```text
OK_part1.jpg        → OK
OK_sample.png       → OK

NOK_part1.jpg       → NOK
NOK_wrong_side.png  → NOK
```

Supported image formats:

```text
.jpg
.jpeg
.png
```

## Training

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the training pipeline:

```bash
python train.py
```

The script automatically uses CUDA when a compatible GPU is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

During training, the following information is reported for each epoch:

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Current learning rate

The best checkpoint from each fold is saved as:

```text
best_model_fold0.pt
best_model_fold1.pt
...
```

## Main Training Settings

| Setting | Value |
|---|---|
| Architecture | ResNet-18 |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | Up to 8 |
| Loss function | CrossEntropyLoss |
| LR scheduler | ReduceLROnPlateau |
| Early stopping patience | 5 epochs |
| Pretrained weights | ImageNet |

The number of folds and maximum epochs can be configured in `train.py`.

## Inference

After training, a saved checkpoint can be loaded and used to classify an image.

The model outputs class probabilities using Softmax:

```text
Image
   ↓
ResNet-18
   ↓
Class Probabilities
   ↓
OK / NOK
   ↓
Confidence Score
```

OpenCV is used to visualize the classification result.

Example output:

```text
OK: Correct Side
```

or:

```text
NOK: WRONG SIDE
```

A green border represents an **OK** prediction and a red border represents **NOK**. The predicted confidence is also displayed on the image.

## ONNX Export

The trained model can be exported to ONNX using:

```text
Helping Tools/convert_onnx.py
```

This allows the model to be used outside the original PyTorch training environment and with other inference or deployment runtimes.

## Results

The project uses Stratified K-Fold cross-validation to evaluate the model while maintaining the class distribution across folds.

Validation accuracy and validation loss are tracked during training, and the checkpoint with the lowest validation loss for each fold is saved.

Final evaluation metrics can be added after training on the final dataset.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- NumPy
- scikit-learn
- Pillow

See `requirements.txt` for the required Python packages.
