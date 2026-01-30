# AI Road Damage Classification

This project implements various Deep Learning models to classify road damage using the **Road Damage Classification and Assessment** dataset from Kaggle.

## Dataset

The dataset used for this project can be found here:
[**Road Damage Classification and Assessment**](https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment)

It contains images categorized into different classes of road damage (e.g., Potholes, Cracks) and quality levels.

## Models

The project explores several model architectures, enhanced with custom `RoadAttention` and `PotholeRefinement` mechanisms to improve classification accuracy. The models are implemented in PyTorch (see `notebook/models_factory.py`).

### Supported Architectures:

- **ResNet18** (`ModifResNet18`)
- **ConvNeXt Tiny** (`ModifConvNeXt`)
- **EfficientNet B0** (`ModifEffNetB0`)
- **EfficientNet B3** (`ModifEffNetB3`)
- **MobileNetV3 Large** (`ModifMobileNet`)

Each model incorporates:

1.  **Backbone**: Pre-trained feature extractor (e.g., ResNet, EfficientNet).
2.  **RoadAttention**: A custom attention module to focus on relevant spatial features.
3.  **PotholeRefinement**: A refinement block to enhance feature representation for damage detection.

## Results

The following accuracies have been achieved with different experiments (based on notebook logs):

| Model / Notebook      | Accuracy   |
| :-------------------- | :--------- |
| **Best Model**        | **97.36%** |
| High Accuracy Variant | 97.14%     |
| General Experiment    | 95.44%     |
| EfficientNet B0       | 95.20%     |
| MobileNet             | 95.20%     |
| EfficientNet B3       | 94.00%     |

## Project Structure

- `notebook/`: Contains Jupyter notebooks used for training and experimentation. Files are named with their achieved accuracy (e.g., `97.36_notebook.ipynb`).
- `notebook/models_factory.py`: functionality for defining and initializing the custom PyTorch models.
- `notebook/checkpoints/`: Directory for storing model checkpoints during training.
- `notebook/utils/`: Utility scripts.
- `convert.py`: Script for converting models (likely to ONNX or other formats).
- `requirements.txt`: Python dependencies.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```
