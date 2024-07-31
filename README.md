[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# No-Code-Image-Classifier-Pytorch - Develop your own Image Classifier

This project provides a no-code interface for developing image classification models using the Torch framework. Use the `setup_and_run` script to set up and open the Gradio-based interface, which simplifies the process of developing and testing image classification models. If you prefer streamlit interface, navigate to `Image Classifer/streamlit interface`, use the `setup_and_run - streamlit` script to set up and open Streamlit-based interface.

## Prerequisites

- Python 3.6 or higher

## Getting Started (Demo Video)

Image Classifer/demo video - *to be uploaded*

<p align="center">
  <img src="demo/demo snap.png" alt="Preview">
</p>

## Running the Scripts

This guide will help you run the `data_loader.py`, `train.py`, `test.py`, and `predict.py` scripts directly from the command line or a Python script.

### Prerequisites

1. **Python Installation**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).

2. **Required Packages**: Install the required packages using `requirements.txt`.
   ```sh
   pip install -r requirements.txt
   ```

### Script Descriptions and Usage

#### 1. `data_loader.py`

This script is used to load, process, split dataset - train, val, tests. and augment data.

**Command Line Usage:**

```sh
python data_loader.py --path <path_to_data> --target_folder <path_to_target_folder> --dim <dimension> --batch_size <batch_size> --num_workers <num_workers> [--augment_data]
```

**Arguments:**

- `--path`: Path to the data.
- `--target_folder`: Path to the target folder where processed data will be saved.
- `--dim`: Dimension for resizing the images.
- `--batch_size`: Batch size for data loading.
- `--num_workers`: Number of workers for data loading.
- `--augment_data` (optional): Flag to enable data augmentation.

**Example:**

```sh
python data_loader.py --path "./data" --target_folder "./processed_data" --dim 224 --batch_size 32 --num_workers 4 --augment_data
```

**Dataset Structure:**

```sh
├── Dataset (Raw)
   ├── class_name_1
   │   └── *.jpg
   ├── class_name_2
   │   └── *.jpg
   ├── class_name_3
   │   └── *.jpg
   └── class_name_4
       └── *.jpg
```

#### 2. `train.py`

This script is used for training and storing the models leveraging transfer learning.

**Command Line Usage:**

```sh
python train.py --base_model_names <model_names> --shape <shape> --data_path <data_path> --log_dir <log_dir> --model_dir <model_dir> --epochs <epochs> --optimizer <optimizer> --learning_rate <learning_rate> --batch_size <batch_size>
```

**Arguments:**

- `--base_model_names`: Comma-separated list of base model names (e.g., 'vgg16,alexnet').
- `--shape`: Image shape (size).
- `--data_path`: Path to the data.
- `--log_dir`: Path to the log directory.
- `--model_dir`: Path to the model directory.
- `--epochs`: Number of training epochs.
- `--optimizer`: Optimizer type ('adam' or 'sgd').
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Batch size for training.

**Example:**

```sh
python train.py --base_model_names "vgg16,alexnet" --shape 224 --data_path "./data" --log_dir "./logs" --model_dir "./models" --epochs 100 --optimizer "adam" --learning_rate 0.001 --batch_size 32
```

#### 3. `test.py`

This script is used for testing and storing the test logs of the trained models.

**Command Line Usage:**

```sh
python test.py --data_path <data_path> --base_model_name <base_model_name> --model_path <model_path> --models_folder_path <models_folder_path> --log_dir <log_dir>
```

**Arguments:**

- `--data_path`: Path to the test data.
- `--base_model_name`: Name of the base model.
- `--model_path` (optional): Path to the specific model file.
- `--models_folder_path` (optional): Path to the folder containing models.
- `--log_dir`: Path to the log directory.

**Example:**

```sh
python test.py --data_path "./test_data" --base_model_name "vgg16" --model_path "./models/vgg16_model.pth" --log_dir "./logs"
```

#### 4. `predict.py`

This script is used for making predictions on new images.

**Command Line Usage:**

```sh
python predict.py --model_path <model_path> --img_path <img_path> --train_dir <train_dir> --base_model_name <base_model_name>
```

**Arguments:**

- `--model_path`: Path to the model file.
- `--img_path`: Path to the image file.
- `--train_dir`: Path to the training dataset.
- `--base_model_name`: Name of the base model.

**Example:**

```sh
python predict.py --model_path "./models/vgg16_model.pth" --img_path "./images/test_image.jpg" --train_dir "./data/train" --base_model_name "vgg16"
```

### Running Scripts in a Python Script

You can also run these scripts programmatically using Python's `subprocess` module specially for Jupyter Notebook users. Here is an example of how to do this for each script:

```python
import subprocess

# Run data_loader.py
subprocess.run([
    "python", "data_loader.py",
    "--path", "./data",
    "--target_folder", "./processed_data",
    "--dim", "224",
    "--batch_size", "32",
    "--num_workers", "4",
    "--augment_data"
])

# Run train.py
subprocess.run([
    "python", "train.py",
    "--base_model_names", "vgg16,alexnet",
    "--shape", "224",
    "--data_path", "./data",
    "--log_dir", "./logs",
    "--model_dir", "./models",
    "--epochs", "100",
    "--optimizer", "adam",
    "--learning_rate", "0.001",
    "--batch_size", "32"
])

# Run test.py
subprocess.run([
    "python", "test.py",
    "--data_path", "./test_data",
    "--base_model_name", "vgg16",
    "--model_path", "./models/vgg16_model.pth",
    "--log_dir", "./logs"
])

# Run predict.py
subprocess.run([
    "python", "predict.py",
    "--model_path", "./models/vgg16_model.pth",
    "--img_path", "./images/test_image.jpg",
    "--train_dir", "./data/train",
    "--base_model_name", "vgg16"
])
```

----
[contributors-shield]: https://img.shields.io/github/contributors/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch.svg?style=flat-square
[contributors-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch.svg?style=flat-square
[forks-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch/network/members
[stars-shield]: https://img.shields.io/github/stars/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch.svg?style=flat-square
[stars-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch/stargazers
[issues-shield]: https://img.shields.io/github/issues/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch.svg?style=flat-square
[issues-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch/issues
[license-shield]: https://img.shields.io/github/license/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch.svg?style=flat-square
[license-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier-Pytorch/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/purushothamann/
