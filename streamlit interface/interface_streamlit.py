__author__ = "Purushothaman Natarajan"
__copyright__ = "Copyright 2024, Purushothaman Natarajan"
__credits__ = ["Purushothaman Natarajan"]
__license__ = "MIT"
__version__ = "V1.0"
__maintainer__ = "Purushothaman Natarajan"
__email__ = "purushothamanprt@gmail.com"
__status__ = "pushed"

import streamlit as st
import subprocess

# Utility function to strip quotes from paths
def strip_quotes(path):
    if isinstance(path, str):
        return path.strip('\'\"')

# Utility function to run a command and handle errors
def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Exception occurred: {str(e)}"

# Define wrapper functions for each script
def run_data_loader(path, target_folder, dim, batch_size, num_workers, augment_data):
    # Strip quotes from paths
    path = strip_quotes(path)
    target_folder = strip_quotes(target_folder)
    
    command = [
        "python", "data_loader.py",
        "--path", path,
        "--target_folder", target_folder,
        "--dim", str(dim),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers)
    ]
    if augment_data:
        command.append("--augment_data")
    
    return run_command(command)

def run_train(base_model_names, shape, data_path, log_dir, model_dir, epochs, optimizer, learning_rate, batch_size):
    if not base_model_names:
        return "Error: You must select at least one base model for training."

    if not shape or not data_path or not log_dir or not model_dir:
        return "Error: Shape, data path, log directory, and model directory are required."

    # Strip quotes from paths
    data_path = strip_quotes(data_path)
    log_dir = strip_quotes(log_dir)
    model_dir = strip_quotes(model_dir)
    
    command = [
        "python", "train.py",
        "--base_model_names", ','.join(base_model_names),
        "--shape", str(shape),
        "--data_path", data_path,
        "--log_dir", log_dir,
        "--model_dir", model_dir,
        "--epochs", str(epochs),
        "--optimizer", optimizer,
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size)
    ]
    
    return run_command(command)

def run_test(model_path, model_dir, img_path, log_dir, test_dir, train_dir, class_names):
    # Strip quotes from paths
    model_path = strip_quotes(model_path)
    model_dir = strip_quotes(model_dir)
    img_path = strip_quotes(img_path)
    log_dir = strip_quotes(log_dir)
    test_dir = strip_quotes(test_dir)
    train_dir = strip_quotes(train_dir)
    
    command = [
        "python", "test.py",
        "--log_dir", log_dir
    ]
    if model_path:
        command.extend(["--model_path", model_path])
    if model_dir:
        command.extend(["--model_dir", model_dir])
    if img_path:
        command.extend(["--img_path", img_path])
    if test_dir:
        command.extend(["--test_dir", test_dir])
    if train_dir:
        command.extend(["--train_dir", train_dir])
    if class_names:
        command.extend(["--class_names"] + class_names.split(","))
    
    return run_command(command)

def run_predict(model_path, img_path, train_dir):
    # Strip quotes from paths
    model_path = strip_quotes(model_path)
    img_path = strip_quotes(img_path)
    train_dir = strip_quotes(train_dir)
    
    command = [
        "python", "predict.py",
        "--model_path", model_path,
        "--img_path", img_path,
        "--train_dir", train_dir
    ]
    
    return run_command(command)

# Streamlit app

st.title("No-Code-Image-Classifier")

st.sidebar.title("Navigation")
options = ["Data Loader", "Training", "Testing", "Prediction"]
choice = st.sidebar.radio("Go to", options)

if choice == "Data Loader":
    st.header("Data Loader")
    
    path = st.text_input("Raw Dataset Path (Path to file)")
    target_folder = st.text_input("Target Folder (Path to directory)")
    dim = st.slider("Dimension", 1, 512, 224)
    batch_size = st.slider("Batch Size", 1, 128, 32)
    num_workers = st.slider("Number of Workers", 1, 16, 4)
    augment_data = st.checkbox("Augment Data")

    if st.button("Run Data Loader"):
        result = run_data_loader(path, target_folder, dim, batch_size, num_workers, augment_data)
        st.text(result)

elif choice == "Training":
    st.header("Training")

    base_models = st.multiselect("Base Models", ['vgg16', 'alexnet', 'convnext_tiny', 'densenet121', 'efficientnet_b0',
    'efficientnet_v2_s', 'googlenet', 'inception_v3', 'mnasnet1_0',
    'mobilenet_v2', 'mobilenet_v3_small', 'regnet_y_400mf', 'resnet18',
    'resnext50_32x4d', 'shufflenet_v2_x1_0', 'squeezenet1_0', 'wide_resnet50_2'])
    shape = st.text_input("Shape", "224")
    data_path = st.text_input("Processed Dataset Path (Path to file)")
    log_dir = st.text_input("Log Directory (Path to directory)")
    model_dir = st.text_input("Model Directory (Path to directory)")
    epochs = st.slider("Epochs", 1, 1000, 100)
    optimizer = st.selectbox("Optimizer", ["adam", "sgd"])
    learning_rate = st.number_input("Learning Rate", value=0.0001)
    batch_size = st.slider("Batch Size", 1, 128, 32)

    if st.button("Run Training"):
        result = run_train(base_models, shape, data_path, log_dir, model_dir, epochs, optimizer, learning_rate, batch_size)
        st.text(result)

elif choice == "Testing":
    st.header("Testing")

    model_path = st.text_input("Model Path (Path to directory)")
    model_dir = st.text_input("Model Directory (Path to directory, optional)")
    img_path = st.file_uploader("Image Path (optional, choose image)")
    log_dir = st.text_input("Log Directory (Path to directory)")
    test_dir = st.text_input("Test Directory (optional, Path to directory)")
    train_dir = st.text_input("Train Directory (optional, Path to directory)")
    class_names = st.text_input("Class Names (optional, comma separated)")

    if st.button("Run Testing"):
        result = run_test(model_path, model_dir, img_path, log_dir, test_dir, train_dir, class_names)
        st.text(result)

elif choice == "Prediction":
    st.header("Prediction")

    model_path = st.text_input("Model Path (Path to directory)")
    img_path = st.file_uploader("Image Path (choose image)", type="filepath")
    train_dir = st.text_input("Train Directory (Path to directory)")

    if st.button("Run Prediction"):
        result = run_predict(model_path, img_path, train_dir)
        st.text(result)
