import gradio as gr
import subprocess
import os
import difflib

# Utility function to strip quotes from paths
def strip_quotes(path):
    if isinstance(path, str):
        return path.strip('\'\"')
    return path

# Utility function to run a command and handle errors
def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Exception occurred: {str(e)}"

# List of allowed model names
allowed_models = [
    'vgg16', 'alexnet', 'convnext_tiny', 'densenet121', 'efficientnet_b0',
    'efficientnet_v2_s', 'googlenet', 'inception_v3', 'mnasnet1_0',
    'mobilenet_v2', 'mobilenet_v3_small', 'regnet_y_400mf', 'resnet18',
    'resnext50_32x4d', 'shufflenet_v2_x1_0', 'squeezenet1_0', 'wide_resnet50_2'
]

# Define wrapper functions for each script
def run_data_loader(path, target_folder, dim, batch_size, num_workers, augment_data):
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

def run_train(base_models, shape, data_path, log_dir, model_dir, epochs, optimizer, learning_rate, batch_size):
    if not base_models:
        return "Error: You must select at least one base model for training."

    if not shape or not data_path or not log_dir or not model_dir:
        return "Error: Shape, data path, log directory, and model directory are required."

    try:
        shape_value = int(shape)
    except ValueError:
        return "Error: Shape must be an integer"

    # Strip quotes from paths
    data_path = strip_quotes(data_path)
    log_dir = strip_quotes(log_dir)
    model_dir = strip_quotes(model_dir)
    
    command = [
        "python", "train.py",
        "--base_model_names", ','.join(base_models),
        "--shape", str(shape_value),
        "--data_path", data_path,
        "--log_dir", log_dir,
        "--model_dir", model_dir,
        "--epochs", str(epochs),
        "--optimizer", optimizer,
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size)
    ]
    
    if optimizer == "sgd":
        # Include alpha and gamma only for SGD
        command.extend(["--alpha", "0.9", "--step_gamma", "0.9"])
    
    return run_command(command)

def run_test(data_path, base_model_name, model_path, models_folder_path, log_dir):
    # Strip quotes from paths
    data_path = strip_quotes(data_path)
    model_path = strip_quotes(model_path)
    models_folder_path = strip_quotes(models_folder_path)
    log_dir = strip_quotes(log_dir)
    
    command = [
        "python", "test.py",
        "--data_path", data_path,
        "--base_model_name", base_model_name,
        "--log_dir", log_dir
    ]
    
    if model_path:
        command.extend(["--model_path", model_path])
    if models_folder_path:
        command.extend(["--models_folder_path", models_folder_path])
    
    return run_command(command)

def run_predict(model_path, img_path, train_dir, log_dir):
    model_path = strip_quotes(model_path)
    train_dir = strip_quotes(train_dir)
    
    # Extract base model name from the file name
    model_filename = os.path.basename(model_path)
    base_model_name = model_filename.split('_')[0]
    
    if base_model_name not in allowed_models:
        closest_match = difflib.get_close_matches(base_model_name, allowed_models, n=1)
        if closest_match:
            base_model_name = closest_match[0]
        else:
            return f"Error: No close match found for '{base_model_name}'. Must be one of {allowed_models}."

    command = [
        "python", "predict.py",
        "--model_path", model_path,
        "--img_path", img_path,
        "--train_dir", train_dir,
        "--base_model_name", base_model_name
    ]
    
    return run_command(command)

# Create Gradio interfaces
data_loader_interface = gr.Interface(
    fn=run_data_loader,
    inputs=[
        gr.Textbox(label="Path (Path to file)"),
        gr.Textbox(label="Target Folder (Path to directory)"),
        gr.Slider(minimum=1, maximum=512, value=224, label="Dimension"),
        gr.Slider(minimum=1, maximum=128, value=32, label="Batch Size"),
        gr.Slider(minimum=1, maximum=16, value=4, label="Number of Workers"),
        gr.Checkbox(label="Augment Data")
    ],
    outputs="text",
    title="Data Loader"
)

train_interface = gr.Interface(
    fn=run_train,
    inputs=[
        gr.CheckboxGroup(
            allowed_models, label="Base Models"
        ),
        gr.Slider(minimum=32, maximum=512, value=224, step=1, label="Shape (Size)"),
        gr.Textbox(label="Data Path (Path to file)"),
        gr.Textbox(label="Log Directory (Path to directory)"),
        gr.Textbox(label="Model Directory (Path to directory)"),
        gr.Slider(minimum=1, maximum=1000, value=100, label="Epochs"),
        gr.Dropdown(["adam", "sgd"], value="adam", label="Optimizer"),
        gr.Number(value=0.001, label="Learning Rate"),
        gr.Slider(minimum=1, maximum=128, value=32, label="Batch Size")
    ],
    outputs="text",
    title="Training"
)

test_interface = gr.Interface(
    fn=run_test,
    inputs=[
        gr.Textbox(label="Data Path (Path to file or directory)"),
        gr.Textbox(label="Base Model Name"),
        gr.Textbox(label="Model Path (Path to model file)"),
        gr.Textbox(label="Models Folder Path (Path to directory with models)"),
        gr.Textbox(label="Log Directory (Path to directory)")
    ],
    outputs="text",
    title="Testing"
)

predict_interface = gr.Interface(
    fn=run_predict,
    inputs=[
        gr.File(label="Model Path (choose model)", type="filepath"),
        gr.Image(type="filepath", label="Image Path (choose image)"),
        gr.Textbox(label="Train Directory (Path to training dataset)")
    ],
    outputs="text",
    title="Prediction"
)

gr.TabbedInterface(
    [data_loader_interface, train_interface, test_interface, predict_interface],
    ["Data Loader", "Training", "Testing", "Prediction"]
).launch(debug=True)
