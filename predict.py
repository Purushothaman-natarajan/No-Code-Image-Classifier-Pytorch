import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from data.CustomModel import CustomModel

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = [torch.nn.functional.softmax(output, dim=1) for output in outputs]
        
        # Average the probabilities if using multiple models
        avg_probabilities = torch.mean(torch.stack(probabilities), dim=0)
        top_prob, top_class = avg_probabilities.topk(1, dim=1)
        return top_class.item(), top_prob.item()

def main(model_path, img_path, train_dir, base_model_names):
    # Define target image size based on model requirements
    target_size = 224  # Adjust if needed

    # Get class names from train directory
    class_names = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    num_classes = len(class_names)

    # Load the model
    model = CustomModel(base_model_names, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the image
    image = load_image(img_path, target_size)

    # Predict the image
    pred, prob = predict(model, image, class_names)
    predicted_label = class_names[pred]
    print(f"Predicted Label: {predicted_label}, Probability: {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pre-trained model and make a prediction on a new image")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to be predicted')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training dataset for inferring class names')
    parser.add_argument('--base_model_names', type=str, nargs='+', required=True, help='List of base model names')

    args = parser.parse_args()
    main(args.model_path, args.img_path, args.train_dir, args.base_model_names)
