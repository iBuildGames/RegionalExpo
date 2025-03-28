import torch
import torch.nn as nn
from torchvision import models
import streamlit as st
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import os

# Device configuration for cross-platform compatibility
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # For M1 Mac
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# Visualize feature maps
def visualize_feature_maps(model, input_tensor, num_feature_maps=16):
    try:
        feature_extractor = nn.Sequential(*list(model.model.children())[:-2])
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output)
        
        hook = feature_extractor[-1].register_forward_hook(hook_fn)
        
        with torch.no_grad():
            feature_extractor(input_tensor.to(device))
        
        hook.remove()
        
        if not feature_maps:
            st.error("Failed to extract feature maps.")
            return
        
        feature_maps = feature_maps[0]
        
        plt.figure(figsize=(15, 10))
        plt.suptitle("Feature Maps Visualization", fontsize=16)
        
        grid_size = int(np.ceil(np.sqrt(min(num_feature_maps, feature_maps.shape[1]))))
        
        for i in range(min(num_feature_maps, feature_maps.shape[1])):
            feature_map = feature_maps[0, i, :, :].cpu().numpy()
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(feature_map, cmap='viridis')
            plt.title(f'Feature Map {i+1}')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        st.subheader("Feature Maps Analysis")
        st.write("Visualization of intermediate representations learned by the model:")
        st.pyplot(plt)
        
        st.write("### Interpretation")
        st.markdown("""
        - Each feature map represents a different learned pattern or texture
        - Brighter areas indicate stronger activations
        - Early layers capture basic features like edges and colors
        - Deeper layers capture more complex, abstract representations
        """)
    
    except Exception as e:
        st.error(f"Error in feature maps visualization: {e}")

def add_feature_map_visualization(model, input_tensor):
    with st.expander("Model Feature Maps"):
        visualize_feature_maps(model, input_tensor)

# Model definition
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, 3)

    def forward(self, x):
        return self.model(x)

# Load model with error handling
def load_model(model_path="newer_model.pth"):
    try:
        model = YourModel()
        # Use map_location to handle different devices
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        handle1 = target_layer.register_forward_hook(self.save_activation)
        handle2 = target_layer.register_backward_hook(self.save_gradient)
        self.hook_handles.extend([handle1, handle2])
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, class_idx):
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured.")
        
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)
        return cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def show_gradcam(model, input_tensor, predicted_class):
    with st.expander("Grad-CAM Visualization (Model Attention Areas)"):
        target_layer = model.model.layer4[-1]
        gradcam = GradCAM(model, target_layer)
        
        try:
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            model.zero_grad()
            class_loss = output[0, predicted_class]
            class_loss.backward()
            
            cam = gradcam.generate_cam(predicted_class)
            
            image_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
            
            cam_resized = F.interpolate(
                torch.from_numpy(cam).unsqueeze(0).unsqueeze(0), 
                size=(image_np.shape[0], image_np.shape[1]), 
                mode='bilinear'
            ).squeeze().numpy()
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(image_np)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Grad-CAM Activation Map")
            plt.imshow(image_np)
            plt.imshow(cam_resized, cmap='jet', alpha=0.5)
            plt.axis('off')
            
            st.pyplot(plt)
            
            st.markdown("""
            ### Grad-CAM Interpretation
            - Red/yellow areas indicate regions most influential to the model's decision
            - More intense colors show higher importance for classification
            - This tool helps understand which image parts influenced the diagnosis
            """)
            
        except Exception as e:
            st.error(f"Error generating Grad-CAM: {e}")
        
        finally:
            gradcam.remove_hooks()

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Disease info
disease_info = {
    "normal": "The eye appears healthy with no visible signs of infection or abnormality.",
    "epiphore": "Epiphora is excessive tearing caused by blocked tear ducts or tear overproduction.",
    "keratite": "Keratitis is corneal inflammation from infection or trauma, requiring prompt medical attention."
}

# Main Streamlit app
def main():
    st.title("AI Vision: Preventive Eye Disease Detection")
    st.write("Upload an eye image to detect potential conditions.")
    
    model = load_model()
    if model is None:
        return
    
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg", "tiff"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor.to(device))
            probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            predicted_class = np.argmax(probabilities)
        
        class_names = ["normal", "epiphore", "keratite"]
        predicted_class_name = class_names[predicted_class]
        
        st.subheader(f"Prediction: **{predicted_class_name.upper()}**")
        st.write(f"Confidence: **{probabilities[predicted_class]*100:.2f}%**")
        
        st.write("### Class Probabilities:")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name.capitalize()}: {probabilities[i]*100:.2f}%")
            st.progress(float(probabilities[i]))
        
        st.write(f"### Information about {predicted_class_name}:")
        st.write(disease_info[predicted_class_name])
        
        show_gradcam(model, input_tensor, predicted_class)
        add_feature_map_visualization(model, input_tensor)
        
        st.info("This app is a support tool and not a substitute for medical diagnosis. Consult a healthcare professional if in doubt.")

if __name__ == "__main__":
    main()