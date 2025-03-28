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

# Configuration de l'appareil pour compatibilité multi-plateforme
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Pour M1 Mac
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# Visualisation des cartes de caractéristiques
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
            st.error("Impossible d'extraire les cartes de caractéristiques.")
            return
        
        feature_maps = feature_maps[0]
        
        plt.figure(figsize=(15, 10))
        plt.suptitle("Visualisation des Cartes de Caractéristiques", fontsize=16)
        
        grid_size = int(np.ceil(np.sqrt(min(num_feature_maps, feature_maps.shape[1]))))
        
        for i in range(min(num_feature_maps, feature_maps.shape[1])):
            feature_map = feature_maps[0, i, :, :].cpu().numpy()
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(feature_map, cmap='viridis')
            plt.title(f'Carte {i+1}')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        st.subheader("Analyse des Cartes de Caractéristiques")
        st.write("Visualisation des représentations intermédiaires apprises par le modèle :")
        st.pyplot(plt)
        
        st.write("### Interprétation")
        st.markdown("""
        - Chaque carte représente un motif ou une texture appris différent
        - Les zones plus lumineuses indiquent des activations plus fortes
        - Les couches précoces capturent des caractéristiques de base (bords, couleurs)
        - Les couches profondes capturent des représentations plus complexes
        """)
    
    except Exception as e:
        st.error(f"Erreur dans la visualisation : {e}")

def add_feature_map_visualization(model, input_tensor):
    with st.expander("Cartes de Caractéristiques du Modèle"):
        visualize_feature_maps(model, input_tensor)

# Définition du modèle
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, 3)

    def forward(self, x):
        return self.model(x)

# Chargement du modèle avec gestion d'erreurs
def load_model(model_path="newer_model.pth"):
    try:
        model = YourModel()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Implémentation de GradCAM
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
            raise ValueError("Gradients ou activations non capturés.")
        
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
    with st.expander("Visualisation Grad-CAM (Zones d'attention)"):
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
            plt.title("Image Originale")
            plt.imshow(image_np)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Carte Grad-CAM")
            plt.imshow(image_np)
            plt.imshow(cam_resized, cmap='jet', alpha=0.5)
            plt.axis('off')
            
            st.pyplot(plt)
            
            st.markdown("""
            ### Interprétation de Grad-CAM
            - Les zones rouges/jaunes montrent les régions influençant la décision
            - Plus la couleur est intense, plus la région est importante
            - Cet outil explique quelles parties ont influencé le diagnostic
            """)
            
        except Exception as e:
            st.error(f"Erreur lors de la génération Grad-CAM : {e}")
        
        finally:
            gradcam.remove_hooks()

# Prétraitement
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Informations sur les maladies
disease_info = {
    "normal": "L'œil semble en bonne santé, sans signes visibles d'infection ou d'anomalie.",
    "epiphore": "L'épiphore est un larmoiement excessif causé par une obstruction des voies lacrymales ou une hyperproduction de larmes.",
    "keratite": "La kératite est une inflammation de la cornée qui peut résulter d'une infection ou d'un traumatisme. Elle nécessite une intervention médicale rapide."
}

# Application Streamlit principale
def main():
    st.title("Vision IA : Détection Préventive des Maladies Oculaires")
    st.write("Soumettez une image de l'œil pour détecter d'éventuelles pathologies.")
    
    model = load_model()
    if model is None:
        return
    
    uploaded_file = st.file_uploader("Téléchargez une image...", type=["png", "jpg", "jpeg", "tiff"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Image Téléchargée", use_container_width=True)
        
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor.to(device))
            probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            predicted_class = np.argmax(probabilities)
        
        class_names = ["normal", "epiphore", "keratite"]
        predicted_class_name = class_names[predicted_class]
        
        st.subheader(f"Prédiction : **{predicted_class_name.upper()}**")
        st.write(f"Confiance : **{probabilities[predicted_class]*100:.2f}%**")
        
        st.write("### Probabilités par classe :")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name.capitalize()} : {probabilities[i]*100:.2f}%")
            st.progress(float(probabilities[i]))
        
        st.write(f"### Information sur {predicted_class_name} :")
        st.write(disease_info[predicted_class_name])
        
        show_gradcam(model, input_tensor, predicted_class)
        add_feature_map_visualization(model, input_tensor)
        
        st.info("Cette application est un outil d'aide et ne remplace pas un diagnostic médical. Consultez un professionnel de santé en cas de doute.")

if __name__ == "__main__":
    main()