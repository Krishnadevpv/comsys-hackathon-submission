#!/usr/bin/env python3
"""
COMSYS Hackathon-5 Solution
Face Recognition and Gender Classification under Adverse Visual Conditions
Author: KRISHNADEV P V
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
from tqdm import tqdm
import json
import random
from collections import defaultdict
from src.verification_dataset import FaceVerificationDataset

class GenderDataset(Dataset):
    """Dataset class for Task A - Gender Classification"""
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load data from taska folder
        task_path = os.path.join(data_path, 'taska', split)
        
        for gender_idx, gender in enumerate(['female', 'male']):
            gender_path = os.path.join(task_path, gender)
            if os.path.exists(gender_path):
                for img_file in os.listdir(gender_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(gender_path, img_file)
                        self.samples.append(img_path)
                        self.labels.append(gender_idx)  # 0: female, 1: male
        
        print(f"Gender Dataset {split}: {len(self.samples)} images")
        print(f"Female: {self.labels.count(0)}, Male: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label

class FaceRecognitionDataset(Dataset):
    """Dataset class for Task B - Face Recognition"""
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []
        self.person_to_idx = {}
        
        # Load data from taskb folder
        task_path = os.path.join(data_path, 'taskb', split)
        person_idx = 0
        
        for person_folder in os.listdir(task_path):
            person_path = os.path.join(task_path, person_folder)
            if os.path.isdir(person_path):
                self.person_to_idx[person_folder] = person_idx
                
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_path, img_file)
                        self.samples.append(img_path)
                        self.labels.append(person_idx)
                
                person_idx += 1
        
        self.num_classes = len(self.person_to_idx)
        print(f"Face Recognition Dataset {split}: {len(self.samples)} images")
        print(f"Number of persons: {self.num_classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label

class GenderClassifier(nn.Module):
    """Gender Classification Model - Task A"""
    
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.backbone(x)

class FaceRecognitionModel(nn.Module):
    """Face Recognition Model - Task B"""
    
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
import torch.nn as nn
from torchvision.models import resnet50

class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FaceEmbeddingModel, self).__init__()
        backbone = resnet50(pretrained=True)
        modules = list(backbone.children())[:-1]  # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*modules)
        self.embedding = nn.Linear(backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2-normalize the embeddings
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        # Contrastive loss formula
        loss = (label.float() * distance.pow(2)) + \
               ((1 - label.float()) * torch.clamp(self.margin - distance, min=0).pow(2))
        return loss.mean()

# ✅ Task B - Face Verification Test Script
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import json


def test_face_verification(data_path, model_path="models/face_verification.pth", threshold=0.7):
    """
    - Match test images (including distorted) against person folders
    - Output: 1 if matches same person, 0 if matches different person
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_path = os.path.join(data_path, "taskb", "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embedding model
    model = FaceEmbeddingModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Build reference embeddings for each person folder
    print("🔍 Building reference embeddings for each person...")
    person_embeddings = {}
    
    for person_folder in os.listdir(val_path):
        person_path = os.path.join(val_path, person_folder)
        if not os.path.isdir(person_path):
            continue
            
        # Get reference image (non-distorted)
        reference_images = []
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'distort' not in img_file.lower():
                img_path = os.path.join(person_path, img_file)
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embedding = model(image_tensor).cpu()
                reference_images.append(embedding)
        
        if reference_images:
            # Average embeddings if multiple reference images
            person_embeddings[person_folder] = torch.stack(reference_images).mean(dim=0)

    # Test matching protocol
    print("🔎 Testing face matching protocol...")
    all_predictions = []
    all_true_labels = []
    test_results = []

    for person_folder in os.listdir(val_path):
        person_path = os.path.join(val_path, person_folder)
        distorted_path = os.path.join(person_path, "distortion")
        
        if not os.path.isdir(distorted_path):
            continue
            
        # Test each distorted image
        for dist_img in os.listdir(distorted_path):
            if not dist_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            test_img_path = os.path.join(distorted_path, dist_img)
            test_image = Image.open(test_img_path).convert("RGB")
            test_tensor = transform(test_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                test_embedding = model(test_tensor).cpu()
            
            # Compare against all person folders
            max_similarity = -1
            best_match = None
            
            for ref_person, ref_embedding in person_embeddings.items():
                similarity = F.cosine_similarity(test_embedding, ref_embedding).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = ref_person
            
            # Determine if it's a match (1) or non-match (0)
            is_match = 1 if (best_match == person_folder and max_similarity > threshold) else 0
            true_label = 1 if best_match == person_folder else 0
            
            all_predictions.append(is_match)
            all_true_labels.append(true_label)
            
            test_results.append({
                "test_image": test_img_path,
                "true_person": person_folder,
                "predicted_person": best_match,
                "similarity_score": round(max_similarity, 4),
                "prediction": is_match,
                "true_label": true_label
            })

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, zero_division=0)

    print(f"\n📈 Face Matching Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save detailed results
    os.makedirs("results", exist_ok=True)
    results = {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "test_details": test_results
    }
    
    with open("results/face_matching_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results["metrics"]
class COMSYSHackathonSolution:
    """Main solution class"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Models
        self.gender_model = None
        self.face_model = None
        
    def load_data(self, data_path):
        """Load datasets for both tasks"""
        print("Loading datasets...")
        
        # Task A - Gender Classification
        self.gender_train_dataset = GenderDataset(data_path, 'train', self.train_transform)
        self.gender_val_dataset = GenderDataset(data_path, 'val', self.val_transform)
        
        # Task B - Face Recognition
        self.face_train_dataset = FaceRecognitionDataset(data_path, 'train', self.train_transform)
        self.face_val_dataset = FaceRecognitionDataset(data_path, 'val', self.val_transform)
        
        # Data loaders
        self.gender_train_loader = DataLoader(self.gender_train_dataset, batch_size=32, shuffle=True, num_workers=4)
        self.gender_val_loader = DataLoader(self.gender_val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        self.face_train_loader = DataLoader(self.face_train_dataset, batch_size=32, shuffle=True, num_workers=4)
        self.face_val_loader = DataLoader(self.face_val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        print("Datasets loaded successfully!")
        
    def train_gender_classifier(self, epochs=20):
        """Train gender classification model"""
        print("\n🚀 Training Gender Classifier (Task A)...")
        
        self.gender_model = GenderClassifier().to(self.device)
        class_weights = torch.tensor([1.0, 0.5]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.gender_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.gender_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.gender_train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.gender_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct/total:.2f}%'})
            
            train_acc = 100 * correct / total
            avg_loss = running_loss / len(self.gender_train_loader)
            train_losses.append(avg_loss)
            
            # Validation phase
            val_acc = self.evaluate_gender_model()
            val_accuracies.append(val_acc)
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.gender_model.state_dict(), 'models/gender_classifier.pth')
                print(f'New best model saved with validation accuracy: {best_acc:.2f}%')
        
        # Plot training history
        self.plot_training_history(train_losses, val_accuracies, 'Gender Classification')
        
    def train_face_recognition(self, epochs=25):
        """Train face recognition model"""
        print("\n🚀 Training Face Recognition Model (Task B)...")
        
        num_classes = self.face_train_dataset.num_classes
        self.face_model = FaceRecognitionModel(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.face_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        
        best_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.face_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.face_train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.face_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct/total:.2f}%'})
            
            train_acc = 100 * correct / total
            avg_loss = running_loss / len(self.face_train_loader)
            train_losses.append(avg_loss)
            
            # Validation phase
            val_acc = self.evaluate_face_model()
            val_accuracies.append(val_acc)
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.face_model.state_dict(), 'models/face_recognition.pth')
                print(f'New best model saved with validation accuracy: {best_acc:.2f}%')
        
        # Plot training history
        self.plot_training_history(train_losses, val_accuracies, 'Face Recognition')
    

    

    
    def evaluate_gender_model(self):
        """Evaluate gender classification model"""
        if self.gender_model is None:
            return 0.0
            
        self.gender_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.gender_val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.gender_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def evaluate_face_model(self):
        """Evaluate face recognition model"""
        if self.face_model is None:
            return 0.0
            
        self.face_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.face_val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.face_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def test_models(self, data_path):
        """Test both models and return detailed metrics"""
        print("\n📊 Testing Models...")
        
        # Load models
        if os.path.exists('models/gender_classifier.pth'):
            self.gender_model = GenderClassifier().to(self.device)
            self.gender_model.load_state_dict(torch.load('models/gender_classifier.pth', map_location=self.device))
            print("Gender classifier loaded successfully")
        else:
            print("Gender classifier not found! Please train first.")
            return None, None
        
        # Load face recognition model
        face_val_dataset = FaceRecognitionDataset(data_path, 'val', self.val_transform)
        if os.path.exists('models/face_recognition.pth'):
            self.face_model = FaceRecognitionModel(face_val_dataset.num_classes).to(self.device)
            self.face_model.load_state_dict(torch.load('models/face_recognition.pth', map_location=self.device))
            print("Face recognition model loaded successfully")
        else:
            print("Face recognition model not found! Please train first.")
            return None, None
        
        # Test gender classification
        gender_metrics = self.test_gender_classification(data_path)
        
        # Test face recognition
        face_metrics = self.test_face_recognition(data_path)
        
        return gender_metrics, face_metrics
    
    def test_gender_classification(self, data_path):
        """Test gender classification with detailed metrics"""
        print("\nTesting Gender Classification...")

         # ✅ Load trained model
        if self.gender_model is None:
            if os.path.exists('models/gender_classifier.pth'):
                self.gender_model = GenderClassifier().to(self.device)
                self.gender_model.load_state_dict(torch.load('models/gender_classifier.pth', map_location=self.device))
                print("Gender classifier loaded successfully")
            else:
                print("❌ gender_classifier.pth not found!")
                return None
        
        test_dataset = GenderDataset(data_path, 'val', self.val_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.gender_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing Gender Classification"):
                images = images.to(self.device)
                outputs = self.gender_model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm, ['Female', 'Male'], 'Gender Classification')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def test_face_recognition(self, data_path):
        """Test face recognition with detailed metrics"""
        print("\nTesting Face Recognition...")
        
        test_dataset = FaceRecognitionDataset(data_path, 'val', self.val_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.face_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing Face Recognition"):
                images = images.to(self.device)
                outputs = self.face_model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_training_history(self, losses, accuracies, title):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title(f'{title} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.title(f'{title} - Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/{title.lower().replace(" ", "_")}_training_history.png')
        plt.show()
    
    def plot_confusion_matrix(self, cm, classes, title):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'{title} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'results/{title.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.show()


def train_face_verification(data_path, epochs=10, batch_size=32, margin=1.0):
        print("🚀 Training Face Verification Model...")
        from src.verification_dataset import FaceVerificationDataset
        # Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Dataset & Dataloader
        train_dataset = FaceVerificationDataset(
            root_dir=os.path.join(data_path, 'taskb'),
            split='train',
            transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Model, loss, optimizer
        model = FaceEmbeddingModel().to('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = ContrastiveLoss(margin=margin)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for img1, img2, label in pbar:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1 = model(img1)
                out2 = model(img2)
                loss = loss_fn(out1, out2, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

            # Save model after last epoch
            if epoch == epochs - 1:
                torch.save(model.state_dict(), "models/face_verification.pth")
                print("✅ Saved: models/face_verification.pth")

# finetuning
def fine_tune_gender_classifier(self, additional_epochs=3):
    """Fine-tune with reduced learning rate and focused training"""
    print("🔧 Fine-tuning Gender Classifier...")
    
    # Load best model
    if os.path.exists('models/gender_classifier.pth'):
        self.gender_model = GenderClassifier().to(self.device)
        self.gender_model.load_state_dict(torch.load('models/gender_classifier.pth', map_location=self.device))
    else:
        print("❌ No trained model found! Train first.")
        return
    
    # Use much lower learning rate for fine-tuning
    optimizer = optim.Adam(self.gender_model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.9384  # Your current best
    
    for epoch in range(additional_epochs):
        # Training with early stopping if no improvement
        self.gender_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.gender_train_loader, desc=f'Fine-tune Epoch {epoch+1}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.gender_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct/total:.2f}%'})
        
        # Validation
        val_acc = self.evaluate_gender_model()
        print(f'Fine-tune Epoch {epoch+1}: Val Acc: {val_acc:.4f}%')
        
        # Save only if improvement
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(self.gender_model.state_dict(), 'models/gender_classifier.pth')
            print(f'✅ Improved to {best_acc:.4f}%')
        else:
            print('⚠️ No improvement, consider stopping')
            if epoch > 0:  # Early stopping after 1 epoch without improvement
                break
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='COMSYS Hackathon-5 Solution')
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--data_path', required=True, help='Path to dataset')
    parser.add_argument('--gender_epochs', type=int, default=20, help='Epochs for gender classification')
    parser.add_argument('--face_epochs', type=int, default=25, help='(Unused) Epochs for face recognition')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune gender classifier')
    parser.add_argument(
        '--task',
        choices=['gender', 'verification', 'both'],
        default='all',
        help='Which task(s) to run: gender, verification, or both'
    )

    
    args = parser.parse_args()
    
    # Create solution instance
    solution = COMSYSHackathonSolution()
    
    if args.mode == 'train':
        print("🎯 COMSYS Hackathon-5 Training Mode")
        print("=" * 50)

        
        if args.task in ['gender', 'both']:
            solution.load_data(args.data_path)
            if args.finetune:
               solution.fine_tune_gender_classifier(additional_epochs=args.gender_epochs)
            else:
               solution.train_gender_classifier(args.gender_epochs)
        
        
        if args.task in ['verification', 'both']:
            # Train Task B verification model
            train_face_verification(args.data_path, epochs=15)

        
        print("\n✅ Training completed successfully!")
        
    elif args.mode == 'test':
        print("🧪 COMSYS Hackathon-5 Testing Mode")
        print("=" * 50)

        gender_metrics = None
        face_metrics = None

        if args.task == 'gender':
            gender_metrics = solution.test_gender_classification(args.data_path)

        elif args.task == 'verification':
            face_metrics = test_face_verification(args.data_path)

        elif args.task == 'both':
            gender_metrics = solution.test_gender_classification(args.data_path)
            face_metrics = test_face_verification(args.data_path)

        # Print metrics
        if gender_metrics:
            print("\n" + "=" * 60)
            print("🎯 Task A – Gender Classification Results")
            print("=" * 60)
            print(f"✅ Accuracy   : {gender_metrics['accuracy']:.4f}")
            print(f"✅ Precision  : {gender_metrics['precision']:.4f}")
            print(f"✅ Recall     : {gender_metrics['recall']:.4f}")
            print(f"✅ F1-Score   : {gender_metrics['f1_score']:.4f}")
            print("=" * 60)

        if face_metrics:
            print("\n" + "=" * 60)
            print("🎯 Task B – Face Verification Results")
            print("=" * 60)
            print(f"✅ Accuracy   : {face_metrics['accuracy']:.4f}")
            print(f"✅ Precision  : {face_metrics['precision']:.4f}")
            print(f"✅ Recall     : {face_metrics['recall']:.4f}")
            print(f"✅ F1-Score   : {face_metrics['f1_score']:.4f}")
            print("=" * 60)

        if gender_metrics and face_metrics:
            weighted_score = 0.3 * gender_metrics['accuracy'] + 0.7 * face_metrics['accuracy']
            print("\n" + "=" * 60)
            print(f"🏆 Combined Weighted Score: {weighted_score:.4f}")
            print("=" * 60)


        # Save results
        results = {
            "gender_classification": gender_metrics,
            "face_verification": face_metrics,
        }

        if gender_metrics and face_metrics:
            results["weighted_score"] = weighted_score

        os.makedirs("results", exist_ok=True)
        with open("results/test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("✅ Results saved to results/test_results.json")

    elif args.task == 'verification':
        print("🧪 Testing Task B - Face Verification")
        test_face_verification(args.data_path)


if __name__ == "__main__":
    main()
