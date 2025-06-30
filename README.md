# COMSYS Hackathon-5: Face Recognition and Gender Classification under Adverse Visual Conditions

**Author:** KRISHNADEV P V  
**Competition:** COMSYS Hackathon-5  
**Tasks:** Binary Gender Classification + Face Verification under Distorted Conditions

## 🎯 Project Overview

This project tackles two challenging computer vision tasks:
- **Task A:** Gender Classification (Binary Classification)
- **Task B:** Face Recognition/Verification under adverse visual conditions

The solution employs deep learning models with sophisticated data augmentation and contrastive learning approaches to handle distorted and challenging visual conditions.

## 🏆 Results

### Task A - Gender Classification
- **Accuracy:** 94.08%
- **Precision:** 94.11%
- **Recall:** 94.08%
- **F1-Score:** 94.09%

### Task B - Face Verification
- **Accuracy:** 99.09% 🔥
- **Precision:** 100.00% 🎯
- **Recall:** 98.38%
- **F1-Score:** 99.18%

### 🏆 Combined Weighted Score: 97.58%

## 🔧 Technical Architecture

### Models Used
1. **Gender Classifier:** ResNet-18 backbone with custom classification head
2. **Face Verification:** ResNet-50 with embedding layer and contrastive loss
3. **Data Augmentation:** Random flips, rotation, color jittering for robustness

### Key Features
- ✅ Handles adverse visual conditions (distortion, lighting variations)
- ✅ Contrastive learning for face verification
- ✅ Class-weighted training for imbalanced datasets
- ✅ Comprehensive evaluation metrics and visualization
- ✅ Modular, extensible codebase

## 📁 Project Structure

```
comsys-hackathon-5/
├── main.py                 # Main training and testing script
├── src/
│   └── verification_dataset.py  # Face verification dataset loader
├── models/                 # Saved model weights
│   ├── gender_classifier.pth
│   └── face_verification.pth
├── results/                # Output results and visualizations
│   ├── test_results.json
│   ├── confusion_matrices/
│   └── training_plots/
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 📦 Download Pretrained Models

Download the pretrained models required for testing:

| Model | Task | Link |
|-------|------|------|
| `gender_classifier.pth` | Task A: Gender Classification | [Download](https://drive.google.com/file/d/1ppsem-CfcSGW9PDyjCLINkUwg-DkDTEZ/view?usp=drive_link) |
| `face_verification.pth` | Task B: Face Verification | [Download](https://drive.google.com/file/d/1gwFCYWTImXGoUhKXzEB6zCOZOGNrU8OJ/view?usp=drive_link) |

After downloading, place the files inside the `models/` folder.


## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow
```

### Dataset Structure

```
data/raw/Comys_Hackathon5/
├── taska/
│   ├── train/female/, train/male/
│   └── val/female/, val/male/
└── taskb/
    ├── train/person_X/{clear, distortion/*.jpg}
    └── val/person_X/{clear, distortion/*.jpg}
```

## 💻 Usage

### Training

data_path=data/raw/Comys_Hackathon5

#### Train Gender Classification Only
```bash
 main.py --mode train --task gender --data_path $data_path
```

#### Train Face Verification Only
```bash
main.py --mode train --task verification --data_path $data_path
```

#### Train Both Tasks
```bash
main.py --mode train --task both --data_path $data_path
```

#### Fine-tuning (Optional)
```bash
python main.py --mode train --data_path /path/to/dataset --task gender --finetune
```

### Testing

data_path=data/raw/Comys_Hackathon5

#### Test Gender Classification
```bash
python main.py --mode test --task gender --data_path $data_path
```

#### Test Face Verification
```bash
python main.py --mode test --task verification --data_path $data_path
```

#### Test Both Tasks
```bash
python main.py --mode test --task both --data_path $data_path
```

## 📊 Model Details

### Task A: Gender Classification
- **Architecture:** ResNet-18 with custom classification head
- **Input Size:** 224×224 RGB images
- **Loss Function:** Cross-entropy with class weighting
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** StepLR (step_size=7, gamma=0.1)

### Task B: Face Verification
- **Architecture:** ResNet-50 with embedding layer (512-dim)
- **Loss Function:** Contrastive Loss (margin=1.0)
- **Similarity Metric:** Cosine similarity
- **Threshold:** 0.7 (configurable)

## 🔍 Data Augmentation Strategy

The models use comprehensive data augmentation to handle adverse conditions:
- Random horizontal flipping (50% probability)
- Random rotation (±10 degrees)
- Color jittering (brightness, contrast, saturation, hue)
- Standard ImageNet normalization

## 📈 Training Insights

### Gender Classification
- Achieves 94%+ validation accuracy
- Robust to class imbalance through weighted loss
- Fine-tuning capability for marginal improvements

### Face Verification
- **Outstanding 99.09% accuracy** with perfect precision
- Uses contrastive learning for robust embeddings
- L2-normalized embeddings for superior similarity computation
- Exceptionally handles distorted images through advanced data augmentation

## 🔬 Evaluation Metrics

The solution provides comprehensive evaluation:
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrices** with visualizations
- **Training History Plots** (loss and accuracy curves)
- **Detailed Test Results** saved in JSON format

## 🛠️ Advanced Features

### Fine-tuning Support
```bash
python main.py --mode train --data_path /path/to/dataset --task gender --finetune
```

## 📋 Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.62.0
Pillow>=8.3.0
```

## 🎛️ Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--mode` | Training or testing mode | Required | `train`, `test` |
| `--data_path` | Path to dataset directory | Required | - |
| `--task` | Which task(s) to run | `both` | `gender`, `verification`, `both` |
| `--gender_epochs` | Epochs for gender training | 20 | Any integer |
| `--finetune` | Enable fine-tuning | False | Flag |

## 📊 Sample Results

### Gender Classification Confusion Matrix
```
           Predicted
Actual   Female  Male
Female     67     12
Male       13    330
```

### Training Performance
- **Gender Classification:** Converges to 94%+ accuracy with excellent generalization
- **Face Verification:** Achieves exceptional 99%+ accuracy with perfect precision
- **Combined Weighted Score:** Outstanding 97.58% overall performance

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in the code (default: 32)
   - Use smaller model or reduce image resolution

2. **Dataset Path Issues**
   - Ensure dataset structure matches the expected format
   - Check file permissions and paths

3. **Missing Dependencies**
   - Install all requirements: `pip install -r requirements.txt`

## 📝 Technical Notes

- **GPU Recommended:** Training benefits significantly from CUDA acceleration
- **Memory Requirements:** ~8GB GPU memory for full training
- **Training Time:** ~30-45 minutes per task on modern GPU
- **Reproducibility:** Set random seeds for consistent results

## 🔄 Model Architecture Flow

```
Input Image (224×224×3)
    ↓
Data Augmentation
    ↓
Feature Extraction (ResNet Backbone)
    ↓
Task-Specific Head
    ↓
Classification/Embedding Output
```

## 🎯 Hackathon Submission Checklist

- ✅ Training and validation results for both tasks
- ✅ Well-documented source code
- ✅ Pretrained model weights
- ✅ Test script with evaluation metrics
- ✅ Clean, reproducible code
- ✅ Comprehensive README

## 🏅 Performance Highlights

- **🎯 Near-Perfect Face Verification:** 99.09% accuracy with 100% precision
- **🔥 Exceptional Combined Score:** 97.58% weighted performance
- **💪 Robust to Visual Distortions:** Superior handling of image degradations
- **⚡ High Gender Classification:** 94%+ accuracy on binary classification
- **🚀 Production Excellence:** State-of-the-art results with comprehensive evaluation
- **🎪 Competition Ready:** Top-tier performance metrics across all tasks

## 📄 License

This project is developed for the COMSYS Hackathon-5 competition.

---
