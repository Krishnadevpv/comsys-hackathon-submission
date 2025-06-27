# setup_project.py - Creates complete project structure
import os
import json

def create_project_structure():
    """Create all necessary folders and files"""
    
    # Create directory structure
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "results",
        "notebooks",
        "src"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Create requirements.txt
    requirements = """torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.3.0
tqdm>=4.62.0
argparse
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
       f.write(requirements)
    print("📄 Created: requirements.txt")
    
    # Create README.md
    readme_content = """# COMSYS Hackathon-5 Solution
## Face Recognition and Gender Classification under Adverse Visual Conditions

### Team Information
- **Participant:** KRISHNADEV P V
- **Event:** COMSYS Hackathon-5, 2025
- **Theme:** Robust Face Recognition and Gender Classification under Adverse Visual Conditions

### Project Overview
This project tackles two computer vision tasks:
1. **Task A:** Gender Classification (Binary Classification) - 30% weight
2. **Task B:** Face Recognition/Matching (Multi-class Classification) - 70% weight

### Dataset: FACECOM
The FACECOM dataset contains face images under various challenging conditions:
- Motion Blur
- Overexposed/Sunny scenes  
- Foggy conditions
- Rainy weather simulation
- Low light visibility
- Uneven lighting or glare

### Installation
```bash
# Clone the repository
git clone [your-repo-url]
cd comsys-hackathon-5

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Training
```bash
python main.py --mode train --data_path path/to/dataset
```

#### Testing
```bash
python main.py --mode test --data_path path/to/test_data
```

### Project Structure
```
comsys-hackathon-5/
├── data/
│   ├── raw/              # Original FACECOM dataset
│   └── processed/        # Preprocessed data
├── models/               # Saved model weights
│   ├── gender_classifier.pth
│   └── face_recognition.pth
├── results/              # Output results and visualizations
├── src/                  # Source code modules
├── notebooks/            # Jupyter notebooks for exploration
├── main.py              # Main training/testing script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Model Architecture
- **Gender Classification:** ResNet18-based binary classifier
- **Face Recognition:** ResNet50-based embedding model with matching

### Results
Training and validation results will be saved in the `results/` directory.

### Submission
This repository contains:
- [x] Well-documented source code
- [x] Pretrained model weights
- [x] Test script with evaluation metrics
- [x] README with clear instructions

### Author
**KRISHNADEV P V**
COMSYS Hackathon-5 Participant
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
     f.write(readme_content)
    print("📄 Created: README.md")
    
    # Create simple test script
    test_script = """#!/usr/bin/env python3
# test.py - Simple test script for evaluation
import sys
import os
sys.path.append('.')

from main import COMSYSHackathonSolution

def main():
    if len(sys.argv) != 2:
        print("Usage: python test.py <test_data_path>")
        sys.exit(1)
    
    test_data_path = sys.argv[1]
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test data path '{test_data_path}' does not exist!")
        sys.exit(1)
    
    # Create solution and test
    solution = COMSYSHackathonSolution()
    gender_metrics, face_metrics = solution.test_models(test_data_path)
    
    # Print results
    print("\\n📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print("\\nTask A - Gender Classification:")
    print(f"Accuracy:  {gender_metrics['accuracy']:.4f}")
    print(f"Precision: {gender_metrics['precision']:.4f}")
    print(f"Recall:    {gender_metrics['recall']:.4f}")
    print(f"F1-Score:  {gender_metrics['f1_score']:.4f}")
    
    print("\\nTask B - Face Recognition:")
    print(f"Accuracy:  {face_metrics['accuracy']:.4f}")
    print(f"Precision: {face_metrics['precision']:.4f}")
    print(f"Recall:    {face_metrics['recall']:.4f}")
    print(f"F1-Score:  {face_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
"""
    
    with open("test.py", "w", encoding='utf-8') as f:
        f.write(test_script)
    print("📄 Created: test.py")
    
    print("\n✅ Project setup complete!")
    print("\nNext steps:")
    print("1. Put your FACECOM dataset in 'data/raw/'")
    print("2. Install requirements: pip install -r requirements.txt")
    print("3. Train models: python main.py --mode train --data_path data/raw")
    print("4. Test models: python main.py --mode test --data_path data/raw")

if __name__ == "__main__":
    create_project_structure()