<<<<<<< HEAD
# comsys-hackathon-5
Solution for COMSYS Hackathon-5 - Gender Classification &amp; Face Verification
=======
# COMSYS Hackathon-5 Solution
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
>>>>>>> 21c0df44 (Initial commit for COMSYS Hackathon project)
