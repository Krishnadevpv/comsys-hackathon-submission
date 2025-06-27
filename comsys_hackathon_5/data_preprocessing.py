#!/usr/bin/env python3
"""
Data Preprocessing and Analysis Script
COMSYS Hackathon-5 - FACECOM Dataset
Author: KRISHNADEV P V
"""

import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json

def analyze_dataset(data_path):
    """Analyze the FACECOM dataset structure and statistics"""
    
    print("🔍 FACECOM Dataset Analysis")
    print("=" * 50)
    
    base_path = data_path
    
    if not os.path.exists(base_path):
        print(f"Error: Dataset not found at {base_path}")
        return
    
    analysis = {
        'task_a': {'train': {'male': 0, 'female': 0}, 'val': {'male': 0, 'female': 0}},
        'task_b': {'train': {}, 'val': {}},
        'image_stats': {'corrupted': [], 'total': 0}
    }
    
    # Analyze Task A (Gender Classification)
    print("\n📊 Task A - Gender Classification Analysis")
    print("-" * 40)
    
    for split in ['train', 'val']:
        task_a_path = os.path.join(base_path, 'taska', split)
        if os.path.exists(task_a_path):
            for gender in ['male', 'female']:
                gender_path = os.path.join(task_a_path, gender)
                if os.path.exists(gender_path):
                    count = len([f for f in os.listdir(gender_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    analysis['task_a'][split][gender] = count
                    print(f"{split.capitalize()} - {gender.capitalize()}: {count} images")
    
    total_task_a = sum(sum(split.values()) for split in analysis['task_a'].values())
    print(f"Total Task A images: {total_task_a}")
    
    # Analyze Task B (Face Recognition)
    print("\n👤 Task B - Face Recognition Analysis")
    print("-" * 40)
    
    # Initialize Task B structure
    for split in ['train', 'val']:
        analysis['task_b'][split] = {'num_persons': 0, 'persons': {}, 'total_images': 0}
    
    for split in ['train', 'val']:
        task_b_path = os.path.join(base_path, 'taskb', split)
        if os.path.exists(task_b_path):
            persons = os.listdir(task_b_path)
            analysis['task_b'][split]['num_persons'] = len(persons)
            
            total_images = 0
            for person in persons:
                person_path = os.path.join(task_b_path, person)
                if os.path.isdir(person_path):
                    count = len([f for f in os.listdir(person_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    analysis['task_b'][split]['persons'][person] = count
                    total_images += count
            
            analysis['task_b'][split]['total_images'] = total_images
            print(f"{split.capitalize()} - Persons: {len(persons)}, Images: {total_images}")
        else:
            print(f"TaskB {split} folder not found")
    
    # Check image integrity
    print("\n🔍 Checking Image Integrity...")
    corrupted_count = check_image_integrity(base_path)
    analysis['image_stats']['corrupted_count'] = corrupted_count
    
    # Calculate total images (fix the key error here)
    total_task_b = sum(
        analysis['task_b'][split]['total_images'] for split in ['train', 'val']
    )
    analysis['image_stats']['total'] = total_task_a + total_task_b
    
    # Save analysis
    os.makedirs('results', exist_ok=True)
    with open('results/dataset_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to results/dataset_analysis.json")
    
    # Create visualizations
    create_analysis_plots(analysis)
    
    return analysis

def check_image_integrity(base_path):
    """Check for corrupted images in the dataset"""
    
    corrupted_images = []
    total_checked = 0
    
    # Check Task A
    for split in ['train', 'val']:
        for gender in ['male', 'female']:
            gender_path = os.path.join(base_path, 'taska', split, gender)
            if os.path.exists(gender_path):
                for img_file in os.listdir(gender_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(gender_path, img_file)
                        try:
                            with Image.open(img_path) as img:
                                img.verify()
                            total_checked += 1
                        except Exception as e:
                            corrupted_images.append(img_path)
                            print(f"Corrupted image: {img_path} - {e}")
    
    # Check Task B
    for split in ['train', 'val']:
        task_b_path = os.path.join(base_path, 'taskb', split)
        if os.path.exists(task_b_path):
            for person in os.listdir(task_b_path):
                person_path = os.path.join(task_b_path, person)
                if os.path.isdir(person_path):
                    for img_file in os.listdir(person_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(person_path, img_file)
                            try:
                                with Image.open(img_path) as img:
                                    img.verify()
                                total_checked += 1
                            except Exception as e:
                                corrupted_images.append(img_path)
                                print(f"Corrupted image: {img_path} - {e}")
    
    print(f"Checked {total_checked} images, found {len(corrupted_images)} corrupted")
    return len(corrupted_images)

def create_analysis_plots(analysis):
    """Create visualization plots for dataset analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Task A Distribution
    ax1 = axes[0, 0]
    splits = ['train', 'val']
    male_counts = [analysis['task_a'][split]['male'] for split in splits]
    female_counts = [analysis['task_a'][split]['female'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    ax1.bar(x - width/2, male_counts, width, label='Male', color='skyblue')
    ax1.bar(x + width/2, female_counts, width, label='Female', color='pink')
    ax1.set_xlabel('Dataset Split')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Task A - Gender Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Task B Distribution
    ax2 = axes[0, 1]
    task_b_train = analysis['task_b']['train']['total_images']
    task_b_val = analysis['task_b']['val']['total_images']
    
    ax2.bar(['Train', 'Val'], [task_b_train, task_b_val], color=['lightgreen', 'lightcoral'])
    ax2.set_xlabel('Dataset Split')
    ax2.set_ylabel('Number of Images')
    ax2.set_title('Task B - Image Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Overall Task Distribution
    ax3 = axes[1, 0]
    total_task_a = sum(sum(split.values()) for split in analysis['task_a'].values())
    total_task_b = task_b_train + task_b_val
    
    # Handle case where both totals are 0
    if total_task_a == 0 and total_task_b == 0:
        ax3.text(0.5, 0.5, 'No Data Found', ha='center', va='center', fontsize=14)
        ax3.set_title('Overall Dataset Distribution - No Data')
    else:
        if total_task_a == 0:
            ax3.pie([total_task_b], labels=['Task B\n(Face Recognition)'], 
                    autopct='%1.1f%%', colors=['lightgreen'])
        elif total_task_b == 0:
            ax3.pie([total_task_a], labels=['Task A\n(Gender)'], 
                    autopct='%1.1f%%', colors=['lightblue'])
        else:
            ax3.pie([total_task_a, total_task_b], labels=['Task A\n(Gender)', 'Task B\n(Face Recognition)'], 
                    autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
        ax3.set_title('Overall Dataset Distribution')
    
    # Person Distribution in Task B
    ax4 = axes[1, 1]
    train_persons = analysis['task_b']['train']['num_persons']
    val_persons = analysis['task_b']['val']['num_persons']
    
    ax4.bar(['Train', 'Val'], [train_persons, val_persons], color=['orange', 'purple'])
    ax4.set_xlabel('Dataset Split')
    ax4.set_ylabel('Number of Persons')
    ax4.set_title('Task B - Number of Unique Persons')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Analysis plots saved to results/dataset_analysis.png")

def create_sample_visualizations(data_path, num_samples=5):
    """Create sample visualizations from each task"""

    print("\n🖼️ Creating Sample Visualizations...")

    base_path = data_path

    if not os.path.exists(base_path):
        print(f"❌ Base path not found: {base_path}")
        return

    # Task A Samples
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    # Male samples
    male_path = os.path.join(base_path, 'taska', 'train', 'male')
    male_images = [f for f in os.listdir(male_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

    for i in range(num_samples):
        ax = axes[0, i]
        if i < len(male_images):
            img_path = os.path.join(male_path, male_images[i])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'Male {i+1}')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center')
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center')
        ax.axis('off')

    # Female samples
    female_path = os.path.join(base_path, 'taska', 'train', 'female')
    female_images = [f for f in os.listdir(female_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

    for i in range(num_samples):
        ax = axes[1, i]
        if i < len(female_images):
            img_path = os.path.join(female_path, female_images[i])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'Female {i+1}')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center')
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center')
        ax.axis('off')

    plt.suptitle('Task A - Gender Classification Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/task_a_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Task B Samples - Show clear + distorted
    # Task B Samples - Show clear + distorted
    task_b_path = os.path.join(base_path, 'taskb', 'train')
    if os.path.exists(task_b_path):
        persons = [d for d in os.listdir(task_b_path) if os.path.isdir(os.path.join(task_b_path, d))][:3]
        fig, axes = plt.subplots(len(persons), 4, figsize=(14, 3.5 * len(persons)))

        if len(persons) == 1:
            axes = axes.reshape(1, -1)

        for row_idx, person in enumerate(persons):
            person_path = os.path.join(task_b_path, person)

            # Get clear image
            clear_img = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            clear_img_path = os.path.join(person_path, clear_img[0]) if clear_img else None

            # Get distorted images from 'distortion' subfolder
            distorted_folder = os.path.join(person_path, 'distortion')
            distorted_imgs = []
            if os.path.exists(distorted_folder):
                distorted_imgs = sorted([
                    os.path.join(distorted_folder, f)
                    for f in os.listdir(distorted_folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])[:3]

            selected_imgs = [clear_img_path] + distorted_imgs

            for col_idx in range(4):
                ax = axes[row_idx][col_idx]
                if col_idx < len(selected_imgs) and selected_imgs[col_idx]:
                    try:
                        img = Image.open(selected_imgs[col_idx])
                        ax.imshow(img)
                        ax.set_title(os.path.basename(selected_imgs[col_idx])[:14])
                    except:
                        ax.text(0.5, 0.5, 'Error Loading', ha='center')
                else:
                    ax.text(0.5, 0.5, 'No Image', ha='center')
                ax.axis('off')

        plt.suptitle('Task B - Face Recognition: Clear + Distorted Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/task_b_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("❌ Task B path not found")

    print("✅ Sample visualizations saved!")

def setup_data_structure(source_path, target_path):
    """Setup proper data structure for training"""
    
    print(f"\n📁 Setting up data structure...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    source_base = os.path.join(source_path, "Comys_Hackathon5")
    
    if not os.path.exists(source_base):
        print(f"Error: Source path {source_base} not found!")
        return False
    
    # Create target directories
    os.makedirs(target_path, exist_ok=True)
    
    # Copy Task A structure
    for split in ['train', 'val']:
        for gender in ['male', 'female']:
            source_dir = os.path.join(source_base, 'taska', split, gender)
            target_dir = os.path.join(target_path, 'taska', split, gender)
            
            if os.path.exists(source_dir):
                os.makedirs(target_dir, exist_ok=True)
                print(f"Created: {target_dir}")
    
    # Copy Task B structure
    for split in ['train', 'val']:
        source_dir = os.path.join(source_base, 'taskb', split)
        target_dir = os.path.join(target_path, 'taskb', split)
        
        if os.path.exists(source_dir):
            os.makedirs(target_dir, exist_ok=True)
            for person in os.listdir(source_dir):
                person_source = os.path.join(source_dir, person)
                person_target = os.path.join(target_dir, person)
                if os.path.isdir(person_source):
                    os.makedirs(person_target, exist_ok=True)
            print(f"Created: {target_dir}")
    
    print("✅ Data structure setup complete!")
    return True

def main():
    """Main function for data preprocessing"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='FACECOM Dataset Analysis and Preprocessing')
    parser.add_argument('--data_path', required=True, help='Path to raw dataset')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset')
    parser.add_argument('--visualize', action='store_true', help='Create sample visualizations')
    parser.add_argument('--setup', action='store_true', help='Setup data structure')
    parser.add_argument('--target_path', help='Target path for setup (default: data/raw)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist!")
        return
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    if args.analyze:
        analysis = analyze_dataset(args.data_path)
    
    if args.visualize:
        create_sample_visualizations(args.data_path)
    
    if args.setup:
        target_path = args.target_path or 'data/raw'
        setup_data_structure(args.data_path, target_path)
    
    print("\n🎉 Data preprocessing complete!")

if __name__ == "__main__":
    main()