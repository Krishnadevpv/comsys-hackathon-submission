#!/usr/bin/env python3
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
    print("\n📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print("\nTask A - Gender Classification:")
    print(f"Accuracy:  {gender_metrics['accuracy']:.4f}")
    print(f"Precision: {gender_metrics['precision']:.4f}")
    print(f"Recall:    {gender_metrics['recall']:.4f}")
    print(f"F1-Score:  {gender_metrics['f1_score']:.4f}")
    
    print("\nTask B - Face Recognition:")
    print(f"Accuracy:  {face_metrics['accuracy']:.4f}")
    print(f"Precision: {face_metrics['precision']:.4f}")
    print(f"Recall:    {face_metrics['recall']:.4f}")
    print(f"F1-Score:  {face_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
