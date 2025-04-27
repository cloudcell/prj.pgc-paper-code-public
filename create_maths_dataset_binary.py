#!/usr/bin/env python3
"""
Script to convert arithmetic text datasets to binary format with sliding windows.
Takes the arithmetic datasets from data/ARITHMETIC/maths and creates binary datasets
with a sliding window of 16 characters, moving 1 character at a time.
Each window predicts the next character in the sequence.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import glob

# Constants
WINDOW_SIZE = 16  # Size of the sliding window in characters
SLIDE_STEP = 1    # Move the window by 1 character at a time

class ArithmeticBinaryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation."""
    # Only consider ASCII characters (0-127)
    ascii_val = ord(char) & 127
    return [int(b) for b in format(ascii_val, '08b')]

def process_text_file(file_path):
    """
    Process text file and convert to binary sequences with sliding windows.
    Uses a window of 16 characters and slides by 1 character at a time.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_features = []
    all_labels = []
    
    print(f"Processing {file_path}...")
    for line_idx, line in enumerate(tqdm(lines, desc=f"Processing lines in {os.path.basename(file_path)}")):
        # Remove newline character but keep all spaces
        if line.endswith('\n'):
            line = line[:-1]
        
        # We need at least WINDOW_SIZE + 1 characters (window + next character to predict)
        if len(line) < WINDOW_SIZE + 1:
            continue  # Skip lines that are too short
        
        # Process each possible window in the line
        for i in range(len(line) - WINDOW_SIZE):
            # Get the current window
            window = line[i:i+WINDOW_SIZE]
            
            # The next character after our window is the label
            next_char = line[i+WINDOW_SIZE]
            
            # Convert window characters to binary
            window_binary = []
            for char in window:
                window_binary.extend(char_to_binary(char))
            
            # Add to our dataset
            all_features.append(window_binary)
            all_labels.append(ord(next_char) & 127)  # Get ASCII value of next char
    
    return all_features, all_labels

def main():
    # Input and output paths
    input_dir = 'data/ARITHMETIC/maths'
    output_dir = 'data/ARITHMETIC/binary'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all text files in the input directory
    input_files = glob.glob(os.path.join(input_dir, '*.txt'))
    
    if not input_files:
        print(f"No text files found in {input_dir}")
        return
    
    # Process each file
    for input_file in input_files:
        file_basename = os.path.basename(input_file)
        file_name = os.path.splitext(file_basename)[0]
        
        # Process the text file
        features, labels = process_text_file(input_file)
        
        if not features:
            print(f"No valid samples generated from {file_basename}")
            continue
        
        # Create the dataset
        dataset = ArithmeticBinaryDataset(features, labels)
        
        # Save the dataset
        output_path = os.path.join(output_dir, f'{file_name}_binary.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump({'features': dataset.features, 'labels': dataset.labels}, f)
        
        print(f"Dataset created for {file_basename} with {len(dataset)} samples")
        print(f"Sample entry:")
        print(f"Features shape: {dataset.features[0].shape}")
        print(f"Label (ASCII value): {dataset.labels[0].item()}")
        print(f"Label as character: {chr(dataset.labels[0].item())}")
        print("-" * 50)
    
    print("All datasets processed successfully!")

if __name__ == "__main__":
    main()
