# Code for Paper: "Polimorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Author: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3


import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm

class TextBinaryDataset(Dataset):
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
    """Process text file and convert to binary sequences with sliding windows."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert all valid ASCII characters to binary
    binary_data = []
    ascii_chars = []
    
    print("Converting characters to binary...")
    for char in tqdm(text, desc="Processing characters"):
        if ord(char) < 128:  # Only process ASCII characters
            binary_data.extend(char_to_binary(char))
            ascii_chars.append(char)
    
    # Create samples using sliding window
    window_size = 784  # Same as MNIST
    features = []
    labels = []
    
    # We need at least window_size binary digits plus one character for the label
    if len(binary_data) >= window_size + 8:
        # Calculate number of possible windows when sliding by 8 bits
        total_windows = (len(binary_data) - window_size) // 8
        print("\nCreating sliding windows...")
        for i in tqdm(range(total_windows), desc="Creating samples"):
            # Get window starting at i*8 (sliding by 8 bits each time)
            window_start = i * 8
            window = binary_data[window_start:window_start + window_size]
            
            # The next character after our window
            next_char_pos = (window_start + window_size) // 8
            if next_char_pos < len(ascii_chars):
                features.append(window)
                labels.append(ord(ascii_chars[next_char_pos]) & 127)  # Get ASCII value of next char
    
    return features, labels

def main():
    # Input and output paths
    # input_file = 'data/NLP/gutenberg/data/art_of_war_pg132.txt'
    input_file = 'data/NLP/gutenberg/data/corpus_fgmtw/corpus_fgmtw.txt'
    output_dir = 'data/NLP/raw'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the text file
    features, labels = process_text_file(input_file)
    
    # Create the dataset
    dataset = TextBinaryDataset(features, labels)
    
    # Save the dataset
    output_path = os.path.join(output_dir, 'corpus_fgmtw_dataset.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({'features': dataset.features, 'labels': dataset.labels}, f)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Sample entry:")
    print(f"Features shape: {dataset.features[0].shape}")
    print(f"Label (ASCII value): {dataset.labels[0].item()}")
    print(f"Label as character: {chr(dataset.labels[0].item())}")

if __name__ == "__main__":
    main()
