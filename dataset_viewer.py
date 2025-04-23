# Code for Paper: "Polimorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Author: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3

import tkinter as tk
from tkinter import ttk
import pickle
import torch
import numpy as np
from PIL import Image, ImageTk
import os
from threading import Thread

class LoadingDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Loading Dataset")
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center the dialog
        w = 300
        h = 100
        ws = parent.winfo_screenwidth()
        hs = parent.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.top.geometry(f'{w}x{h}+{int(x)}+{int(y)}')
        
        self.frame = ttk.Frame(self.top, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.label = ttk.Label(self.frame, text="Loading dataset...")
        self.label.grid(row=0, column=0, pady=5)
        
        self.progress = ttk.Progressbar(self.frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, pady=5)
        self.progress.start(10)

    def destroy(self):
        self.top.destroy()

class DatasetSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Selector")
        
        self.frame = ttk.Frame(root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(self.frame, text="Select a dataset to view:").grid(row=0, column=0, pady=10)
        
        # Scan for available datasets
        self.datasets = self.find_datasets()
        
        # Dataset listbox
        self.listbox = tk.Listbox(self.frame, height=10, width=50)
        self.listbox.grid(row=1, column=0, pady=5)
        
        for dataset in self.datasets:
            self.listbox.insert(tk.END, os.path.basename(dataset))
        
        if self.datasets:
            self.listbox.selection_set(0)
        
        ttk.Button(self.frame, text="Open Dataset", command=self.open_dataset).grid(row=2, column=0, pady=10)
    
    def find_datasets(self):
        # Search recursively in all subfolders within 'data' for .pkl files
        datasets = []
        for root, _, files in os.walk('data'):
            for file in files:
                if file.endswith('.pkl'):
                    datasets.append(os.path.join(root, file))
        return datasets
    
    def open_dataset(self):
        selection = self.listbox.curselection()
        if not selection:
            return
        
        dataset_path = self.datasets[selection[0]]
        
        # Show loading dialog
        loading_dialog = LoadingDialog(self.root)
        self.root.update()
        
        # Load dataset in a separate thread
        def load_dataset():
            try:
                with open(dataset_path, 'rb') as f:
                    data = pickle.load(f)
                self.root.after(0, lambda: self.show_viewer(data, dataset_path))
            finally:
                self.root.after(0, loading_dialog.destroy)
        
        Thread(target=load_dataset, daemon=True).start()
    
    def show_viewer(self, data, dataset_path):
        # Hide selector frame
        self.frame.grid_remove()
        
        # Show viewer
        DatasetViewer(self.root, data['features'], data['labels'], dataset_path)

class DatasetViewer:
    def __init__(self, root, features, labels, dataset_path):
        self.root = root
        self.features = features
        self.labels = labels
        self.dataset_path = dataset_path
        self.current_idx = 0
        
        # Default shape configuration
        self.feature_shape = None
        self.auto_detect_shape = True
        
        # Main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Image display
        self.canvas = tk.Canvas(self.main_frame, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Navigation frame
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_sample).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_sample).grid(row=0, column=1, padx=5)
        
        # Info frame
        info_frame = ttk.Frame(self.main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Label(info_frame, text="Sample Index:").grid(row=0, column=0, padx=5)
        self.index_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.index_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Button(info_frame, text="Go", command=self.go_to_index).grid(row=0, column=2, padx=5)
        
        # Label display
        self.label_var = tk.StringVar()
        ttk.Label(info_frame, text="Label:").grid(row=1, column=0, padx=5, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.label_var).grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W)
        
        # Features as text display
        self.features_text_var = tk.StringVar()
        ttk.Label(info_frame, text="Features as Text:").grid(row=2, column=0, padx=5, sticky=tk.W)
        
        # Use a text widget for features text to allow scrolling and selection
        features_frame = ttk.Frame(info_frame)
        features_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        self.features_text = tk.Text(features_frame, width=30, height=3, wrap=tk.WORD)
        self.features_text.pack(side=tk.LEFT, fill=tk.BOTH)
        
        features_scrollbar = ttk.Scrollbar(features_frame, command=self.features_text.yview)
        features_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.features_text.config(yscrollcommand=features_scrollbar.set)
        
        # Total samples
        ttk.Label(info_frame, text=f"Total Samples: {len(self.labels)}").grid(row=3, column=0, columnspan=3, pady=5)
        
        # Display first sample
        self.show_current_sample()
        
        # Bind keyboard shortcuts
        root.bind('<Left>', lambda e: self.prev_sample())
        root.bind('<Right>', lambda e: self.next_sample())
        root.bind('<Return>', lambda e: self.go_to_index())
    
    def create_menu_bar(self):
        # Create menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # Create Dataset menu
        dataset_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Dataset", menu=dataset_menu)
        
        # Add shape configuration submenu
        shape_menu = tk.Menu(dataset_menu, tearoff=0)
        dataset_menu.add_cascade(label="Configure Shape", menu=shape_menu)
        
        # Add common shape options
        shape_menu.add_command(label="Auto Detect (Default)", command=lambda: self.set_shape("auto"))
        shape_menu.add_command(label="MNIST (28x28)", command=lambda: self.set_shape((28, 28)))
        shape_menu.add_command(label="8x16 (128 features)", command=lambda: self.set_shape((8, 16)))
        shape_menu.add_command(label="16x8 (128 features)", command=lambda: self.set_shape((16, 8)))
        shape_menu.add_command(label="11x11+7 (128 features)", command=lambda: self.set_shape((11, 12)))  # 11x11=121, +7 more = 128
        shape_menu.add_separator()
        shape_menu.add_command(label="Custom Shape...", command=self.set_custom_shape)
        
        # Add info option
        dataset_menu.add_command(label="Dataset Info", command=self.show_dataset_info)
        
        # Add file menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Different Dataset", command=self.open_different_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
    def set_shape(self, shape):
        if shape == "auto":
            self.auto_detect_shape = True
            self.feature_shape = None
        else:
            self.auto_detect_shape = False
            self.feature_shape = shape
        
        # Refresh the current image with the new shape
        self.show_current_sample()
    
    def set_custom_shape(self):
        # Create a dialog to get custom shape
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Custom Shape")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Frame for inputs
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Height and width inputs
        ttk.Label(frame, text="Height:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        height_var = tk.StringVar()
        height_entry = ttk.Entry(frame, textvariable=height_var, width=10)
        height_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Width:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        width_var = tk.StringVar()
        width_entry = ttk.Entry(frame, textvariable=width_var, width=10)
        width_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Feature size display
        feature_size = self.features[0].numel()
        ttk.Label(frame, text=f"Total Features: {feature_size}").grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Error message display
        error_var = tk.StringVar()
        error_label = ttk.Label(frame, textvariable=error_var, foreground="red")
        error_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        def apply_shape():
            try:
                height = int(height_var.get())
                width = int(width_var.get())
                
                if height * width != feature_size:
                    error_var.set(f"Error: Height × Width must equal {feature_size}")
                    return
                
                self.set_shape((height, width))
                dialog.destroy()
            except ValueError:
                error_var.set("Error: Please enter valid numbers")
        
        ttk.Button(button_frame, text="Apply", command=apply_shape).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Set focus to height entry
        height_entry.focus_set()
    
    def show_dataset_info(self):
        # Create a dialog to show dataset info
        dialog = tk.Toplevel(self.root)
        dialog.title("Dataset Information")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Frame for info
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset info
        info_text = f"""
Dataset Path: {self.dataset_path}
Number of Samples: {len(self.labels)}
Feature Size: {self.features[0].numel()}
Current Shape: {self.feature_shape if not self.auto_detect_shape else 'Auto-detected'}
        """
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=10)
    
    def create_image(self, features):
        # Get the size of the features
        feature_size = features.numel()
        
        # Determine the appropriate dimensions for visualization
        if self.auto_detect_shape:
            if feature_size == 784:  # MNIST (28x28)
                img_size = (28, 28)
            elif feature_size == 128:  # 128 features
                img_size = (8, 16)
            else:
                # For other sizes, try to make a square-ish image
                side = int(np.sqrt(feature_size))
                if side * side == feature_size:
                    img_size = (side, side)
                else:
                    # Find the closest factors
                    for i in range(side, 0, -1):
                        if feature_size % i == 0:
                            img_size = (i, feature_size // i)
                            break
                    else:
                        # If no factors found, use a rectangular shape
                        img_size = (1, feature_size)
        else:
            # Use the user-specified shape
            img_size = self.feature_shape
        
        # Reshape and convert to numpy array
        img_array = features.reshape(img_size).numpy()
        
        # Scale to 0-255 for display
        img_array = (img_array * 255).astype(np.uint8)
        
        # Scale up the image for better visibility (maintain aspect ratio)
        scale_factor = 280 / max(img_size)
        scaled_size = (int(img_size[1] * scale_factor), int(img_size[0] * scale_factor))
        
        # Create and return the image
        img = Image.fromarray(img_array).resize(scaled_size, Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(img)
    
    def show_current_sample(self):
        # Update index display
        self.index_var.set(str(self.current_idx))
        
        # Get current sample
        features = self.features[self.current_idx]
        label = self.labels[self.current_idx].item()
        
        # Create and display image
        self.current_image = self.create_image(features)
        self.canvas.delete("all")
        self.canvas.create_image(140, 140, image=self.current_image)
        
        # Update label display
        self.label_var.set(f"ASCII: {label} ('{chr(label)}')")
        
        # Convert features to ASCII text
        feature_values = features.numpy()
        feature_text = ""
        
        # Try to interpret features as ASCII characters
        try:
            # Check if features are binary (0s and 1s)
            is_binary = np.all(np.isin(feature_values, [0, 1]))
            
            if is_binary:
                # Process as 8-bit sequences (bytes)
                feature_size = len(feature_values)
                bytes_data = []
                
                # Process every 8 bits as a byte
                for i in range(0, feature_size, 8):
                    if i + 8 <= feature_size:
                        # Convert 8 bits to a byte
                        byte_value = 0
                        for j in range(8):
                            byte_value = (byte_value << 1) | int(feature_values[i + j])
                        bytes_data.append(byte_value)
                
                # Convert bytes to ASCII characters (only printable range)
                printable_chars = []
                for byte in bytes_data:
                    if 32 <= byte <= 126:  # Printable ASCII range
                        printable_chars.append(chr(byte))
                
                if printable_chars:
                    feature_text = "".join(printable_chars)
                else:
                    # If no printable characters, show binary representation
                    feature_text = "Binary: " + " ".join([f"{byte:08b}" for byte in bytes_data])
            
            # If not binary or no bytes were processed, fall back to previous methods
            if not is_binary or not feature_text:
                if feature_values.max() <= 1.0:
                    # Scale to 0-255 range
                    ascii_values = (feature_values * 255).astype(np.uint8)
                    # Filter to printable ASCII range (32-126)
                    printable_indices = np.where((ascii_values >= 32) & (ascii_values <= 126))[0]
                    if len(printable_indices) > 0:
                        printable_chars = [chr(int(ascii_values[i])) for i in printable_indices]
                        feature_text = "".join(printable_chars)
                    else:
                        feature_text = "(No printable ASCII characters found)"
                else:
                    # For features already in numeric range, convert directly
                    ascii_values = feature_values.astype(np.uint8)
                    # Filter to printable ASCII range (32-126)
                    printable_indices = np.where((ascii_values >= 32) & (ascii_values <= 126))[0]
                    if len(printable_indices) > 0:
                        printable_chars = [chr(int(ascii_values[i])) for i in printable_indices]
                        feature_text = "".join(printable_chars)
                    else:
                        feature_text = "(No printable ASCII characters found)"
        except Exception as e:
            feature_text = f"Error converting to ASCII: {str(e)}"
        
        # Update features text display
        self.features_text.config(state=tk.NORMAL)
        self.features_text.delete(1.0, tk.END)
        self.features_text.insert(tk.END, feature_text)
        self.features_text.config(state=tk.DISABLED)
    
    def next_sample(self):
        if self.current_idx < len(self.labels) - 1:
            self.current_idx += 1
            self.show_current_sample()
    
    def prev_sample(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_sample()
    
    def go_to_index(self):
        try:
            idx = int(self.index_var.get())
            if 0 <= idx < len(self.labels):
                self.current_idx = idx
                self.show_current_sample()
        except ValueError:
            pass
    
    def open_different_dataset(self):
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Remove the menu bar
        self.root.config(menu="")
        
        # Show the dataset selector again
        selector = DatasetSelector(self.root)
        selector.frame.grid()

def main():
    root = tk.Tk()
    app = DatasetSelector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
