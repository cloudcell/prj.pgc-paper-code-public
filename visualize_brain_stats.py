#!/usr/bin/env python3

# Code for Paper: "Polymorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Author: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3


"""
Brain Statistics Visualization Script

This script visualizes data from brain_stats_train_epoch_*.json files in a selected stats folder.
It ensures correct numerical ordering of the epoch files.
"""

import os
import json
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, Scale, Button, Frame, Label, HORIZONTAL
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import argparse
import imageio
import tempfile
from PIL import Image
import io

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way (e.g., epoch_1, epoch_2, ..., epoch_10)
    instead of lexicographical sorting (epoch_1, epoch_10, epoch_2, ...)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def extract_epoch_number(filename):
    """Extract the epoch number from the filename."""
    match = re.search(r'brain_stats_train_epoch_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return 0

def load_stats_files(stats_dir):
    """Load all brain_stats_train_epoch_*.json files from the given directory."""
    pattern = os.path.join(stats_dir, "brain_stats_train_epoch_*.json")
    files = glob.glob(pattern)
    
    # Sort files by epoch number
    files.sort(key=extract_epoch_number)
    
    if not files:
        print(f"No brain_stats_train_epoch_*.json files found in {stats_dir}")
        return []
    
    print(f"Found {len(files)} epoch files.")
    print(f"First file: {os.path.basename(files[0])}")
    print(f"Last file: {os.path.basename(files[-1])}")
    
    return files

def load_json_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def visualize_top_blocks(data, epoch, ax=None):
    """Visualize the top blocks as a 3D scatter plot."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    
    # Extract coordinates and counts
    coords = np.array([block['coords'] for block in data['top_blocks']])
    counts = np.array([block['count'] for block in data['top_blocks']])
    
    # Normalize counts for size
    sizes = 50 * (counts / counts.max())
    
    # Create a custom colormap
    cmap = plt.cm.viridis
    
    # Plot the scatter points
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=sizes, c=counts, cmap=cmap, alpha=0.7
    )
    
    # Add a colorbar
    plt.colorbar(scatter, ax=ax, label='Count')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Top Blocks - Epoch {epoch}')
    
    # Find the maximum coordinate values for each dimension
    if len(coords) > 0:
        max_x = np.max(coords[:, 0]) + 0.5
        max_y = np.max(coords[:, 1]) + 0.5
        max_z = np.max(coords[:, 2]) + 0.5
    else:
        max_x = max_y = max_z = 4  # Default if no blocks
    
    # Set axis limits based on the maximum values found
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    
    return ax

def visualize_top_pathways(data, epoch, ax=None):
    """Visualize the top pathways as 3D lines."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    
    # Create a colormap
    cmap = plt.cm.plasma
    
    # Find the maximum coordinate values across all pathways
    all_coords = []
    for pathway_data in data['top_pathways']:
        pathway = np.array(pathway_data['pathway'])
        all_coords.append(pathway)
    
    if all_coords:
        all_coords = np.vstack(all_coords)
        max_x = np.max(all_coords[:, 0]) + 0.5
        max_y = np.max(all_coords[:, 1]) + 0.5
        max_z = np.max(all_coords[:, 2]) + 0.5
    else:
        max_x = max_y = max_z = 4  # Default if no pathways
    
    # Plot each pathway
    for i, pathway_data in enumerate(data['top_pathways'][:5]):  # Limit to top 5 for clarity
        pathway = np.array(pathway_data['pathway'])
        count = pathway_data['count']
        
        # Normalize count for color
        color = cmap(i / min(5, len(data['top_pathways'])))
        
        # Plot the pathway as a line
        ax.plot(pathway[:, 0], pathway[:, 1], pathway[:, 2], 
                marker='o', linestyle='-', linewidth=2, 
                color=color, label=f'Pathway {i+1} (count: {count})')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Top Pathways - Epoch {epoch}')
    
    # Set axis limits based on the maximum values found
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    
    # Add legend
    ax.legend()
    
    return ax

class BrainStatsVisualizer:
    def __init__(self, files=None):
        self.files = files or []
        self.current_frame = 0
        self.is_playing = False
        self.interval = 500  # Default interval in ms (2 fps)
        self.setup_ui()
        
    def setup_ui(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Brain Stats Visualizer")
        self.root.geometry("1200x800")
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create a frame for the matplotlib figure
        self.fig_frame = Frame(self.root)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the matplotlib figure
        self.fig = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
        # Embed the matplotlib figure in the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add the matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame)
        self.toolbar.update()
        
        # Create a frame for the controls
        self.control_frame = Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add a slider for the current frame
        self.frame_label = Label(self.control_frame, text="Frame:")
        self.frame_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.frame_slider = Scale(self.control_frame, from_=0, to=max(len(self.files)-1, 1), 
                                 orient=HORIZONTAL, length=300, 
                                 command=self.on_frame_change)
        self.frame_slider.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add a slider for the playback speed (logarithmic scale)
        self.speed_label = Label(self.control_frame, text="Speed (fps):")
        self.speed_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.speed_slider = Scale(self.control_frame, from_=1, to=100, 
                                 orient=HORIZONTAL, length=200,
                                 label="",
                                 command=self.on_speed_change)
        self.speed_slider.set(20)  # Default speed (about 2 fps)
        self.speed_slider.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add current speed display label
        self.speed_display = Label(self.control_frame, text="2.0 fps")
        self.speed_display.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add play/pause button
        self.play_button = Button(self.control_frame, text="Play", 
                                 command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add record GIF button
        self.record_button = Button(self.control_frame, text="Record GIF", 
                                   command=self.record_gif)
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add status label
        self.status_label = Label(self.root, text="")
        self.status_label.pack(side=tk.BOTTOM, pady=(0, 10))
        
        # Initialize the speed with the default value
        self.on_speed_change(self.speed_slider.get())
        
        # Draw the initial frame if files are loaded
        if self.files:
            self.update_frame(0)
        else:
            self.status_label.config(text="No data loaded. Use File > Open to load brain stats files.")
            
    def create_menu_bar(self):
        """Create the menu bar with File and Help menus."""
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Stats Folder...", command=self.open_stats_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Usage Guide", command=self.show_usage_guide)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def open_stats_folder(self):
        """Open a stats folder and load the brain stats files."""
        stats_dir = select_stats_folder()
        if not stats_dir:
            return
        
        # Load the files
        files = load_stats_files(stats_dir)
        if not files:
            self.status_label.config(text=f"No brain_stats_train_epoch_*.json files found in {stats_dir}")
            return
        
        # Update the visualizer with the new files
        self.files = files
        
        # Update the frame slider range
        self.frame_slider.config(from_=0, to=len(self.files)-1)
        
        # Draw the first frame
        self.update_frame(0)
    
    def show_usage_guide(self):
        """Show a usage guide dialog."""
        guide_text = """
Brain Stats Visualizer - Usage Guide

Navigation:
- Use the frame slider to navigate between epochs
- Use the play/pause button to animate through epochs
- Adjust the speed slider to control animation speed
- Use the matplotlib toolbar for zooming, panning, and saving images

Visualization:
- Left plot: Top blocks as 3D scatter plot
- Right plot: Top pathways as 3D lines
- Colors indicate frequency/count

Recording:
- Click "Record GIF" to save the animation as a GIF file
- For large datasets, the recorder will sample frames to keep file size reasonable

Tips:
- Use logarithmic speed control for fine adjustments at lower speeds
- Rotate the 3D plots using the mouse for better viewing angles
- Save individual frames using the matplotlib toolbar
        """
        
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Usage Guide")
        guide_window.geometry("600x500")
        
        text_widget = tk.Text(guide_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, guide_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        close_button = Button(guide_window, text="Close", command=guide_window.destroy)
        close_button.pack(pady=10)
    
    def show_about(self):
        """Show an about dialog."""
        about_text = """
Brain Stats Visualizer

A tool for visualizing brain statistics data from JSON files.

Features:
- 3D visualization of brain blocks and pathways
- Animation of epoch progression
- GIF recording capability
- Interactive controls

Created: April 2025
        """
        
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("400x300")
        
        text_widget = tk.Text(about_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        close_button = Button(about_window, text="Close", command=about_window.destroy)
        close_button.pack(pady=10)
    
    def update_frame(self, frame_idx):
        """Update the visualization with the given frame index."""
        self.current_frame = frame_idx
        
        # Clear the figure completely, including colorbars
        self.fig.clear()
        
        # Recreate the axes with 3D projection
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
        # Load the data for the current frame
        file_path = self.files[frame_idx]
        data = load_json_data(file_path)
        
        if data:
            epoch = data.get('epoch', extract_epoch_number(file_path))
            
            # Update the frame slider if it's not the source of the update
            if self.frame_slider.get() != frame_idx:
                self.frame_slider.set(frame_idx)
            
            # Visualize the data
            visualize_top_blocks(data, epoch, self.ax1)
            visualize_top_pathways(data, epoch, self.ax2)
            
            # Update the status label
            self.status_label.config(text=f"Showing epoch {epoch} ({frame_idx+1}/{len(self.files)})")
        
        # Redraw the canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_frame_change(self, value):
        """Called when the frame slider value changes."""
        frame_idx = int(float(value))
        if frame_idx != self.current_frame:
            self.update_frame(frame_idx)
    
    def on_speed_change(self, value):
        """Called when the speed slider value changes."""
        # Convert the slider value (1-100) to a logarithmic scale for better control
        # This gives finer control at lower speeds and broader adjustments at higher speeds
        slider_value = int(float(value))
        
        # Apply logarithmic scaling: fps ranges from 0.1 to 30
        # log_scale goes from log(1) to log(101) which is 0 to ~4.6
        # We map this to 0.1 to 30 fps
        if slider_value == 1:  # Handle the special case for slider value 1
            fps = 0.1
        else:
            log_scale = np.log(slider_value)
            fps = 0.1 + (30.0 - 0.1) * (log_scale / np.log(100))
        
        # Set the interval in milliseconds
        self.interval = int(1000 / fps)
        
        # Update the display with the actual fps value (rounded to 1 decimal place)
        self.speed_display.config(text=f"{fps:.1f} fps")
    
    def toggle_play(self):
        """Toggle between play and pause."""
        if self.is_playing:
            # Stop playing
            self.is_playing = False
            self.play_button.config(text="Play")
            if hasattr(self, 'play_job'):
                self.root.after_cancel(self.play_job)
        else:
            # Start playing
            self.is_playing = True
            self.play_button.config(text="Pause")
            self.play_next_frame()
    
    def play_next_frame(self):
        """Play the next frame and schedule the next update."""
        if not self.is_playing:
            return
        
        # Calculate the next frame index (with wraparound)
        next_frame = (self.current_frame + 1) % len(self.files)
        
        # Update the visualization
        self.update_frame(next_frame)
        
        # Schedule the next update
        self.play_job = self.root.after(self.interval, self.play_next_frame)
    
    def record_gif(self):
        """Record the animation as a GIF."""
        # Ask for the output file
        output_file = filedialog.asksaveasfilename(
            title="Save GIF Animation",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")]
        )
        
        if not output_file:
            return
        
        # Disable the UI during recording
        self.status_label.config(text="Recording GIF... Please wait.")
        self.record_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.frame_slider.config(state=tk.DISABLED)
        self.speed_slider.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            frames = []
            buffers = []  # Keep references to buffers to prevent them from being garbage collected
            
            # Determine how many frames to capture (use every Nth frame if too many)
            total_frames = len(self.files)
            max_frames = 100  # Maximum number of frames for the GIF
            
            if total_frames > max_frames:
                frame_step = total_frames // max_frames
            else:
                frame_step = 1
            
            # Capture frames
            for i in range(0, total_frames, frame_step):
                # Update the status
                self.status_label.config(text=f"Recording frame {i+1}/{total_frames}...")
                self.root.update()
                
                # Update the visualization
                self.update_frame(i)
                
                # Save the current figure to a buffer
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                
                # Open the image with PIL and append to frames
                img = Image.open(buf)
                img = img.copy()  # Create a copy that doesn't depend on the buffer
                frames.append(img)
                
                # Keep a reference to the buffer
                buffers.append(buf)
            
            # Save the frames as a GIF
            self.status_label.config(text="Saving GIF...")
            self.root.update()
            
            # Save with PIL
            if frames:
                frames[0].save(
                    output_file,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=self.interval,  # Use the current playback speed
                    loop=0  # Loop forever
                )
                
            self.status_label.config(text=f"GIF saved to {output_file}")
            
        except Exception as e:
            self.status_label.config(text=f"Error creating GIF: {str(e)}")
            print(f"Error creating GIF: {e}")
        finally:
            # Re-enable the UI
            self.record_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.NORMAL)
            self.frame_slider.config(state=tk.NORMAL)
            self.speed_slider.config(state=tk.NORMAL)
    
    def run(self):
        """Run the visualizer."""
        self.root.mainloop()

def visualize_stats_folder(stats_dir, output_file=None, show_animation=True):
    """Visualize all brain stats files in the given directory."""
    files = load_stats_files(stats_dir)
    if not files:
        return
    
    if show_animation:
        # Use the interactive visualizer
        visualizer = BrainStatsVisualizer(files)
        visualizer.run()
    else:
        # Just show the first and last epoch
        first_data = load_json_data(files[0])
        last_data = load_json_data(files[-1])
        
        if first_data and last_data:
            fig = plt.figure(figsize=(15, 12))
            
            # First epoch blocks
            ax1 = fig.add_subplot(221, projection='3d')
            visualize_top_blocks(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax1)
            
            # Last epoch blocks
            ax2 = fig.add_subplot(222, projection='3d')
            visualize_top_blocks(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax2)
            
            # First epoch pathways
            ax3 = fig.add_subplot(223, projection='3d')
            visualize_top_pathways(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax3)
            
            # Last epoch pathways
            ax4 = fig.add_subplot(224, projection='3d')
            visualize_top_pathways(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax4)
            
            plt.tight_layout()
            plt.show()

def select_stats_folder():
    """Open a dialog to select the stats folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Start in the brain_stats directory if it exists
    initial_dir = "/home/x/nlp/prj.PhD/prj.SBrain-vis/20250424/brain_stats"
    if not os.path.exists(initial_dir):
        initial_dir = "/home/x/nlp/prj.PhD/prj.SBrain-vis"
    
    # First select the stats directory (which contains multiple experiment folders)
    stats_dir = filedialog.askdirectory(
        title="Select Brain Stats Directory",
        initialdir=initial_dir
    )
    
    if not stats_dir:
        return None
    
    # List all subdirectories in the stats directory
    subdirs = [d for d in os.listdir(stats_dir) if os.path.isdir(os.path.join(stats_dir, d))]
    
    # If there are subdirectories that look like experiment folders (stats_*)
    experiment_dirs = [d for d in subdirs if d.startswith("stats_")]
    
    if experiment_dirs:
        # Create a simple dialog to select an experiment folder
        experiment_selector = tk.Toplevel(root)
        experiment_selector.title("Select Experiment Folder")
        experiment_selector.geometry("400x300")
        
        label = tk.Label(experiment_selector, text="Select an experiment folder:")
        label.pack(pady=10)
        
        # Create a listbox for experiment selection
        listbox = tk.Listbox(experiment_selector, width=50, height=10)
        listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Add experiment folders to the listbox
        for exp_dir in sorted(experiment_dirs):
            listbox.insert(tk.END, exp_dir)
        
        # Variable to store the selected experiment
        selected_experiment = [None]
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_experiment[0] = experiment_dirs[selection[0]]
            experiment_selector.destroy()
        
        # Add a select button
        select_button = tk.Button(experiment_selector, text="Select", command=on_select)
        select_button.pack(pady=10)
        
        # Wait for the user to make a selection
        experiment_selector.wait_window()
        
        if selected_experiment[0]:
            return os.path.join(stats_dir, selected_experiment[0])
    
    # If no experiment was selected or there are no experiment folders,
    # return the stats directory itself
    return stats_dir

def main():
    parser = argparse.ArgumentParser(description='Visualize brain stats from JSON files.')
    parser.add_argument('--stats-dir', type=str, help='Directory containing brain_stats_train_epoch_*.json files')
    parser.add_argument('--output', type=str, help='Output file for animation (requires ffmpeg)')
    parser.add_argument('--no-animation', action='store_true', help='Show only first and last epoch instead of animation')
    
    args = parser.parse_args()
    
    # If stats directory is provided, load files directly
    if args.stats_dir:
        files = load_stats_files(args.stats_dir)
        if not files:
            print(f"No brain_stats_train_epoch_*.json files found in {args.stats_dir}")
            return
        
        if args.no_animation:
            # Just show the first and last epoch
            first_data = load_json_data(files[0])
            last_data = load_json_data(files[-1])
            
            if first_data and last_data:
                fig = plt.figure(figsize=(15, 12))
                
                # First epoch blocks
                ax1 = fig.add_subplot(221, projection='3d')
                visualize_top_blocks(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax1)
                
                # Last epoch blocks
                ax2 = fig.add_subplot(222, projection='3d')
                visualize_top_blocks(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax2)
                
                # First epoch pathways
                ax3 = fig.add_subplot(223, projection='3d')
                visualize_top_pathways(first_data, first_data.get('epoch', extract_epoch_number(files[0])), ax3)
                
                # Last epoch pathways
                ax4 = fig.add_subplot(224, projection='3d')
                visualize_top_pathways(last_data, last_data.get('epoch', extract_epoch_number(files[-1])), ax4)
                
                plt.tight_layout()
                plt.show()
        else:
            # Use the interactive visualizer with pre-loaded files
            visualizer = BrainStatsVisualizer(files)
            visualizer.run()
    else:
        # Start the visualizer without files, user can open them from the menu
        visualizer = BrainStatsVisualizer()
        visualizer.run()

if __name__ == "__main__":
    main()
