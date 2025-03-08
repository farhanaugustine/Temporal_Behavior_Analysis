import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import os
import re
import shutil
import ast
import importlib
import sys
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import pandas as pd
import colorsys

# Assuming BoutAnalyzerGUI.py is in a folder named 'scripts'
from scripts.BoutAnalyzerGUI import BoutAnalyzerGUI # Import BoutAnalyzerGUI from scripts folder


class ToolTip(object):
    """Creates a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self._id1 = self.widget.bind("<Enter>", self.enter)
        self._id2 = self.widget.bind("<Leave>", self.leave)
        self._id3 = self.widget.bind("<ButtonPress>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)  # Delay of 500ms

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # No frame or title bar
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def browse_folder(entry_widget: ttk.Entry):
    """Opens a folder dialog and updates the entry widget."""
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_selected)

def run_script(script_name: str, output_folder: str, class_labels_dict: dict, log_text: scrolledtext.ScrolledText,
               frame_rate: str, video_name: str, min_bout_duration: str, max_gap_duration: str):
    """Runs a selected script by importing it and calling its main function, corrected for argument compatibility."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return
    if not video_name:
        messagebox.showerror("Error", "Please select a video name.")
        return
    if not class_labels_dict:
        messagebox.showerror("Error", "Class labels cannot be empty. Please define them.")
        return
    if not isinstance(class_labels_dict, dict):
        messagebox.showerror("Error", "Class Labels must be a dictionary")
        return
    if not frame_rate:
        messagebox.showerror("Error", "Frame rate cannot be empty.")
        return

    try:
        frame_rate_int = int(frame_rate)
        if frame_rate_int <= 0:
            messagebox.showerror("Error", "Frame rate must be a positive integer.")
            return
    except ValueError:
        messagebox.showerror("Error", "Frame rate must be a valid integer.")
        return

    try:
        # Dynamically import the script module
        script_module = importlib.import_module(f"scripts.{script_name}")

        # Prepare arguments as a dictionary to pass to the script's function
        script_args = {} # Initialize as empty dictionary

        # --- Base arguments for most scripts ---
        base_args = {
            "output_folder": output_folder,
            "class_labels": class_labels_dict,
            "frame_rate": frame_rate_int,
            "video_name": video_name,
        }

        # --- Script-specific argument handling based on Version 1 analysis ---
        if script_name == "behavioral_rhythms_single":
            script_args.update(base_args) # Include base arguments
            script_args["prominence"] = 1.0
        elif script_name == "transfer_entropy":
            script_args.update(base_args) # Include base arguments
            script_args["max_lag"] = 150
            script_args["k"] = 3
        elif script_name == "cross_correlation_single":
            script_args.update(base_args) # Include base arguments
            script_args["max_lag_frames"] = 150
        elif script_name == "general_analysis":
            script_args.update(base_args) # Include base arguments
            script_args["min_bout_duration"] = int(min_bout_duration) if min_bout_duration else 3
            script_args["max_gap_duration"] = int(max_gap_duration) if max_gap_duration else 5
        elif script_name == "total_time_spent":
            script_args.update(base_args) # Include base arguments
        elif script_name == "basic_correlation":
            script_args.update(base_args) # Include base arguments
        elif script_name in ("HMMs", "n_gram_analysis", "sequence_mining"):
            csv_folder = os.path.join(output_folder, "csv_output")  # Subfolder name of csv.
            csv_file_name = f"{video_name}_analysis.csv"
            csv_file_path = os.path.join(csv_folder, csv_file_name)

            if not os.path.exists(csv_file_path):
                messagebox.showerror("Error", f"CSV file not found: {csv_file_path}.  Did you run general_analysis first?")
                return
            script_args["csv_file"] = csv_file_path # Only pass csv_file, NO base arguments for these scripts
            # Note: DO NOT include base_args for HMMs, n_gram, sequence_mining as per Version 1
        elif script_name == "fft_analysis_single":
            script_args.update(base_args) # Include base arguments
        elif script_name == "transition_probability_single":
            script_args.update(base_args) # Include base arguments
        elif script_name == "granger_causality":
            script_args.update(base_args) # Include base arguments
        else:
            messagebox.showerror("Error", f"Script '{script_name}' is not properly configured in GUI.py.")
            return

        # Assume each script has a 'main_analysis' function
        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script {script_name} does not have a 'main_analysis' or 'main' function.")
            return

        # Run the script's analysis function, passing arguments as kwargs
        output_result = analysis_function(**script_args)

        # Format and display output in log
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."
        log_text.insert(tk.END, f"Script {script_name} output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script {script_name} executed successfully.\n")


    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.{script_name}")
        log_text.insert(tk.END, f"Error: Could not find script module scripts.{script_name}\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_multi_rhythm_script(output_folder, log_text):
    """Runs the multi-video rhythm analysis script, corrected for arguments."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    rhythm_folder = os.path.join(output_folder, "rhythm_excel")
    if not os.path.exists(rhythm_folder):
        messagebox.showerror("Error", "The 'rhythm_excel' folder does not exist.  Run the single-video rhythm analysis first.")
        return
    try:
        script_module = importlib.import_module("scripts.behavioral_rhythms_multi")

        script_args = {
            "rhythm_folder": rhythm_folder, # Correct argument name
            "output_folder": output_folder, # output_folder is still needed
            "max_time": 1500  # Added max_time as per Version 1
        }

        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script behavioral_rhythms_multi does not have a 'main_analysis' or 'main' function.")
            return

        output_result = analysis_function(**script_args)
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."

        log_text.insert(tk.END, f"Script behavioral_rhythms_multi output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script behavioral_rhythms_multi executed successfully.\n")

    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.behavioral_rhythms_multi")
        log_text.insert(tk.END, f"Error: Could not find script module behavioral_rhythms_multi\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_multi_fft_script(output_folder, log_text):
    """Runs the multi-video FFT analysis script, corrected for arguments."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    fft_folder = os.path.join(output_folder, "fft_excel")
    if not os.path.exists(fft_folder):
        messagebox.showerror("Error", "The 'fft_excel' folder does not exist.  Run the single-video FFT analysis first.")
        return

    try:
        script_module = importlib.import_module("scripts.fft_analysis_multi")

        script_args = {
            "fft_folder": fft_folder, # Correct argument name
            "output_folder": output_folder, # output_folder is still needed
        }
        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script fft_analysis_multi does not have a 'main_analysis' or 'main' function.")
            return

        output_result = analysis_function(**script_args)
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."


        log_text.insert(tk.END, f"Script fft_analysis_multi output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script fft_analysis_multi executed successfully.\n")

    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.fft_analysis_multi")
        log_text.insert(tk.END, f"Error: Could not find script module fft_analysis_multi\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_cross_correlation_combine_script(output_folder, log_text):
    """Runs the multi-video cross-correlation combination script, corrected for arguments."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    cross_corr_folder = os.path.join(output_folder, "cross_correlation_excel")
    if not os.path.exists(cross_corr_folder):
        messagebox.showerror("Error", "The 'cross_correlation_excel' folder does not exist.  Run the single-video cross-correlation analysis first.")
        return

    try:
        script_module = importlib.import_module("scripts.cross_correlation_combine")

        script_args = {
            "cross_corr_folder": cross_corr_folder, # Correct argument name
            "output_folder": output_folder, # output_folder is still needed
        }
        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script cross_correlation_combine does not have a 'main_analysis' or 'main' function.")
            return

        output_result = analysis_function(**script_args)
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."


        log_text.insert(tk.END, f"Script cross_correlation_combine output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script cross_correlation_combine executed successfully.\n")

    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.cross_correlation_combine")
        log_text.insert(tk.END, f"Error: Could not find script module cross_correlation_combine\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_cross_correlation_stats_script(output_folder, log_text):
    """Runs the cross-correlation statistics script, corrected for arguments."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    cross_corr_folder = os.path.join(output_folder, "cross_correlation_excel")
    if not os.path.exists(cross_corr_folder):
        messagebox.showerror("Error", "The 'cross_correlation_excel' folder does not exist.  Run the single-video cross-correlation analysis first.")
        return

    try:
        script_module = importlib.import_module("scripts.cross_correlation_stats")
        script_args = {
            "cross_corr_folder": cross_corr_folder, # Correct argument name
            "output_folder": output_folder, # output_folder is still needed
            "metric": "peak_correlation", #Include the metric - as per Version 1
        }
        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script cross_correlation_stats does not have a 'main_analysis' or 'main' function.")
            return

        output_result = analysis_function(**script_args)
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."


        log_text.insert(tk.END, f"Script cross_correlation_stats output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script cross_correlation_stats executed successfully.\n")

    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.cross_correlation_stats")
        log_text.insert(tk.END, f"Error: Could not find script module cross_correlation_stats\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_multi_transition_prob_script(output_folder, log_text):
    """Runs the multi-video transition probability analysis script, corrected for arguments."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    transition_folder = os.path.join(output_folder, "transition_probabilities_excel")
    if not os.path.exists(transition_folder):
        messagebox.showerror("Error", "The 'transition_probabilities_excel' folder does not exist. Run the single-video transition probability analysis first.")
        return

    try:
        script_module = importlib.import_module("scripts.transition_probability_multi")

        script_args = {
            "transition_folder": transition_folder, # Correct argument name
            "output_folder": output_folder, # output_folder is still needed
        }
        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script transition_probability_multi does not have a 'main_analysis' or 'main' function.")
            return

        output_result = analysis_function(**script_args)
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."

        log_text.insert(tk.END, f"Script transition_probability_multi output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script transition_probability_multi executed successfully.\n")

    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.transition_probability_multi")
        log_text.insert(tk.END, f"Error: Could not find script module transition_probability_multi\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_multi_script(script_name: str, input_folder: str, output_folder: str, log_text: scrolledtext.ScrolledText):
    """Runs a selected multi-video script by importing it and calling its main function, corrected for arguments."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    if not input_folder:
        messagebox.showerror("Error", "Please select an input folder for multi-video analysis.")
        return

    try:
        # Check if the input folder contains CSV files named *_analysis.csv
        csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        if not csv_files:
            messagebox.showerror("Error", "The input folder does not contain any CSV files ending with '.csv'.")
            return

        # Dynamically import the script module
        script_module = importlib.import_module(f"scripts.{script_name}")

        script_args = {
            "input_folder": input_folder, # User defined input folder
            "output_folder": output_folder, # output_folder is still needed
        }

        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script {script_name} does not have a 'main_analysis' or 'main' function.")
            return

        output_result = analysis_function(**script_args)
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."

        # Display output in the log
        log_text.insert(tk.END, f"Script {script_name} output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script {script_name} executed successfully.\n")

    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.{script_name}")
        log_text.insert(tk.END, f"Error: Could not find script module scripts.{script_name}\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")
        
def organize_txt_files(source_folder: str, log_text: scrolledtext.ScrolledText):
    """Organizes TXT files into subfolders based on video name."""
    if not os.path.exists(source_folder):
        messagebox.showerror("Error", f"Source folder '{source_folder}' does not exist.")
        return

    video_files = {}
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(source_folder, filename)
            match = re.match(r"^(.*?)_\d+\.txt$", filename)
            if match:
                video_name = match.group(1).rstrip("_")  # Remove trailing underscores
                if video_name not in video_files:
                    video_files[video_name] = []
                video_files[video_name].append(filepath)
            else:
                log_text.insert(tk.END, f"Warning: Could not extract video name from '{filename}'. Skipping.\n")

    for video_name, files in video_files.items():
        target_folder = os.path.join(source_folder, video_name)
        os.makedirs(target_folder, exist_ok=True)
        for file_path in files:
            try:
                shutil.move(file_path, target_folder)
                log_text.insert(tk.END, f"Moved '{os.path.basename(file_path)}' to '{target_folder}'\n")
            except Exception as e:
                log_text.insert(tk.END, f"Error moving '{file_path}': {e}\n")
                messagebox.showerror("Error", f"Error moving '{file_path}': {e}")

    messagebox.showinfo("Info", "TXT files organized successfully!")

def create_gui() -> tk.Tk:
    """Creates the main application GUI."""
    root = tk.Tk()
    root.title("Behavioral Analysis Tool")

    # --- Logo Frame ---  (Dedicated Frame for Logo, placed at the top-left)
    logo_frame = ttk.Frame(root)
    logo_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10) # Placed at top-left

    try:
        # Conditional logo path:
        if hasattr(sys, '_MEIPASS'): # Check if running as EXE
            logo_path = os.path.join(sys._MEIPASS, "Behavior.png") # Path for EXE
        else:
            logo_path = "Behavior.png" # Path for script execution

        print(f"Current working directory: {os.getcwd()}") # Debugging
        print(f"Attempting to open logo from path: {logo_path}") # Debugging

        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((200, 200), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = tk.Label(logo_frame, image=logo_photo)
        logo_label.image = logo_photo
        logo_label.pack()
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        print("Logo image file not found. Using placeholder.")
        logo_label = tk.Label(logo_frame, text="Logo Placeholder")
        logo_label.pack()
    except Exception as e:
        print(f"An unexpected error occurred while loading logo: {e}")
        logo_label = tk.Label(logo_frame, text="Logo Placeholder")
        logo_label.pack()

    # --- Input Section ---
    input_frame = ttk.LabelFrame(root, text="Input")
    input_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew") # Shifted to column 1 (next to logo)

    # YOLO Output Folder
    ttk.Label(input_frame, text="YOLO Output Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    output_folder_entry = ttk.Entry(input_frame, width=40)
    output_folder_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    ttk.Button(input_frame, text="Browse", command=lambda: browse_folder(output_folder_entry)).grid(row=0, column=2, padx=5, pady=2)

    # Class Labels
    ttk.Label(input_frame, text="Class Labels:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    class_labels_entry = ttk.Entry(input_frame, width=40)
    class_labels_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    class_labels_entry.insert(0, str(class_labels_dict))
    ttk.Button(input_frame, text="Edit", command=lambda: edit_class_labels(class_labels_dict, class_labels_entry)).grid(row=1, column=2, padx=5, pady=2)

    # Video Name
    ttk.Label(input_frame, text="Video Name:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    video_name_entry = ttk.Entry(input_frame, width=40)
    video_name_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

     # Input Folder for Multi-Video Analysis
    ttk.Label(input_frame, text="Multi-Video Input Folder:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
    input_folder_entry = ttk.Entry(input_frame, width=40)
    input_folder_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
    ttk.Button(input_frame, text="Browse", command=lambda: browse_folder(input_folder_entry)).grid(row=3, column=2, padx=5, pady=2)

    # --- Parameters Section ---
    parameter_frame = ttk.LabelFrame(root, text="Parameters")
    parameter_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew") # Shifted to column 2 (next to input)

    # Frame Rate
    ttk.Label(parameter_frame, text="Frame Rate:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    frame_rate_entry = ttk.Entry(parameter_frame, width=10)
    frame_rate_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
    frame_rate_entry.insert(0, "30")

    # Min Bout Duration
    ttk.Label(parameter_frame, text="Min Bout Duration (frames):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    min_bout_duration_entry = ttk.Entry(parameter_frame, width=10)
    min_bout_duration_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
    min_bout_duration_entry.insert(0, "3")

    # Max Gap Duration
    ttk.Label(parameter_frame, text="Max Gap Duration (frames):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    max_gap_duration_entry = ttk.Entry(parameter_frame, width=10)
    max_gap_duration_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
    max_gap_duration_entry.insert(0, "5")

    # --- Organize Button ---
    organize_button = ttk.Button(root, text="Organize TXT Files",
                                  command=lambda: organize_txt_files(output_folder_entry.get(), log_text))
    organize_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew") # Placed below Logo

    # --- Analysis Scripts Section ---
    # Use a main frame, and subframes for categories.
    analysis_frame = ttk.Frame(root)  # Main frame, NOT a LabelFrame
    analysis_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew") # Below Input/Parameter, span 3 cols

    # --- General Analysis Frame ---
    general_frame = ttk.LabelFrame(analysis_frame, text="General Analysis")
    general_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")  # Back to individual frames

    instruction_label = ttk.Label(general_frame, text="Please run 'general_analysis' first.", font=("Arial", 10, "bold"))
    instruction_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    btn_general = ttk.Button(general_frame, text="general_analysis",
                            command=lambda: run_script("general_analysis", output_folder_entry.get(),
                                                        class_labels_dict, log_text,
                                                        frame_rate_entry.get(), video_name_entry.get(),
                                                        min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_general.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    style = ttk.Style()
    style.configure("Red.TButton", foreground="black", background="red")
    btn_general.configure(style="Red.TButton")
    ToolTip(btn_general, "Performs general analysis... Must be run first.")

    btn_total_time = ttk.Button(general_frame, text="total_time_spent",
                                command=lambda: run_script("total_time_spent", output_folder_entry.get(),
                                                            class_labels_dict, log_text,
                                                            frame_rate_entry.get(), video_name_entry.get(),
                                                            min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_total_time.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_total_time, "Calculates the total time spent...")

    # --- Rhythm Analysis Frame ---
    rhythm_frame = ttk.LabelFrame(analysis_frame, text="Rhythm Analysis")
    rhythm_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew") # Next column

    btn_rhythms_single = ttk.Button(rhythm_frame, text="behavioral_rhythms_single",
                                    command=lambda: run_script("behavioral_rhythms_single", output_folder_entry.get(),
                                                                class_labels_dict, log_text,
                                                                frame_rate_entry.get(), video_name_entry.get(),
                                                                min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_rhythms_single.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_rhythms_single, "Detects rhythmic patterns in a single video.")

    btn_rhythms_multi = ttk.Button(rhythm_frame, text="behavioral_rhythms_multi",
                                   command=lambda: run_multi_rhythm_script(output_folder_entry.get(), log_text))
    btn_rhythms_multi.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_rhythms_multi, "Combines rhythm analysis results from multiple videos.")

    # --- Frequency Analysis (FFT) Frame ---
    fft_frame = ttk.LabelFrame(analysis_frame, text="Frequency Analysis (FFT)")
    fft_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")  # Next column

    btn_fft_single = ttk.Button(fft_frame, text="fft_analysis_single",
                                command=lambda: run_script("fft_analysis_single", output_folder_entry.get(),
                                                            class_labels_dict, log_text,
                                                            frame_rate_entry.get(), video_name_entry.get(),
                                                            min_bout_duration_entry.get(),        max_gap_duration_entry.get()))
    btn_fft_single.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_fft_single, "Performs Fast Fourier Transform (FFT) analysis on a single video.") # Tooltip updated

    btn_fft_multi = ttk.Button(fft_frame, text="fft_analysis_multi",
                               command=lambda: run_multi_fft_script(output_folder_entry.get(), log_text))
    btn_fft_multi.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_fft_multi, "Combines FFT analysis results from multiple videos.") # Tooltip updated


    # --- Correlation Analysis Frame ---
    correlation_frame = ttk.LabelFrame(analysis_frame, text="Correlation Analysis") # Frame title simplified
    correlation_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew") # New row, below General

    btn_basic_correlation_single = ttk.Button(correlation_frame, text="basic_correlation_single", # Button name updated
                                    command=lambda: run_script("basic_correlation", output_folder_entry.get(),
                                                                class_labels_dict, log_text,
                                                                frame_rate_entry.get(), video_name_entry.get(),
                                                                min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_basic_correlation_single.grid(row=0, column=0, padx=5, pady=5, sticky="ew") # Button name updated
    ToolTip(btn_basic_correlation_single, "Calculates Spearman rank correlations for a single video.") # Tooltip updated

    btn_cross_correlation_single = ttk.Button(correlation_frame, text="cross_correlation_single",
                                            command=lambda: run_script("cross_correlation_single", output_folder_entry.get(),
                                                                        class_labels_dict, log_text,
                                                                        frame_rate_entry.get(), video_name_entry.get(),
                                                                        min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_cross_correlation_single.grid(row=1, column=0, padx=5, pady=5, sticky="ew") # Stays in the same frame
    ToolTip(btn_cross_correlation_single, "Calculates time-lagged cross-correlations for a single video.") # Tooltip updated

    btn_cross_correlation_combine = ttk.Button(correlation_frame, text="cross_correlation_combine",
                                               command=lambda: run_cross_correlation_combine_script(output_folder_entry.get(), log_text))
    btn_cross_correlation_combine.grid(row=2, column=0, padx=5, pady=5, sticky="ew") # Stays in the same frame
    ToolTip(btn_cross_correlation_combine, "Combines cross-correlation results from multiple videos.") # Tooltip updated

    btn_cross_correlation_stats = ttk.Button(correlation_frame, text="cross_correlation_stats",
                                             command=lambda: run_cross_correlation_stats_script(output_folder_entry.get(), log_text))
    btn_cross_correlation_stats.grid(row=3, column=0, padx=5, pady=5, sticky="ew") # Stays in the same frame
    ToolTip(btn_cross_correlation_stats, "Performs statistical analysis on cross-correlation results from multiple videos.") # Tooltip updated

    # --- Causality Analysis Frame ---
    causality_frame = ttk.LabelFrame(analysis_frame, text="Causality Analysis")
    causality_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew") # Next column, same row as Correlation

    btn_granger = ttk.Button(causality_frame, text="granger_causality",
                            command=lambda: run_script("granger_causality", output_folder_entry.get(),
                                                        class_labels_dict, log_text,
                                                        frame_rate_entry.get(), video_name_entry.get(),
                                                        min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_granger.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_granger, "Performs Granger causality analysis on a single video.") # Tooltip updated

    btn_transfer_entropy = ttk.Button(causality_frame, text="transfer_entropy",
                                    command=lambda: run_script("transfer_entropy", output_folder_entry.get(),
                                                                class_labels_dict, log_text,
                                                                frame_rate_entry.get(), video_name_entry.get(),
                                                                min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_transfer_entropy.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_transfer_entropy, "Calculates transfer entropy for a single video.") # Tooltip updated

    # --- Transition Analysis Frame ---
    transition_frame = ttk.LabelFrame(analysis_frame, text="Transition Analysis")
    transition_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")  # Next column

    btn_transition_single = ttk.Button(transition_frame, text="transition_probability_single",
                                        command=lambda: run_script("transition_probability_single", output_folder_entry.get(),
                                                                    class_labels_dict, log_text,
                                                                    frame_rate_entry.get(), video_name_entry.get(),
                                                                    min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_transition_single.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_transition_single, "Calculates transition probabilities for a single video.") # Tooltip updated

    btn_transition_multi = ttk.Button(transition_frame, text="transition_probability_multi",
                                      command=lambda: run_multi_transition_prob_script(output_folder_entry.get(), log_text))
    btn_transition_multi.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_transition_multi, "Combines transition probability results from multiple videos.") # Tooltip updated

    # --- Sequence Analysis Frame ---
    sequence_frame = ttk.LabelFrame(analysis_frame, text="Sequence Analysis") # Frame title simplified
    sequence_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

    btn_hmms_single = ttk.Button(sequence_frame, text="HMMs_single", # Button name updated
                            command=lambda: run_script("HMMs", output_folder_entry.get(),
                                                        class_labels_dict, log_text,
                                                        frame_rate_entry.get(), video_name_entry.get(),
                                                        min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_hmms_single.grid(row=0, column=0, padx=5, pady=5, sticky="ew") # Button name updated
    ToolTip(btn_hmms_single, "Runs Hidden Markov Models on a single video.") # Tooltip updated

    btn_ngram_single = ttk.Button(sequence_frame, text="n_gram_analysis_single", # Button name updated
                               command=lambda: run_script("n_gram_analysis", output_folder_entry.get(),
                                                            class_labels_dict, log_text,
                                                            frame_rate_entry.get(), video_name_entry.get(),
                                                            min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_ngram_single.grid(row=0, column=1, padx=5, pady=5, sticky="ew") # Button name updated
    ToolTip(btn_ngram_single, "Runs n-gram analysis on a single video.") # Tooltip updated

    btn_sequence_mining_single = ttk.Button(sequence_frame, text="sequence_mining_single", # Button name updated
                                    command=lambda: run_script("sequence_mining", output_folder_entry.get(),
                                                                class_labels_dict, log_text,
                                                                frame_rate_entry.get(), video_name_entry.get(),
                                                                min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_sequence_mining_single.grid(row=0, column=2, padx=5, pady=5, sticky="ew") # Button name updated
    ToolTip(btn_sequence_mining_single, "Runs sequence mining on a single video.") # Tooltip updated

    btn_hmms_multi = ttk.Button(sequence_frame, text="HMMs_multi", # Button moved and name updated
                            command=lambda: run_multi_script("HMMs_multi", input_folder_entry.get(), output_folder_entry.get(), log_text))
    btn_hmms_multi.grid(row=1, column=0, padx=5, pady=5, sticky="ew") # Button moved and name updated
    ToolTip(btn_hmms_multi, "Runs Hidden Markov Models on multiple videos.") # Tooltip updated

    btn_ngram_multi = ttk.Button(sequence_frame, text="n_gram_analysis_multi", # Button moved and name updated
                               command=lambda: run_multi_script("n_gram_analysis_multi", input_folder_entry.get(), output_folder_entry.get(), log_text))
    btn_ngram_multi.grid(row=1, column=1, padx=5, pady=5, sticky="ew") # Button moved and name updated
    ToolTip(btn_ngram_multi, "Runs n-gram analysis on multiple videos.") # Tooltip updated

    btn_sequence_mining_multi = ttk.Button(sequence_frame, text="sequence_mining_multi", # Button moved and name updated
                                    command=lambda: run_multi_script("sequence_mining_multi", input_folder_entry.get(), output_folder_entry.get(), log_text))
    btn_sequence_mining_multi.grid(row=1, column=2, padx=5, pady=5, sticky="ew") # Button moved and name updated
    ToolTip(btn_sequence_mining_multi, "Runs sequence mining on multiple videos.") # Tooltip updated

    # --- Viewer Frame ---
    viewer_frame = ttk.LabelFrame(analysis_frame, text="Viewers")
    viewer_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="nsew") # New row below sequence

    btn_bout_viewer = ttk.Button(viewer_frame, text="Bout Viewer", command=launch_bout_viewer_wrapper) # Modified command to call wrapper
    btn_bout_viewer.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_bout_viewer, "Launches the Bout Viewer to review video bouts.")

    # --- Bout Viewer Button ---
    btn_bout_viewer = ttk.Button(viewer_frame, text="Bout Viewer", command=lambda: launch_bout_viewer(root)) # Use lambda to pass root
    btn_bout_viewer.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_bout_viewer, "Launches the Bout Viewer to review video bouts.")

    # --- Data Tools Frame --- 
    data_tools_frame = ttk.LabelFrame(analysis_frame, text="Data Preparation Tools") # Frame title RENAMED
    data_tools_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew") # New row below Viewer Frame

    btn_data_prep = ttk.Button(data_tools_frame, text="Data Preparation Tool", command=launch_label_correction_tool) # Button for Data Preparation Tool - RENAMED
    btn_data_prep.grid(row=0, column=0, padx=5, pady=5, sticky="ew") # Place in Data Tools Frame
    ToolTip(btn_data_prep, "Launches the Data Preparation Tool to create and edit YOLO training datasets.") # Tooltip for Data Preparation Tool - UPDATED

    # --- Log Display ---
    log_frame = ttk.LabelFrame(root, text="Log")
    log_frame.grid(row=3, column=1, columnspan=2, padx=10, pady=10, sticky="nsew") # Shifted to column 1, next to Viewer

    log_text = scrolledtext.ScrolledText(log_frame, width=80, height=15, state='normal')
    log_text.grid(row=0, column=0, sticky="nsew")

    # --- Author and Affiliation ---
    author_label = tk.Label(root, text="Author: Farhan Augustine")
    author_label.grid(row=8, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 0)) # Adjusted row

    affiliation_label = tk.Label(root, text="Affiliation: University of Maryland Baltimore County")
    affiliation_label.grid(row=9, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 0)) # Adjusted row

    affiliation_label = tk.Label(root, text="Built: Year 2025")
    affiliation_label.grid(row=10, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 10))

    # --- Making the GUI Resizable ---
    root.grid_rowconfigure(2, weight=1)  # Analysis frame row expands
    root.grid_rowconfigure(3, weight=1)  # Log frame row expands
    root.grid_columnconfigure(0, weight=0) # Logo column, set weight to 0 or adjust as needed
    root.grid_columnconfigure(1, weight=1) # Input frame column, expand
    root.grid_columnconfigure(2, weight=1) # Parameter frame column expand

    input_frame.columnconfigure(1, weight=1)  # Input frame entries expand
    parameter_frame.columnconfigure(1, weight=1) # Parameter frame entries expand
    analysis_frame.columnconfigure(0, weight=1)  # Analysis buttons expand
    analysis_frame.columnconfigure(1, weight=1)
    analysis_frame.columnconfigure(2, weight=1)
    log_frame.grid_rowconfigure(0, weight=1)
    log_frame.grid_columnconfigure(0, weight=1)

    # Add tooltips to input elements
    ToolTip(output_folder_entry, "Select the folder containing YOLO output TXT files.")
    ToolTip(class_labels_entry, "Enter or edit the class labels as a Python dictionary.")
    ToolTip(video_name_entry, "Enter the name of the video being analyzed (without the extension).")
    ToolTip(frame_rate_entry, "Enter the frame rate of the video (e.g., 30).")
    ToolTip(min_bout_duration_entry, "Enter the minimum duration of a bout in frames.")
    ToolTip(max_gap_duration_entry, "Enter the maximum gap allowed between frames within a bout.")
    ToolTip(organize_button, "Organize TXT files into subfolders named after the video.")
    ToolTip(input_folder_entry, "Select the folder containing multiple CSV files for Multi-Video Analysis.")

    return root

def run_multi_script(script_name: str, input_folder: str, output_folder: str, log_text: scrolledtext.ScrolledText):
    """Runs a selected multi-video script by importing it and calling its main function."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    if not input_folder:
        messagebox.showerror("Error", "Please select an input folder for multi-video analysis.")
        return

    try:
        # Check if the input folder contains CSV files named *_analysis.csv
        csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        if not csv_files:
            messagebox.showerror("Error", "The input folder does not contain any CSV files ending with '.csv'.")
            return

        # Dynamically import the script module
        script_module = importlib.import_module(f"scripts.{script_name}")

        script_args = {
            "input_folder": input_folder, #This is the user defined input folder
            "output_folder": output_folder,
        }

        if hasattr(script_module, 'main_analysis'):
            analysis_function = script_module.main_analysis
        elif hasattr(script_module, 'main'): # Fallback to 'main' if 'main_analysis' not found
            analysis_function = script_module.main
        else:
            messagebox.showerror("Error", f"Script {script_name} does not have a 'main_analysis' or 'main' function.")
            return

        output_result = analysis_function(**script_args)
        script_output_str = str(output_result) if output_result is not None else "No direct output from script function."

        # Display output in the log
        log_text.insert(tk.END, f"Script {script_name} output:\n")
        log_text.insert(tk.END, script_output_str + "\n")
        log_text.insert(tk.END, f"Script {script_name} executed successfully.\n")

    except ModuleNotFoundError:
        messagebox.showerror("Error", f"Could not find script module: scripts.{script_name}")
        log_text.insert(tk.END, f"Error: Could not find script module scripts.{script_name}\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def edit_class_labels(class_labels_dict: dict, class_labels_entry: ttk.Entry):
    """Opens a dialog to edit class labels and updates the entry widget."""
    current_labels_str = str(class_labels_dict)
    new_labels_str = simpledialog.askstring("Edit Class Labels", "Enter class labels as a dictionary (e.g., {0: 'Label1', 1: 'Label2'}):", initialvalue=current_labels_str)
    if new_labels_str:
        try:
            new_labels_dict = ast.literal_eval(new_labels_str)  # Use ast.literal_eval()
            if isinstance(new_labels_dict, dict):
                class_labels_dict.clear()
                class_labels_dict.update(new_labels_dict)
                class_labels_entry.delete(0, tk.END)
                class_labels_entry.insert(0, str(class_labels_dict))
            else:
                messagebox.showerror("Error", "Invalid input. Please enter a valid dictionary.")
        except (ValueError, SyntaxError) as e:  # Catch SyntaxError too
            messagebox.showerror("Error", f"Invalid input: {e}")

def launch_bout_viewer_wrapper(): # Wrapper function to get root from create_gui scope
    """Launches the Bout Viewer GUI in a new window."""
    global root # Assuming 'root' is defined in the create_gui scope and you want to access it
    launch_bout_viewer(root)

def launch_bout_viewer(main_root): # Pass main root as argument
    """Launches the Bout Viewer GUI in a new window."""
    bout_viewer_app = BoutAnalyzerGUI(master=main_root) 
    
class LabelCorrectionApp(object): # Modified to inherit from 'object', not tk.Tk
    def __init__(self, root): # Now takes 'root' as argument, which will be a Toplevel window
        self.root = root
        root.title("YOLO Data Preparation Tool") # More descriptive title

        # --- Instance Variables ---
        self.image_folder = ""
        self.label_folder = ""
        self.image_files = []
        self.current_image_index = 0
        self.image = None
        self.photo = None
        self.labels = []
        self.selected_label_index = -1
        self.class_names = []
        self.class_colors = {}
        self.unsaved_changes = False
        self.drawing_mode = False  # Rectangle drawing mode flag
        self.start_x = None
        self.start_y = None
        self.current_rect_id = None

        # --- GUI Elements ---
        # Top Frame for folder selection
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        self.img_folder_button = tk.Button(
            top_frame, text="Select Image Folder", command=self.select_image_folder
        )
        self.img_folder_button.pack(side=tk.LEFT, padx=5)

        self.label_folder_button = tk.Button(
            top_frame, text="Select Label Folder", command=self.select_label_folder
        )
        self.label_folder_button.pack(side=tk.LEFT, padx=5)

        self.class_name_button = tk.Button(
            top_frame, text="Select Class Names File", command=self.load_class_names
        )
        self.class_name_button.pack(side=tk.LEFT, padx=5)

        # Middle Frame for image display
        middle_frame = tk.Frame(root)
        middle_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(middle_frame)  # Use a Canvas instead of a Label
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.select_label_or_start_rect) # Modified binding
        self.canvas.bind("<B1-Motion>", self.draw_rectangle) # Binding for drawing
        self.canvas.bind("<ButtonRelease-1>", self.finish_rectangle) # Binding for drawing

        # Bottom Frame for navigation and editing
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(pady=10)

        self.prev_button = tk.Button(
            bottom_frame, text="Previous", command=self.previous_image
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(
            bottom_frame, text="Next", command=self.next_image
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.label_list = ttk.Treeview(
            bottom_frame,
            columns=("Class", "x_center", "y_center", "width", "height"),
            show="headings",
        )
        self.label_list.heading("Class", text="Class")
        self.label_list.heading("x_center", text="x_center")
        self.label_list.heading("y_center", text="y_center")
        self.label_list.heading("width", text="width")
        self.label_list.heading("height", text="height")
        self.label_list.pack(side=tk.LEFT, padx=5)
        self.label_list.bind("<<TreeviewSelect>>", self.on_label_select)

        # Edit Frame (right side of bottom frame)
        edit_frame = tk.Frame(bottom_frame)
        edit_frame.pack(side=tk.RIGHT, padx=5)

        tk.Label(edit_frame, text="Class:").grid(row=0, column=0, sticky=tk.W)
        self.class_entry = ttk.Combobox(edit_frame)
        self.class_entry.grid(row=0, column=1, sticky=tk.EW)

        tk.Label(edit_frame, text="x_center:").grid(row=1, column=0, sticky=tk.W)
        self.x_center_entry = tk.Entry(edit_frame)
        self.x_center_entry.grid(row=1, column=1, sticky=tk.EW)

        tk.Label(edit_frame, text="y_center:").grid(row=2, column=0, sticky=tk.W)
        self.y_center_entry = tk.Entry(edit_frame)
        self.y_center_entry.grid(row=2, column=1, sticky=tk.EW)

        tk.Label(edit_frame, text="width:").grid(row=3, column=0, sticky=tk.W)
        self.width_entry = tk.Entry(edit_frame)
        self.width_entry.grid(row=3, column=1, sticky=tk.EW)

        tk.Label(edit_frame, text="height:").grid(row=4, column=0, sticky=tk.W)
        self.height_entry = tk.Entry(edit_frame)
        self.height_entry.grid(row=4, column=1, sticky=tk.EW)

        self.update_button = tk.Button(
            edit_frame, text="Update Label", command=self.update_label
        )
        self.update_button.grid(row=5, column=0, columnspan=2, pady=5)

        self.delete_button = tk.Button(
            edit_frame, text="Delete Label", command=self.delete_label
        )
        self.delete_button.grid(row=6, column=0, columnspan=2, pady=5)

        self.add_button = tk.Button(
            edit_frame, text="Add Label", command=self.add_label
        )
        self.add_button.grid(row=7, column=0, columnspan=2, pady=5)

        self.save_button = tk.Button(
            edit_frame, text="Save Labels", command=self.save_labels
        )
        self.save_button.grid(row=8, column=0, columnspan=2, pady=5)

        self.draw_rect_button = tk.Button( # New button to toggle drawing mode
            edit_frame, text="Draw Rectangle", command=self.toggle_drawing_mode
        )
        self.draw_rect_button.grid(row=9, column=0, columnspan=2, pady=5)

        # Status Bar
        self.status_bar = tk.Label(
            root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Key bindings for navigation
        root.bind("<Left>", lambda event: self.previous_image())
        root.bind("<Right>", lambda event: self.next_image())
        root.bind("<Return>", lambda event: self.update_label())
        root.bind("<Delete>", lambda event: self.delete_label())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.draw_rect_button.config(relief=tk.SUNKEN) # Visual feedback button is pressed
            self.status_bar.config(text="Rectangle drawing mode ON. Click and drag to draw a bounding box.")
        else:
            self.draw_rect_button.config(relief=tk.RAISED)
            self.status_bar.config(text="Rectangle drawing mode OFF.")

    def select_label_or_start_rect(self, event):
        if self.drawing_mode:
            self.start_rectangle(event)
        else:
            self.select_label(event)

    def start_rectangle(self, event):
        # Start drawing rectangle
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline='red', width=2, tags="temp_rect"
        )

    def draw_rectangle(self, event):
        # Update rectangle while dragging
        if self.drawing_mode and self.current_rect_id:
            self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, event.x, event.y)

    def finish_rectangle(self, event):
        # Finish drawing and add label
        if self.drawing_mode and self.current_rect_id:
            self.drawing_mode = False
            self.draw_rect_button.config(relief=tk.RAISED) # Reset button visual
            x1, y1, x2, y2 = self.canvas.coords(self.current_rect_id)
            self.canvas.delete(self.current_rect_id) # Remove temp rect
            self.current_rect_id = None

            # Ensure x1 < x2 and y1 < y2
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)

            # Convert canvas coords to image coords
            image_x1 = (x_min - (self.canvas_width - self.displayed_width) // 2) / self.displayed_width
            image_y1 = (y_min - (self.canvas_height - self.displayed_height) // 2) / self.displayed_height
            image_x2 = (x_max - (self.canvas_width - self.displayed_width) // 2) / self.displayed_width
            image_y2 = (y_max - (self.canvas_height - self.displayed_height) // 2) / self.displayed_height

            # Calculate normalized YOLO format
            x_center = (image_x1 + image_x2) / 2
            y_center = (image_y1 + image_y2) / 2
            width = abs(image_x2 - image_x1)
            height = abs(image_y2 - image_y1)

            # Set values in entry fields for user to adjust class etc.
            self.x_center_entry.delete(0, tk.END)
            self.y_center_entry.delete(0, tk.END)
            self.width_entry.delete(0, tk.END)
            self.height_entry.delete(0, tk.END)
            self.x_center_entry.insert(0, f"{x_center:.6f}")
            self.y_center_entry.insert(0, f"{y_center:.6f}")
            self.width_entry.insert(0, f"{width:.6f}")
            self.height_entry.insert(0, f"{height:.6f}")

            self.status_bar.config(text="Rectangle drawn. Please select class and click 'Add Label' to finalize.")

    def select_image_folder(self):
        self.image_folder = filedialog.askdirectory(parent=self.root) # Parent is now self.root
        if self.image_folder:
            self.status_bar.config(text=f"Image folder: {self.image_folder}")
            self.load_image_files()
            self.load_and_display_image()  # Use the combined function

    def select_label_folder(self):
        self.label_folder = filedialog.askdirectory(parent=self.root) # Parent is now self.root
        if self.label_folder:
            self.status_bar.config(text=f"Label folder: {self.label_folder}")
            self.load_and_display_image()  # Reload with new label folder

    def load_class_names(self):
        class_names_path = filedialog.askopenfilename(parent=self.root, filetypes=[("Text Files", "*.txt")]) # Parent is now self.root
        if class_names_path:
            try:
                with open(class_names_path, "r") as f:
                    self.class_names = [line.strip() for line in f]
                    self.class_entry["values"] = self.class_names
                    self.generate_class_colors()
                self.status_bar.config(
                    text=f"Class names loaded from {class_names_path}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Error loading class names: {e}", parent=self.root) # Parent is now self.root

    def generate_class_colors(self):
        num_classes = len(self.class_names)
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            rgb = tuple(int(c * 255) for c in rgb)
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            self.class_colors[self.class_names[i]] = hex_color

    def load_image_files(self):
        if not self.image_folder:
            return
        self.image_files = sorted(
            [
                f
                for f in os.listdir(self.image_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]
        )
        self.current_image_index = 0
        self.unsaved_changes = False

    # --- COMBINED FUNCTION: load_and_display_image ---
    def load_and_display_image(self):
        """Loads the image, loads labels, and draws bounding boxes (all in one)."""
        if not self.image_files or not self.image_folder:
            return

        if self.unsaved_changes:
            if messagebox.askyesno(
                "Unsaved Changes", "You have unsaved changes. Save before proceeding?", parent=self.root # Parent is now self.root
            ):
                self.save_labels()
            self.unsaved_changes = False

        image_path = os.path.join(
            self.image_folder, self.image_files[self.current_image_index]
        )
        try:
            self.image = Image.open(image_path)
            self.display_image()  # Display image on canvas
            self.load_labels()  # Load labels *first*
            self.draw_bounding_boxes()  # *Then* draw boxes
            self.status_bar.config(
                text=f"Loaded: {self.image_files[self.current_image_index]}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}", parent=self.root) # Parent is now self.root
            self.image = None
            self.canvas.delete("all")  # Clear canvas

    def display_image(self):
        """Displays the image on the canvas, resized to fit."""
        if self.image is None:
            return

        # Resize image to fit canvas while maintaining aspect ratio
        width, height = self.image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate scale factor
        scale = min(canvas_width / width, canvas_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = self.image.resize((new_width, new_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)

        # Clear previous image and display new one
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.photo
        )
        self.canvas.image = self.photo  # Keep reference
        self.displayed_width = new_width  # Store for click scaling
        self.displayed_height = new_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def load_labels(self):
        self.labels = []
        self.label_list.delete(*self.label_list.get_children())
        if not self.label_folder:
            return
        label_filename = (
            os.path.splitext(self.image_files[self.current_image_index])[0] + ".txt"
        )
        label_path = os.path.join(self.label_folder, label_filename)
        if not os.path.exists(label_path):
            return
        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = parts
                    try:
                        label_data = {
                            "class_id": int(class_id),
                            "x_center": float(x_center),
                            "y_center": float(y_center),
                            "width": float(width),
                            "height": float(height),
                        }
                        self.labels.append(label_data)
                        class_name = (
                            self.class_names[label_data["class_id"]]
                            if self.class_names
                            and label_data["class_id"] < len(self.class_names)
                            else str(label_data["class_id"])
                        )
                        self.label_list.insert(
                            "",
                            "end",
                            values=(class_name, x_center, y_center, width, height),
                        )
                    except ValueError:
                        continue
        except Exception as e:
            messagebox.showerror("Error", f"Error loading labels: {e}", parent=self.root) # Parent is now self.root
        # NO draw_bounding_boxes() here!  It's now part of load_and_display_image()

    def draw_bounding_boxes(self):
        if self.image is None: return
        self.canvas.delete("bbox")
        self.canvas.delete("bbox_text")
        width, height = self.image.size
        scale_x = self.displayed_width / width  # Correct scaling
        scale_y = self.displayed_height / height

        offset_x = (self.canvas_width - self.displayed_width) // 2  # Center
        offset_y = (self.canvas_height - self.displayed_height) // 2

        for i, label in enumerate(self.labels):  # Use enumerate for indexing
            x_center, y_center, bb_width, bb_height = [
                label[key] for key in ("x_center", "y_center", "width", "height")
            ]  # Unpack directly

            # Scale coordinates
            x_center *= self.displayed_width
            y_center *= self.displayed_height
            bb_width *= self.displayed_width
            bb_height *= self.displayed_height

            # Calculate bounding box corners
            x1, y1 = int(x_center - bb_width / 2), int(y_center - bb_height / 2)
            x2, y2 = int(x_center + bb_width / 2), int(y_center + bb_height / 2)

            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y

            class_name = (
                self.class_names[label["class_id"]]
                if self.class_names and label["class_id"] < len(self.class_names)
                else str(label["class_id"])
            )
            color = self.class_colors.get(class_name, "#0000FF")

            # Draw rectangle on canvas
            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=2, tags="bbox"
            )

            # Draw text on canvas
            text_id = self.canvas.create_text(
                x1,
                y1 - 10,
                text=class_name,
                fill=color,
                anchor=tk.SW,
                tags="bbox_text",
            )

            self.labels[i]["rect_id"] = rect_id
            self.labels[i]["text_id"] = text_id

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_and_display_image()  # Use the combined function
            self.selected_label_index = -1

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_and_display_image()  # Use the combined function
            self.selected_label_index = -1

    def on_label_select(self, event):
        selected_items = self.label_list.selection()
        if selected_items:
            selected_item = selected_items[0]
            self.selected_label_index = self.label_list.index(selected_item)
            self.populate_edit_fields()

    def populate_edit_fields(self):
        if self.selected_label_index == -1:
            return
        label = self.labels[self.selected_label_index]
        if self.class_names and label["class_id"] < len(self.class_names):
            self.class_entry.set(self.class_names[label["class_id"]])
        else:
            self.class_entry.set(str(label["class_id"]))
        for entry, key in zip(
            [
                self.x_center_entry,
                self.y_center_entry,
                self.width_entry,
                self.height_entry,
            ],
            ["x_center", "y_center", "width", "height"],
        ):
            entry.delete(0, tk.END)
            entry.insert(0, str(label[key]))

    def update_label(self):
        if self.selected_label_index == -1:
            return
        try:
            class_value = self.class_entry.get()
            try:
                class_id = self.class_names.index(class_value)
            except ValueError:
                class_id = int(class_value)
            x_center, y_center, width, height = map(
                float,
                [
                    self.x_center_entry.get(),
                    self.y_center_entry.get(),
                    self.width_entry.get(),
                    self.height_entry.get(),
                ],
            )
            if not all(0.0 <= val <= 1.0 for val in [x_center, y_center, width, height]):
                raise ValueError("Values must be between 0.0 and 1.0")

            self.labels[self.selected_label_index] = {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
            class_name = (
                self.class_names[class_id]
                if self.class_names and class_id < len(self.class_names)
                else str(class_id)
            )
            self.label_list.item(
                self.label_list.get_children()[self.selected_label_index],
                values=(class_name, x_center, y_center, width, height),
            )
            self.save_labels() # save to disk
            self.draw_bounding_boxes()  # Redraw *after* saving
            self.status_bar.config(text="Label updated and saved.")
            self.unsaved_changes = False

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}", parent=self.root) # Parent is now self.root

    def delete_label(self):
        if self.selected_label_index == -1:
            return
        if messagebox.askyesno("Confirm", "Delete label?", parent=self.root): # Parent is now self.root
            # remove the label from canvas
            self.canvas.delete(self.labels[self.selected_label_index]["rect_id"])
            self.canvas.delete(self.labels[self.selected_label_index]["text_id"])
            # delete from list
            del self.labels[self.selected_label_index]
            self.label_list.delete(self.label_list.get_children()[self.selected_label_index])
            self.selected_label_index = -1
            self.save_labels() # save to disk
            self.status_bar.config(text="Label deleted and saved.")
            self.clear_edit_fields()
            self.unsaved_changes = False


    def add_label(self):
        try:
            class_value = self.class_entry.get()
            try:
                class_id = self.class_names.index(class_value)
            except ValueError:
                class_id = int(class_value)
            x_center, y_center, width, height = map(
                float,
                [
                    self.x_center_entry.get(),
                    self.y_center_entry.get(),
                    self.width_entry.get(),
                    self.height_entry.get(),
                ],
            )
            if not all(0.0 <= val <= 1.0 for val in [x_center, y_center, width, height]):
                raise ValueError("Values must be between 0.0 and 1.0")
            new_label = {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
            self.labels.append(new_label)
            class_name = (
                self.class_names[class_id]
                if self.class_names and class_id < len(self.class_names)
                else str(class_id)
            )
            self.label_list.insert(
                "", "end", values=(class_name, x_center, y_center, width, height)
            )
            self.save_labels()  # Save immediately
            self.draw_bounding_boxes()  # Redraw *after* saving
            self.status_bar.config(text="Label added and saved.")
            self.selected_label_index = len(self.labels) - 1
            self.unsaved_changes = False
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}", parent=self.root) # Parent is now self.root
    def save_labels(self):
      if not self.label_folder: return
      label_filename = os.path.splitext(self.image_files[self.current_image_index])[0] + ".txt"
      label_path = os.path.join(self.label_folder, label_filename)
      try:
          with open(label_path, "w") as f:
              for label in self.labels:
                  f.write(f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}\n")
          self.status_bar.config(text=f"Labels saved to {label_path}")
          self.unsaved_changes = False  # Reset after successful save
      except Exception as e:
          messagebox.showerror("Error", f"Error saving labels: {e}", parent=self.root) # Parent is now self.root

    def clear_edit_fields(self):
        self.class_entry.set("")
        for entry in [
            self.x_center_entry,
            self.y_center_entry,
            self.width_entry,
            self.height_entry,
        ]:
            entry.delete(0, tk.END)

    def select_label(self, event):
        if self.image is None:
            return
        # Convert click coordinates to original image coordinates
        x = (event.x - (self.canvas_width - self.displayed_width) // 2) / self.displayed_width * self.image.width
        y = (event.y - (self.canvas_height - self.displayed_height) // 2) / self.displayed_height * self.image.height

        for i, label in enumerate(self.labels):
            x_center, y_center, width, height = [
                label[key] * self.image.width
                for key in ("x_center", "y_center", "width", "height")
            ]
            x1, y1, x2, y2 = (
                int(x_center - width / 2),
                int(y_center - height / 2),
                int(x_center + width / 2),
                int(y_center + height / 2),
            )
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_label_index = i
                for item in self.label_list.get_children():
                    if self.label_list.index(item) == i:
                        self.label_list.selection_set(item)  # Select in Treeview
                        break
                self.populate_edit_fields()
                break  # Stop after finding the first match
        else:  # Important: If no label is found
             self.selected_label_index = -1
             self.label_list.selection_remove(self.label_list.selection())  # Deselect
             self.clear_edit_fields()

    def on_close(self):
        if self.unsaved_changes:
            if messagebox.askyesno(
                "Unsaved Changes", "You have unsaved changes. Save before closing?", parent=self.root # Parent is now self.root
            ):
                self.save_labels()
        self.root.destroy()


def launch_label_correction_tool():
    """Launches the Label Correction Tool GUI in a new window."""
    label_correction_app = LabelCorrectionApp(tk.Toplevel()) # Use Toplevel to make it a child window
    # No mainloop() needed here, LabelCorrectionApp runs its own loop when instantiated.


if __name__ == '__main__':
    root_main = create_gui() # create main root for testing
    root_main.mainloop()