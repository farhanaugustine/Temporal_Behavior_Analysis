import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import os
import subprocess
import re
import shutil
import ast

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
    """Runs a selected script from the scripts folder."""
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

    # Input validation for min_bout_duration and max_gap_duration (ONLY needed for general_analysis)
    if script_name == "general_analysis": #ONLY general_analysis needs these two.
        try:
            min_bout_duration = int(min_bout_duration)
            if min_bout_duration <= 0:
                messagebox.showerror("Error", "Minimum bout duration must be a positive integer.")
                return
        except ValueError:
            messagebox.showerror("Error", "Minimum bout duration must be a valid integer.")
            return

        try:
            max_gap_duration = int(max_gap_duration)
            if max_gap_duration < 0:  # Allow a gap of 0 (no gap tolerance)
                messagebox.showerror("Error", "Maximum gap duration must be a non-negative integer.")
                return
        except ValueError:
            messagebox.showerror("Error", "Maximum gap duration must be a valid integer.")
            return

    try:
        # Get the absolute path of the directory containing main.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "scripts", f"{script_name}.py")

        # Construct the command-line arguments.
        command = [
            "python",
            script_path,
        ]

        # Add arguments based on script name, conditionally.
        if script_name == "behavioral_rhythms_single":
            command.extend(["--output_folder", output_folder,
                            "--class_labels", str(class_labels_dict),
                            "--frame_rate", str(frame_rate_int),
                            "--video_name", video_name,
                            "--prominence", "1.0"])
        elif script_name == "transfer_entropy":
            command.extend(["--output_folder", output_folder,
                            "--class_labels", str(class_labels_dict),
                            "--frame_rate", str(frame_rate_int),
                            "--video_name", video_name,
                            "--max_lag", "150", '--k', '3'])
        elif script_name == "cross_correlation_single":
            command.extend(["--output_folder", output_folder,
                            "--class_labels", str(class_labels_dict),
                            "--frame_rate", str(frame_rate_int),
                            "--video_name", video_name,
                            "--max_lag_frames", "150"])
        elif script_name == "general_analysis":
            command.extend(["--output_folder", output_folder,
                            "--class_labels", str(class_labels_dict),
                            "--frame_rate", str(frame_rate_int),
                            "--video_name", video_name,
                            "--min_bout_duration", str(min_bout_duration),
                            "--max_gap_duration", str(max_gap_duration)])
        elif script_name in ("HMMs", "n_gram_analysis", "sequence_mining"):
            csv_folder = os.path.join(output_folder, "csv_output") #Subfolder name of csv.
            csv_file_name = f"{video_name}_analysis.csv"  #CSV is based on the video_name, includes "_analysis"
            csv_file_path = os.path.join(csv_folder, csv_file_name)

            if not os.path.exists(csv_file_path):
                messagebox.showerror("Error", f"CSV file not found: {csv_file_path}.  Did you run general_analysis first?")
                return

            command.extend(["--csv_file", csv_file_path, "--output_folder", output_folder, "--video_name", video_name]) #Only the CSV file path, output folder and video name so that pngs are saved to the correct spot

        # Use subprocess.run to execute the script
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        # Display output in the log
        log_text.insert(tk.END, f"Script {script_name} output:\n")
        log_text.insert(tk.END, result.stdout)
        if result.stderr:
            log_text.insert(tk.END, f"Script {script_name} errors:\n{result.stderr}\n")
        log_text.insert(tk.END, f"Script {script_name} executed successfully.\n")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running {script_name}: {e}")
        log_text.insert(tk.END, f"Error running {script_name}:\n{e.stdout}\n{e.stderr}\n")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find script: {script_name}.py\nPath: {script_path}") #More descriptive error message
        log_text.insert(tk.END, f"Error: Could not find {script_name}.py\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_multi_rhythm_script(output_folder, log_text):
    """Runs the multi-video rhythm analysis script."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    rhythm_folder = os.path.join(output_folder, "rhythm_excel")
    if not os.path.exists(rhythm_folder):
        messagebox.showerror("Error", "The 'rhythm_excel' folder does not exist.  Run the single-video rhythm analysis first.")
        return
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "scripts", "behavioral_rhythms_multi.py")

        command = [
            "python",
            script_path,
            "--rhythm_folder", rhythm_folder,
            "--output_folder", output_folder,
            "--max_time", "1500" #Added max_time
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        log_text.insert(tk.END, f"Script behavioral_rhythms_multi output:\n")
        log_text.insert(tk.END, result.stdout)
        if result.stderr:
            log_text.insert(tk.END, f"Script behavioral_rhythms_multi errors:\n{result.stderr}\n")
        log_text.insert(tk.END, f"Script behavioral_rhythms_multi executed successfully.\n")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running behavioral_rhythms_multi: {e}")
        log_text.insert(tk.END, f"Error running behavioral_rhythms_multi:\n{e.stdout}\n{e.stderr}\n")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find script: behavioral_rhythms_multi.py")
        log_text.insert(tk.END, f"Error: Could not find behavioral_rhythms_multi.py\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_multi_fft_script(output_folder, log_text):
    """Runs the multi-video FFT analysis script."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    fft_folder = os.path.join(output_folder, "fft_excel")
    if not os.path.exists(fft_folder):
        messagebox.showerror("Error", "The 'fft_excel' folder does not exist.  Run the single-video FFT analysis first.")
        return

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "scripts", "fft_analysis_multi.py")

        command = [
            "python",
            script_path,
            "--fft_folder", fft_folder,
            "--output_folder", output_folder,
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        log_text.insert(tk.END, f"Script fft_analysis_multi output:\n")
        log_text.insert(tk.END, result.stdout)
        if result.stderr:
            log_text.insert(tk.END, f"Script fft_analysis_multi errors:\n{result.stderr}\n")
        log_text.insert(tk.END, f"Script fft_analysis_multi executed successfully.\n")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running fft_analysis_multi: {e}")
        log_text.insert(tk.END, f"Error running fft_analysis_multi:\n{e.stdout}\n{e.stderr}\n")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find script: fft_analysis_multi.py")
        log_text.insert(tk.END, f"Error: Could not find fft_analysis_multi.py\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_cross_correlation_combine_script(output_folder, log_text):
    """Runs the multi-video cross-correlation combination script."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    cross_corr_folder = os.path.join(output_folder, "cross_correlation_excel")
    if not os.path.exists(cross_corr_folder):
        messagebox.showerror("Error", "The 'cross_correlation_excel' folder does not exist.  Run the single-video cross-correlation analysis first.")
        return

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "scripts", "cross_correlation_combine.py")

        command = [
            "python",
            script_path,
            "--cross_corr_folder", cross_corr_folder,
            "--output_folder", output_folder,
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        log_text.insert(tk.END, f"Script cross_correlation_combine output:\n")
        log_text.insert(tk.END, result.stdout)
        if result.stderr:
            log_text.insert(tk.END, f"Script cross_correlation_combine errors:\n{result.stderr}\n")
        log_text.insert(tk.END, f"Script cross_correlation_combine executed successfully.\n")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running cross_correlation_combine: {e}")
        log_text.insert(tk.END, f"Error running cross_correlation_combine:\n{e.stdout}\n{e.stderr}\n")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find script: cross_correlation_combine.py")
        log_text.insert(tk.END, f"Error: Could not find cross_correlation_combine.py\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_cross_correlation_stats_script(output_folder, log_text):
    """Runs the cross-correlation statistics script."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    cross_corr_folder = os.path.join(output_folder, "cross_correlation_excel")
    if not os.path.exists(cross_corr_folder):
        messagebox.showerror("Error", "The 'cross_correlation_excel' folder does not exist.  Run the single-video cross-correlation analysis first.")
        return

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "scripts", "cross_correlation_stats.py")

        command = [
            "python",
            script_path,
            "--cross_corr_folder", cross_corr_folder,
            "--output_folder", output_folder,
            "--metric", "peak_correlation", #Include the metric
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        log_text.insert(tk.END, f"Script cross_correlation_stats output:\n")
        log_text.insert(tk.END, result.stdout)
        if result.stderr:
            log_text.insert(tk.END, f"Script cross_correlation_stats errors:\n{result.stderr}\n")
        log_text.insert(tk.END, f"Script cross_correlation_stats executed successfully.\n")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running cross_correlation_stats: {e}")
        log_text.insert(tk.END, f"Error running cross_correlation_stats:\n{e.stdout}\n{e.stderr}\n")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find script: cross_correlation_stats.py")
        log_text.insert(tk.END, f"Error: Could not find cross_correlation_stats.py\n")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

def run_multi_transition_prob_script(output_folder, log_text):
    """Runs the multi-video transition probability analysis script."""
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    transition_folder = os.path.join(output_folder, "transition_probabilities_excel")
    if not os.path.exists(transition_folder):
        messagebox.showerror("Error", "The 'transition_probabilities_excel' folder does not exist. Run the single-video transition probability analysis first.")
        return

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "scripts", "transition_probability_multi.py")

        command = [
            "python",
            script_path,
            "--transition_folder", transition_folder,
            "--output_folder", output_folder,
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        log_text.insert(tk.END, f"Script transition_probability_multi output:\n")
        log_text.insert(tk.END, f"Script transition_probability_multi errors:\n{result.stderr}\n")
        log_text.insert(tk.END, f"Script transition_probability_multi executed successfully.\n")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running transition_probability_multi: {e}")
        log_text.insert(tk.END, f"Error running transition_probability_multi:\n{e.stdout}\n{e.stderr}\n")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find script: transition_probability_multi.py")
        log_text.insert(tk.END, f"Error: Could not find transition_probability_multi.py\n")
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

    # --- Author and Affiliation ---
    author_label = tk.Label(root, text="Author: Farhan Augustine")
    author_label.grid(row=5, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 0))

    affiliation_label = tk.Label(root, text="Affiliation: University of Maryland Baltimore County")
    affiliation_label.grid(row=6, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 0))

    affiliation_label = tk.Label(root, text="Built: Year 2025")
    affiliation_label.grid(row=7, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 10))

    # --- Input Section ---
    input_frame = ttk.LabelFrame(root, text="Input")
    input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

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
    parameter_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

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
    organize_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")


    # --- Analysis Scripts Section ---
    # Use a main frame, and subframes for categories.
    analysis_frame = ttk.Frame(root)  # Main frame, NOT a LabelFrame
    analysis_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

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
    ToolTip(btn_fft_single, "Performs Fast Fourier Transform (FFT) analysis.")

    btn_fft_multi = ttk.Button(fft_frame, text="fft_analysis_multi",
                               command=lambda: run_multi_fft_script(output_folder_entry.get(), log_text))
    btn_fft_multi.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_fft_multi, "Combines FFT analysis results from multiple videos.")


    # --- Correlation Analysis Frame ---
    correlation_frame = ttk.LabelFrame(analysis_frame, text="Correlation Analysis")
    correlation_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew") # New row, below General

    btn_basic_correlation = ttk.Button(correlation_frame, text="basic_correlation",
                                    command=lambda: run_script("basic_correlation", output_folder_entry.get(),
                                                                class_labels_dict, log_text,
                                                                frame_rate_entry.get(), video_name_entry.get(),
                                                                min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_basic_correlation.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_basic_correlation, "Calculates Spearman rank correlations.")

    btn_cross_correlation_single = ttk.Button(correlation_frame, text="cross_correlation_single",
                                            command=lambda: run_script("cross_correlation_single", output_folder_entry.get(),
                                                                        class_labels_dict, log_text,
                                                                        frame_rate_entry.get(), video_name_entry.get(),
                                                                        min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_cross_correlation_single.grid(row=1, column=0, padx=5, pady=5, sticky="ew") # Stays in the same frame
    ToolTip(btn_cross_correlation_single, "Calculates time-lagged cross-correlations.")

    btn_cross_correlation_combine = ttk.Button(correlation_frame, text="cross_correlation_combine",
                                               command=lambda: run_cross_correlation_combine_script(output_folder_entry.get(), log_text))
    btn_cross_correlation_combine.grid(row=2, column=0, padx=5, pady=5, sticky="ew") # Stays in the same frame
    ToolTip(btn_cross_correlation_combine, "Combines cross-correlation results.")

    btn_cross_correlation_stats = ttk.Button(correlation_frame, text="cross_correlation_stats",
                                             command=lambda: run_cross_correlation_stats_script(output_folder_entry.get(), log_text))
    btn_cross_correlation_stats.grid(row=3, column=0, padx=5, pady=5, sticky="ew") # Stays in the same frame
    ToolTip(btn_cross_correlation_stats, "Performs statistical analysis on cross-correlation.")

    # --- Causality Analysis Frame ---
    causality_frame = ttk.LabelFrame(analysis_frame, text="Causality Analysis")
    causality_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew") # Next column, same row as Correlation

    btn_granger = ttk.Button(causality_frame, text="granger_causality",
                            command=lambda: run_script("granger_causality", output_folder_entry.get(),
                                                        class_labels_dict, log_text,
                                                        frame_rate_entry.get(), video_name_entry.get(),
                                                        min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_granger.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_granger, "Performs Granger causality analysis.")

    btn_transfer_entropy = ttk.Button(causality_frame, text="transfer_entropy",
                                    command=lambda: run_script("transfer_entropy", output_folder_entry.get(),
                                                                class_labels_dict, log_text,
                                                                frame_rate_entry.get(), video_name_entry.get(),
                                                                min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_transfer_entropy.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_transfer_entropy, "Calculates transfer entropy.")

    # --- Transition Analysis Frame ---
    transition_frame = ttk.LabelFrame(analysis_frame, text="Transition Analysis")
    transition_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")  # Next column

    btn_transition_single = ttk.Button(transition_frame, text="transition_probability_single",
                                        command=lambda: run_script("transition_probability_single", output_folder_entry.get(),
                                                                    class_labels_dict, log_text,
                                                                    frame_rate_entry.get(), video_name_entry.get(),
                                                                    min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_transition_single.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_transition_single, "Calculates transition probabilities.")

    btn_transition_multi = ttk.Button(transition_frame, text="transition_probability_multi",
                                      command=lambda: run_multi_transition_prob_script(output_folder_entry.get(), log_text))
    btn_transition_multi.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_transition_multi, "Combines transition probability results.")

    # --- Sequence Analysis Frame ---
    sequence_frame = ttk.LabelFrame(analysis_frame, text="Sequence Analysis")
    sequence_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

    btn_hmms = ttk.Button(sequence_frame, text="HMMs",
                            command=lambda: run_script("HMMs", output_folder_entry.get(),
                                                        class_labels_dict, log_text,
                                                        frame_rate_entry.get(), video_name_entry.get(),
                                                        min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_hmms.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_hmms, "Runs the HMMs.py script.")

    btn_ngram = ttk.Button(sequence_frame, text="n_gram_analysis",
                               command=lambda: run_script("n_gram_analysis", output_folder_entry.get(),
                                                            class_labels_dict, log_text,
                                                            frame_rate_entry.get(), video_name_entry.get(),
                                                            min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_ngram.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ToolTip(btn_ngram, "Runs the n_gram_analysis.py script.")

    btn_sequence_mining = ttk.Button(sequence_frame, text="sequence_mining",
                                    command=lambda: run_script("sequence_mining", output_folder_entry.get(),
                                                                class_labels_dict, log_text,
                                                                frame_rate_entry.get(), video_name_entry.get(),
                                                                min_bout_duration_entry.get(), max_gap_duration_entry.get()))
    btn_sequence_mining.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
    ToolTip(btn_sequence_mining, "Runs the sequence_mining.py script.")

    # --- Multi-Video Analysis Frame ---
    multi_frame = ttk.LabelFrame(analysis_frame, text="Multi-Video Analysis")
    multi_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")

    btn_hmms_multi = ttk.Button(multi_frame, text="HMMs_multi",
                            command=lambda: run_multi_script("HMMs_multi", input_folder_entry.get(), output_folder_entry.get(), log_text))
    btn_hmms_multi.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    ToolTip(btn_hmms_multi, "Runs the HMMs_multi.py script.")

    btn_ngram_multi = ttk.Button(multi_frame, text="n_gram_analysis_multi",
                               command=lambda: run_multi_script("n_gram_analysis_multi", input_folder_entry.get(), output_folder_entry.get(), log_text))
    btn_ngram_multi.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ToolTip(btn_ngram_multi, "Runs the n_gram_analysis_multi.py script.")

    btn_sequence_mining_multi = ttk.Button(multi_frame, text="sequence_mining_multi",
                                    command=lambda: run_multi_script("sequence_mining_multi", input_folder_entry.get(), output_folder_entry.get(), log_text))
    btn_sequence_mining_multi.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
    ToolTip(btn_sequence_mining_multi, "Runs the sequence_mining_multi.py script.")

    # --- Log Display ---
    log_frame = ttk.LabelFrame(root, text="Log")
    log_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    log_text = scrolledtext.ScrolledText(log_frame, width=80, height=15, state='normal')
    log_text.grid(row=0, column=0, sticky="nsew")

    # --- Making the GUI Resizable ---
    root.grid_rowconfigure(2, weight=1)  # Analysis frame row expands
    root.grid_rowconfigure(3, weight=1)  # Log frame row expands
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    input_frame.columnconfigure(1, weight=1)  # Input frame entries expand
    # analysis_frame.columnconfigure(0, weight=1)  # Analysis buttons expand
    # analysis_frame.columnconfigure(1, weight=1)
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
    """Runs a selected script from the scripts folder."""
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

        # Get the absolute path of the directory containing main.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "scripts", f"{script_name}.py")

        # Construct the command-line arguments.
        command = [
            "python",
            script_path,
            "--input_folder", input_folder, #This is the user defined input folder
            "--output_folder", output_folder,
        ]

        # Use subprocess.run to execute the script
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        # Display output in the log
        log_text.insert(tk.END, f"Script {script_name} output:\n")
        log_text.insert(tk.END, result.stdout)
        if result.stderr:
            log_text.insert(tk.END, f"Script {script_name} errors:\n{result.stderr}\n")
        log_text.insert(tk.END, f"Script {script_name} executed successfully.\n")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running {script_name}: {e}")
        log_text.insert(tk.END, f"Error running {script_name}:\n{e.stdout}\n{e.stderr}\n")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find script: {script_name}.py\nPath: {script_path}") #More descriptive error message
        log_text.insert(tk.END, f"Error: Could not find {script_name}.py\n")
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

if __name__ == '__main__':
    app = create_gui()
    app.mainloop()