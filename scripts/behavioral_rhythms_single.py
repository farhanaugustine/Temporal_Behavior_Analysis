import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import ast  

def calculate_behavioral_rhythms(csv_file_path, class_labels, frame_rate=30, prominence=1):
    """Detects behavioral rhythms using peak detection (single video)."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return f"Error: CSV file not found or empty: {csv_file_path}" # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}" # Return error string

    rhythms = {}
    for class_label in class_labels.values():
        activity_signal = pd.Series([x == class_label for x in df['Class Label'].tolist()])
        peaks, _ = find_peaks(activity_signal.astype(int), prominence=prominence)
        peak_times = [peak / frame_rate for peak in peaks]
        rhythms[class_label] = peak_times
    return rhythms

def save_rhythms_to_excel(rhythms, output_folder, video_name):
    """Saves rhythm results to Excel."""
    if rhythms:
        excel_path = os.path.join(output_folder, f"{video_name}_behavioral_rhythms.xlsx")
        try:
            rows = []
            for class_label, peak_times in rhythms.items():
                rows.append({"Behavior": class_label, "Peak Times (seconds)": str(peak_times)})
            df_rhythms = pd.DataFrame(rows)
            df_rhythms.to_excel(excel_path, sheet_name="Behavioral Rhythms", index=False)
            return f"Behavioral rhythms saved to: {excel_path}" # Return success string
        except Exception as e:
            return f"Error saving rhythms to Excel: {e}" # Return error string
    return "No rhythms to save." # Return string if no rhythms


def main_analysis(output_folder, class_labels, frame_rate, video_name, prominence=1.0): # Keyword args, defaults
    """Main function to run single-video rhythm analysis."""

    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Return error string

    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        return f"Error: {csv_file_path} not found. Run general_analysis first." # Return error string

    rhythm_results = calculate_behavioral_rhythms(csv_file_path, class_labels, frame_rate, prominence)
    if isinstance(rhythm_results, str): # Check for error string from calculate_behavioral_rhythms
        return rhythm_results # Return the error string

    excel_output_msg = save_rhythms_to_excel(rhythm_results, output_folder, video_name)

    output_messages = [msg for msg in [excel_output_msg] if msg is not None] # Collect non-None messages
    return "\n".join(output_messages) if output_messages else "Behavioral rhythm analysis completed. No specific output messages."


if __name__ == "__main__":
    # Example for direct testing:
    output_folder_path = "path/to/your/output_folder" # Replace with real path
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    frame_rate_val = 30
    video_name_val = "your_video_name" # Replace with real video name
    prominence_val = 1.2

    # Dummy Args class for testing (not needed for GUI integration, just for direct test)
    class Args:
        def __init__(self, output_folder, class_labels, frame_rate, video_name, prominence):
            self.output_folder = output_folder
            self.class_labels = str(class_labels) # Pass as string for direct test
            self.frame_rate = frame_rate
            self.video_name = video_name
            self.prominence = prominence

    test_args = Args(output_folder_path, class_labels_dict, frame_rate_val, video_name_val, prominence_val)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Call main_analysis directly with keyword arguments for testing:
    output_message = main_analysis(output_folder=test_args.output_folder, 
                                  class_labels=class_labels_dict, # Pass dict directly
                                  frame_rate=test_args.frame_rate, 
                                  video_name=test_args.video_name,
                                  prominence=test_args.prominence)
    print(output_message) # Print output for direct test