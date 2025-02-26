import os
import numpy as np
import pandas as pd
import argparse
from scipy.signal import find_peaks
import ast  # Import ast

def calculate_behavioral_rhythms(csv_file_path, class_labels, frame_rate=30, prominence=1):
    """Detects behavioral rhythms using peak detection (single video)."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

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
            print(f"Behavioral rhythms saved to: {excel_path}")
        except Exception as e:
            print(f"Error saving rhythms to Excel: {e}")

def main():
    """Main function to parse arguments and run single-video analysis."""
    parser = argparse.ArgumentParser(description="Detect behavioral rhythms for a single video.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.")
    parser.add_argument("--video_name", required=True, help="Video name.")
    parser.add_argument("--prominence", type=float, default=1.0, help="Prominence for peak detection.")
    args = parser.parse_args()

    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    try:
        class_labels_dict = ast.literal_eval(args.class_labels)
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels: {e}")
        return

    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found. Run general_analysis.py first.")
        return

    rhythm_results = calculate_behavioral_rhythms(csv_file_path, class_labels_dict, args.frame_rate, args.prominence)
    if rhythm_results:
        save_rhythms_to_excel(rhythm_results, args.output_folder, args.video_name)

if __name__ == "__main__":
    main()