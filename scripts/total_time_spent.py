import os
import pandas as pd
import argparse
from collections import defaultdict
import ast  # Import ast

def calculate_total_time_spent(csv_file_path, class_labels, frame_rate=30):
    """Calculates the total time spent on each behavior."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    total_time = defaultdict(lambda: (0, 0))
    for class_label in class_labels.values():
        total_frames = len(df[df["Class Label"] == class_label])
        total_time_seconds = total_frames / frame_rate
        total_time[class_label] = (total_frames, total_time_seconds)

    return total_time

def save_total_time_to_excel(total_time_results, output_folder, video_name):
    """Saves total time spent results to Excel."""

    if total_time_results:
        excel_path = os.path.join(output_folder, f"{video_name}_total_time_spent.xlsx")
        try:
            rows = []
            for class_label, (total_frames, total_seconds) in total_time_results.items():
                rows.append({"Class Label": class_label, "Total Frames": total_frames, "Total Seconds": total_seconds})
            df_total_time = pd.DataFrame(rows)
            df_total_time.to_excel(excel_path, sheet_name="Total Time Spent", index=False)
            print(f"Total time spent results saved to: {excel_path}")

        except Exception as e:
            print(f"Error saving total time spent results to Excel: {e}")
def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Calculate total time spent on each behavior.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.")
    parser.add_argument("--video_name", required=True, help="Video name.")
    args = parser.parse_args()

    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    try:
        class_labels_dict = ast.literal_eval(args.class_labels)  # Use ast.literal_eval
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels: {e}")
        return

    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")
    if os.path.exists(csv_file_path):
        total_time_results = calculate_total_time_spent(csv_file_path, class_labels_dict, args.frame_rate)
        save_total_time_to_excel(total_time_results, args.output_folder, args.video_name)
    else:
        print(f"Error: {csv_file_path} not found.  Run general_analysis.py first.")

if __name__ == "__main__":
    main()