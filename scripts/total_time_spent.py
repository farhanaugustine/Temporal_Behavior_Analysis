import os
import pandas as pd
from collections import defaultdict
import ast  

def calculate_total_time_spent(csv_file_path, class_labels, frame_rate=30):
    """Calculates the total time spent on each behavior."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return f"Error: CSV file not found or empty: {csv_file_path}", None # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}", None # Return error string

    total_time = defaultdict(lambda: (0, 0))
    for class_label in class_labels.values():
        total_frames = len(df[df["Class Label"] == class_label])
        total_time_seconds = total_frames / frame_rate
        total_time[class_label] = (total_frames, total_time_seconds)

    return None, total_time # Return None for error, and results

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
            return f"Total time spent results saved to: {excel_path}" # Return success message

        except Exception as e:
            return f"Error saving total time spent results to Excel: {e}" # Return error message
    return "No total time data to save." # Return if no data


def main_analysis(output_folder, class_labels, frame_rate, video_name): # Keyword args
    """Main function to parse arguments and run analysis."""

    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Return error string

    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        return f"Error: {csv_file_path} not found.  Run general_analysis.py first." # Return error string

    error_time_calc, total_time_results = calculate_total_time_spent(csv_file_path, class_labels, frame_rate) # Get potential error and results
    if error_time_calc: # If calculate_total_time_spent returned an error string
        return error_time_calc # Return the error string to GUI

    excel_output_msg = save_total_time_to_excel(total_time_results, output_folder, video_name) # Save and get message

    output_messages = [msg for msg in [excel_output_msg] if msg is not None] # Collect non-None messages
    return "\n".join(output_messages) if output_messages else "Total time spent analysis completed. No specific output messages." # Return combined messages


if __name__ == "__main__":
    # Example for direct testing:
    output_folder_path = "path/to/your/output_folder" # Replace with real path
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    frame_rate_val = 30
    video_name_val = "your_video_name" # Replace with real video name

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, output_folder, class_labels, frame_rate, video_name):
            self.output_folder = output_folder
            self.class_labels = str(class_labels) # Pass as string for direct test
            self.frame_rate = frame_rate
            self.video_name = video_name

    test_args = Args(output_folder_path, class_labels_dict, frame_rate_val, video_name_val)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(output_folder=test_args.output_folder,
                                  class_labels=class_labels_dict, # Pass dict directly
                                  frame_rate=test_args.frame_rate,
                                  video_name=test_args.video_name)
    print(output_message) # Print output for direct test