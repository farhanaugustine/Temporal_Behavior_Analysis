import os
import csv
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem  
import numpy as np
import ast

def parse_yolo_output(txt_file_path, class_labels):
    """Parses a YOLO output file, handling errors."""
    detections = []
    try:
        with open(txt_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:  
                        class_id = int(parts[0])
                        if isinstance(class_labels, list):
                            class_label = class_labels[class_id] if class_id < len(class_labels) else "Unknown"
                        elif isinstance(class_labels, dict):
                            class_label = class_labels.get(class_id, "Unknown")
                        else:
                            raise ValueError("class_labels must be a list or a dictionary")
                        detections.append({"class_id": class_id, "class_label": class_label})
                    except ValueError:
                         print(f"Invalid class ID in {txt_file_path}: {parts[0]}")
                         continue 
                else:
                    print(f"Skipping invalid line in {txt_file_path}: {line.strip()}") 

    except (FileNotFoundError, ValueError, IndexError) as e:
        return f"Error processing {txt_file_path}: {e}", []  # Return error string and empty list
    return None, detections # Return None for error, and detections


def analyze_detections(output_folder, class_labels, csv_output_folder, video_name, min_bout_duration=3, max_gap_duration=5, frame_rate=30):
    """Analyzes YOLO outputs, creates CSV, filters bouts."""
    os.makedirs(csv_output_folder, exist_ok=True)
    class_frame_to_bout = defaultdict(lambda: defaultdict(lambda: {}))
    last_detection_frame = defaultdict(lambda: {})
    class_bouts = defaultdict(lambda: defaultdict(list))
    output_messages = [] # List to collect messages

    txt_files = [f for f in os.listdir(output_folder) if f.startswith(video_name + "_") and f.endswith(".txt")]
    txt_files = sorted(txt_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    for filename in txt_files:
        try:
            frame_number = int(filename.split("_")[-1].split(".")[0])
        except ValueError:
            message_invalid_frame = f"Skipping file {filename}: Invalid frame number."
            print(message_invalid_frame)
            output_messages.append(message_invalid_frame)
            continue

        txt_file_path = os.path.join(output_folder, filename)
        error_parse, detections = parse_yolo_output(txt_file_path, class_labels) # Get potential error string
        if error_parse: #If there was an error in parsing
            output_messages.append(error_parse) #Add error to messages
            continue  # Skip to the next file
        if not detections:
            continue  # Skip to the next file if no detections

        for detection in detections:
            class_label = detection["class_label"]

            if class_label in last_detection_frame[video_name]:
                gap = frame_number - last_detection_frame[video_name][class_label] - 1
                if gap <= max_gap_duration:
                    if class_bouts[video_name][class_label] and class_bouts[video_name][class_label][-1][1] < frame_number - 1:
                        class_bouts[video_name][class_label][-1] = (class_bouts[video_name][class_label][-1][0], frame_number)
                        bout_id = len(class_bouts[video_name][class_label])
                    elif class_bouts[video_name][class_label] and class_bouts[video_name][class_label][-1][1] == frame_number - 1:
                        class_bouts[video_name][class_label][-1] = (class_bouts[video_name][class_label][-1][0], frame_number)
                        bout_id = len(class_bouts[video_name][class_label])
                    else:
                        class_bouts[video_name][class_label].append((frame_number, frame_number))
                        bout_id = len(class_bouts[video_name][class_label])
                else:
                    class_bouts[video_name][class_label].append((frame_number, frame_number))
                    bout_id = len(class_bouts[video_name][class_label])
            else:
                class_bouts[video_name][class_label].append((frame_number, frame_number))
                bout_id = len(class_bouts[video_name][class_label])

            last_detection_frame[video_name][class_label] = frame_number
            class_frame_to_bout[video_name][frame_number][class_label] = bout_id

    # Minimum Bout Duration Filtering (same as before)
    filtered_bouts = defaultdict(lambda: defaultdict(list))
    for video_name, class_data in class_bouts.items():
        for class_label, bouts in class_data.items():
            for start_frame, end_frame in bouts:
                if (end_frame - start_frame + 1) >= min_bout_duration:
                    filtered_bouts[video_name][class_label].append((start_frame, end_frame))

    # Rebuild class_frame_to_bout with filtered bouts (same as before)
    class_frame_to_bout = defaultdict(lambda: defaultdict(lambda: {}))
    bout_id_counter = defaultdict(lambda: defaultdict(int))
    for video_name, class_data in filtered_bouts.items():
        for class_label, bouts in class_data.items():
            for start_frame, end_frame in bouts:
                bout_id_counter[video_name][class_label] += 1
                bout_id = bout_id_counter[video_name][class_label]
                for frame_num in range(start_frame, end_frame + 1):
                    class_frame_to_bout[video_name][frame_num][class_label] = bout_id

    # Write to CSV
    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")
    try:
        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Frame Number", "Class Label", "Bout ID"])
            for frame_number in sorted(class_frame_to_bout[video_name]):
                for class_label, bout_id in class_frame_to_bout[video_name][frame_number].items():
                    csv_writer.writerow([frame_number, class_label, bout_id])
        message_csv_saved = f"CSV file saved: {csv_file_path}"
        print(message_csv_saved)
        output_messages.append(message_csv_saved)
        return csv_file_path, output_messages # Return path and messages

    except Exception as e:
        message_csv_error = f"Error writing to CSV: {e}"
        print(message_csv_error)
        output_messages.append(message_csv_error)
        return None, output_messages # Return None and messages

def calculate_average_time_between_bouts(csv_file_path, class_label, frame_rate=30):
    """Calculates the average time between consecutive bouts."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return f"Error: CSV file not found or empty: {csv_file_path}", None # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}", None # Return error string

    df['bout_start'] = ((df['Class Label'] == class_label) &
                        ((df['Frame Number'] == 1) |
                         (df['Class Label'].shift(1) != class_label)))

    df['bout_end'] = ((df['Class Label'] == class_label) &
                      (df['Class Label'].shift(-1) != class_label))

    start_frames = df[df['bout_start']].index.tolist()
    end_frames = df[df['bout_end']].index.tolist()

    if len(start_frames) < 2:
         return None, (0, 0) #Return 0s if less than 2 bouts

    time_diffs = [start_frames[i] - end_frames[i-1] for i in range(1, len(start_frames))]
    avg_time_frames = sum(time_diffs) / len(time_diffs) if time_diffs else 0
    avg_time_seconds = avg_time_frames / frame_rate

    return None, (avg_time_frames, avg_time_seconds) #Return None for error, and results

def calculate_bout_info(csv_file_path, class_label, frame_rate=30):
    """Calculates bout statistics (number, duration)."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return f"Error: CSV file not found or empty: {csv_file_path}", None # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}", None # Return error string

    df['bout_start'] = ((df['Class Label'] == class_label) &
                        ((df['Frame Number'] == 1) |
                         (df['Class Label'].shift(1) != class_label)))
    df['bout_end'] = ((df['Class Label'] == class_label) &
                      (df['Class Label'].shift(-1) != class_label))

    start_frames = df[df['bout_start']]['Frame Number'].tolist()
    end_frames = df[df['bout_end']]['Frame Number'].tolist()
    num_bouts = len(start_frames)
    bout_durations = [end_frames[i] - start_frames[i] + 1 for i in range(num_bouts)]  # +1 for inclusive duration
    avg_duration_frames = sum(bout_durations) / num_bouts if num_bouts else 0
    avg_duration_seconds = avg_duration_frames / frame_rate

    return None, (num_bouts, avg_duration_frames, avg_duration_seconds, bout_durations) #Return None for error, and results

def save_analysis_to_excel(csv_file_path, output_folder, class_labels, frame_rate, video_name):
    """Saves analysis results to an Excel file."""
    excel_path = os.path.join(output_folder, f"{video_name}_general_analysis.xlsx")
    output_messages = [] # List to collect messages

    try:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            all_data = []
            for class_label in class_labels.values():
                error_time_between, (avg_time_frames, avg_time_seconds) = calculate_average_time_between_bouts(csv_file_path, class_label, frame_rate) # Get potential error and results
                error_bout_info, (num_bouts, avg_bout_duration_frames, avg_bout_duration_seconds, bout_durations) = calculate_bout_info(csv_file_path, class_label, frame_rate) # Get potential error and results

                if error_time_between: #If there was an error from calculate_average_time_between_bouts
                    output_messages.append(error_time_between) # Add error message

                if error_bout_info: #If there was an error from calculate_bout_info
                    output_messages.append(error_bout_info) #Add error message

                if not error_time_between and not error_bout_info: # Proceed only if there were no errors
                    data = {
                        "Class Label": class_label,
                        "Average Time Between Bouts (frames)": avg_time_frames,
                        "Average Time Between Bouts (seconds)": avg_time_seconds,
                        "Number of Bouts": num_bouts,
                        "Average Bout Duration (frames)": avg_bout_duration_frames,
                        "Average Bout Duration (seconds)": avg_bout_duration_seconds,
                        "Bout Durations (frames)": str(bout_durations)
                    }
                    all_data.append(data)
                #Added: if there is no average_time between bouts (no errors, but no bouts)
                else:
                    data = {
                        "Class Label": class_label,
                        "Average Time Between Bouts (frames)": 0,
                        "Average Time Between Bouts (seconds)": 0,
                        "Number of Bouts": 0,
                        "Average Bout Duration (frames)": 0,
                        "Average Bout Duration (seconds)": 0,
                        "Bout Durations (frames)": str([0])
                    }
                    all_data.append(data)


            df_general = pd.DataFrame(all_data)
            df_general.to_excel(writer, sheet_name='Bout Information', index=False)
        message_excel_saved = f"General analysis saved to: {excel_path}"
        print(message_excel_saved)
        output_messages.append(message_excel_saved)
        return output_messages #Return messages

    except Exception as e:
        message_excel_error = f"Error saving to Excel: {e}"
        print(message_excel_error)
        output_messages.append(message_excel_error)
        return output_messages #Return messages

def plot_general_analysis(csv_file_path, class_labels, frame_rate, output_folder, video_name):
    """Generates box plots for bout durations, time between bouts, and average bout duration."""

    output_messages = [] # List to collect messages
    data_for_plotting = {}
    for class_label in class_labels.values():
        error_bout_info, (num_bouts, avg_bout_duration_frames, avg_bout_duration_seconds, bout_durations) = calculate_bout_info(csv_file_path, class_label, frame_rate) # Get potential error and results
        error_time_between, (avg_time_seconds, _) = calculate_average_time_between_bouts(csv_file_path, class_label, frame_rate) # Get potential error and results

        if error_bout_info: # If there was an error from calculate_bout_info
            output_messages.append(error_bout_info) # Add error message
            continue # Skip to the next class label
        if error_time_between: # If there was an error from calculate_average_time_between_bouts
            output_messages.append(error_time_between) # Add error message
            continue # Skip to the next class label


        # Prepare data for plotting, handling None values and ensuring lists
        data_for_plotting[class_label] = {
            'bout_durations': bout_durations if bout_durations else [0],  # Ensure a list
            'avg_time_between_bouts': [avg_time_seconds] if avg_time_seconds is not None else [0],  # Ensure a list
            'avg_bout_duration': [avg_bout_duration_seconds] if avg_bout_duration_seconds is not None else [0],  # Ensure a list
        }

    # Box plot for Bout Durations
    plt.figure(figsize=(12, 6))
    bout_durations_data = [data_for_plotting[cl]['bout_durations'] for cl in class_labels.values()]
    plt.boxplot(bout_durations_data, tick_labels=class_labels.values(), showmeans=True, meanline=True)
    plt.title(f'Bout Durations for {video_name}')
    plt.ylabel('Duration (Frames)')
    plt.xlabel('Behavior')
    plt.grid(True)
    plt.tight_layout()
    bout_durations_plot_path = os.path.join(output_folder, f"{video_name}_bout_durations.png")
    plt.savefig(bout_durations_plot_path)
    plt.close()
    message_plot_durations = f"Plot saved: {bout_durations_plot_path}"
    print(message_plot_durations)
    output_messages.append(message_plot_durations)


    # Box plot for Average Time Between Bouts
    plt.figure(figsize=(12, 6))
    avg_times_between_data = [data_for_plotting[cl]['avg_time_between_bouts'] for cl in class_labels.values()]
    plt.boxplot(avg_times_between_data, tick_labels=class_labels.values(), showmeans=True, meanline=True)
    plt.title(f'Average Time Between Bouts for {video_name}')
    plt.ylabel('Time (Seconds)')
    plt.xlabel('Behavior')
    plt.grid(True)
    plt.tight_layout()
    time_between_plot_path = os.path.join(output_folder, f"{video_name}_avg_time_between_bouts.png")
    plt.savefig(time_between_plot_path)
    plt.close()
    message_plot_time_between = f"Plot saved: {time_between_plot_path}"
    print(message_plot_time_between)
    output_messages.append(message_plot_time_between)

    # Box plot for Average Bout Duration
    plt.figure(figsize=(12, 6))
    avg_bout_durations_data = [data_for_plotting[cl]['avg_bout_duration'] for cl in class_labels.values()]
    plt.boxplot(avg_bout_durations_data, tick_labels=class_labels.values(), showmeans=True, meanline=True)
    plt.title(f'Average Bout Duration for {video_name}')
    plt.ylabel('Time (Seconds)')
    plt.xlabel('Behavior')
    plt.grid(True)
    plt.tight_layout()
    avg_duration_plot_path = os.path.join(output_folder, f"{video_name}_avg_bout_duration.png")
    plt.savefig(avg_duration_plot_path)
    plt.close()
    message_plot_avg_duration = f"Plot saved: {avg_duration_plot_path}"
    print(message_plot_avg_duration)
    output_messages.append(message_plot_avg_duration)

    message_plots_saved = f"Plots saved to: {output_folder}"
    print(message_plots_saved)
    output_messages.append(message_plots_saved)

    return output_messages #Return messages

def main_analysis(output_folder, class_labels, frame_rate, video_name, min_bout_duration=3, max_gap_duration=5): # Keyword args with defaults
    """Main function to run general behavioral analysis."""

    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Return error string

    csv_file_path, detect_messages = analyze_detections(output_folder, class_labels, csv_output_folder, video_name, min_bout_duration, max_gap_duration, frame_rate) # Get path and messages
    output_messages = detect_messages # Start with messages from analyze_detections

    if csv_file_path: # If CSV was successfully created
        excel_messages = save_analysis_to_excel(csv_file_path, output_folder, class_labels, frame_rate, video_name) # Get excel messages
        if excel_messages:
            output_messages.extend(excel_messages) # Extend with excel messages
        plot_messages = plot_general_analysis(csv_file_path, class_labels, frame_rate, output_folder, video_name) # Get plot messages
        if plot_messages:
            output_messages.extend(plot_messages) # Extend with plot messages
    else:
        return "\n".join(output_messages) # If CSV failed, return messages and stop


    return "\n".join(output_messages) if output_messages else "General analysis completed. No specific output messages." # Return combined messages

if __name__ == "__main__":
    # Example for direct testing:
    output_folder_path = "path/to/your/yolo_output_folder" # Replace
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    frame_rate_val = 30
    video_name_val = "your_video_name" # Replace
    min_bout_duration_val = 5
    max_gap_duration_val = 7

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, output_folder, class_labels, frame_rate, video_name, min_bout_duration, max_gap_duration):
            self.output_folder = output_folder
            self.class_labels = str(class_labels) # Pass as string for direct test
            self.frame_rate = frame_rate
            self.video_name = video_name
            self.min_bout_duration = min_bout_duration
            self.max_gap_duration = max_gap_duration

    test_args = Args(output_folder_path, class_labels_dict, frame_rate_val, video_name_val, min_bout_duration_val, max_gap_duration_val)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(output_folder=test_args.output_folder, 
                                  class_labels=class_labels_dict, # Pass dict directly
                                  frame_rate=test_args.frame_rate, 
                                  video_name=test_args.video_name,
                                  min_bout_duration=test_args.min_bout_duration,
                                  max_gap_duration=test_args.max_gap_duration)
    print(output_message) # Print output for direct test