import os
import csv
from collections import defaultdict
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import numpy as np

def parse_yolo_output(txt_file_path, class_labels):
    """Parses a YOLO output file, handling errors."""
    detections = []
    try:
        with open(txt_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if isinstance(class_labels, list):
                        class_label = class_labels[class_id] if class_id < len(class_labels) else "Unknown"
                    elif isinstance(class_labels, dict):
                        class_label = class_labels.get(class_id, "Unknown")
                    else:
                        raise ValueError("class_labels must be a list or a dictionary")
                    detections.append({"class_id": class_id, "class_label": class_label})
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error processing {txt_file_path}: {e}")
        return []
    return detections

def analyze_detections(output_folder, class_labels, csv_output_folder, video_name, min_bout_duration=3, max_gap_duration=5, frame_rate=30):
    """
    Analyzes YOLO outputs to create a CSV with frame, class, and bout ID.
    Includes bout filtering based on minimum duration and gap tolerance.
    """
    os.makedirs(csv_output_folder, exist_ok=True)
    class_frame_to_bout = defaultdict(lambda: defaultdict(lambda: {}))
    last_detection_frame = defaultdict(lambda: {})
    class_bouts = defaultdict(lambda: defaultdict(list))

    txt_files = [f for f in os.listdir(output_folder) if f.startswith(video_name + "_") and f.endswith(".txt")]
    txt_files = sorted(txt_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    for filename in txt_files:
        try:
            frame_number = int(filename.split("_")[-1].split(".")[0])
        except ValueError:
            print(f"Skipping file {filename}: Invalid frame number.")
            continue

        txt_file_path = os.path.join(output_folder, filename)
        detections = parse_yolo_output(txt_file_path, class_labels)
        if not detections:
            continue

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

    # Minimum Bout Duration Filtering
    filtered_bouts = defaultdict(lambda: defaultdict(list))
    for video_name, class_data in class_bouts.items():
        for class_label, bouts in class_data.items():
            for start_frame, end_frame in bouts:
                if (end_frame - start_frame + 1) >= min_bout_duration:
                    filtered_bouts[video_name][class_label].append((start_frame, end_frame))

    # Rebuild class_frame_to_bout with filtered bouts
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
        print(f"CSV file saved: {csv_file_path}")
        return csv_file_path  # Return the path on success
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return None  # Return None on error

def calculate_average_time_between_bouts(csv_file_path, class_label, frame_rate=30):
    """Calculates the average time between consecutive bouts."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    df['bout_start'] = ((df['Class Label'] == class_label) &
                        ((df['Frame Number'] == 1) |
                         (df['Class Label'].shift(1) != class_label)))

    df['bout_end'] = ((df['Class Label'] == class_label) &
                      (df['Class Label'].shift(-1) != class_label))

    start_frames = df[df['bout_start']].index.tolist()
    end_frames = df[df['bout_end']].index.tolist()

    if len(start_frames) < 2:
        return 0, 0

    time_diffs = [start_frames[i] - end_frames[i-1] for i in range(1, len(start_frames))]
    avg_time_frames = sum(time_diffs) / len(time_diffs) if time_diffs else 0
    avg_time_seconds = avg_time_frames / frame_rate

    return avg_time_frames, avg_time_seconds

def calculate_bout_info(csv_file_path, class_label, frame_rate=30):
    """Calculates bout statistics (number, duration)."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, None, None

    df['bout_start'] = ((df['Class Label'] == class_label) &
                        ((df['Frame Number'] == 1) |
                         (df['Class Label'].shift(1) != class_label)))
    df['bout_end'] = ((df['Class Label'] == class_label) &
                      (df['Class Label'].shift(-1) != class_label))

    start_frames = df[df['bout_start']]['Frame Number'].tolist()
    end_frames = df[df['bout_end']]['Frame Number'].tolist()
    num_bouts = len(start_frames)
    bout_durations = [end_frames[i] - start_frames[i] + 1 for i in range(num_bouts)]  # +1 inclusive
    avg_duration_frames = sum(bout_durations) / num_bouts if num_bouts else 0
    avg_duration_seconds = avg_duration_frames / frame_rate

    return num_bouts, avg_duration_frames, avg_duration_seconds, bout_durations

def save_analysis_to_excel(csv_file_path, output_folder, class_labels, frame_rate, video_name):
    """Saves analysis results to an Excel file."""
    excel_path = os.path.join(output_folder, f"{video_name}_general_analysis.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            all_data = []
            for class_label in class_labels.values():
                avg_time_frames, avg_time_seconds = calculate_average_time_between_bouts(csv_file_path, class_label, frame_rate)
                num_bouts, avg_bout_duration_frames, avg_bout_duration_seconds, bout_durations = calculate_bout_info(csv_file_path, class_label, frame_rate)

                if avg_time_frames is not None:  # Check for valid results
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

            df_general = pd.DataFrame(all_data)
            df_general.to_excel(writer, sheet_name='Bout Information', index=False)
        print(f"General analysis saved to: {excel_path}")

    except Exception as e:
        print(f"Error saving to Excel: {e}")

def plot_general_analysis(csv_file_path, class_labels, frame_rate, output_folder, video_name):
    """Generates box plots for bout durations and time between bouts."""

    data_for_plotting = {}
    for class_label in class_labels.values():
        num_bouts, avg_bout_duration_frames, avg_bout_duration_seconds, bout_durations = calculate_bout_info(csv_file_path, class_label, frame_rate)
        avg_time_frames, avg_time_seconds = calculate_average_time_between_bouts(csv_file_path, class_label, frame_rate)

        if num_bouts is not None:  # Ensure data exists
            data_for_plotting[class_label] = {
                'bout_durations': bout_durations,
                'avg_time_between_bouts': avg_time_seconds if avg_time_seconds is not None else 0, # Use 0 as a default
                'avg_bout_duration': avg_bout_duration_seconds if avg_bout_duration_seconds is not None else 0,  #Use 0 as default
            }
        else: #If there is no bout, assign all values to 0.
            data_for_plotting[class_label] = {
                'bout_durations': [0],
                'avg_time_between_bouts': 0,
                'avg_bout_duration': 0,
            }

    # Box plot for Bout Durations
    plt.figure(figsize=(12, 6))
    bout_durations_data = [data_for_plotting[cl]['bout_durations'] for cl in class_labels.values()]
    plt.boxplot(bout_durations_data, labels=class_labels.values(), showmeans=True, meanline=True)
    plt.title(f'Bout Durations for {video_name}')
    plt.ylabel('Duration (Frames)')
    plt.xlabel('Behavior')
    #Adding the SEM
    for i, (class_label, data) in enumerate(data_for_plotting.items()):
        if data['bout_durations']:  # Ensure there's data to calculate
            mean_duration = np.mean(data['bout_durations'])
            sem_duration = sem(data['bout_durations'])
            plt.errorbar(i + 1, mean_duration, yerr=sem_duration, fmt='none', ecolor='red', capsize=5, label='SEM' if i == 0 else "") # Only show label once
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{video_name}_bout_durations.png"))
    plt.close()


    # Bar plot for Average Time Between Bouts
    plt.figure(figsize=(12, 6))
    avg_times_between = [data_for_plotting[cl]['avg_time_between_bouts'] for cl in class_labels.values()]
    plt.bar(class_labels.values(), avg_times_between, yerr=[sem([data_for_plotting[cl]['avg_time_between_bouts']]) for cl in class_labels.values()], capsize=5, color='skyblue', edgecolor='black') # SEM
    plt.title(f'Average Time Between Bouts for {video_name}')
    plt.ylabel('Time (Seconds)')
    plt.xlabel('Behavior')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{video_name}_avg_time_between_bouts.png"))
    plt.close()

    # Bar plot for Average Bout Duration
    plt.figure(figsize=(12, 6))
    avg_bout_durations = [data_for_plotting[cl]['avg_bout_duration'] for cl in class_labels.values()]
    plt.bar(class_labels.values(), avg_bout_durations, yerr=[sem([data_for_plotting[cl]['avg_bout_duration']]) for cl in class_labels.values()], capsize=5, color='lightgreen', edgecolor='black') #SEM
    plt.title(f'Average Bout Duration for {video_name}')
    plt.ylabel('Time (Seconds)')
    plt.xlabel('Behavior')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{video_name}_avg_bout_duration.png"))
    plt.close()

    print(f"Plots saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Perform general behavioral analysis.")
    parser.add_argument("--output_folder", required=True, help="Path to the YOLO output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate of the video.")
    parser.add_argument("--video_name", required=True, help="Video name.")
    parser.add_argument("--min_bout_duration", type=int, default=3, help="Minimum bout duration in frames (default: 3).")
    parser.add_argument("--max_gap_duration", type=int, default=5, help="Maximum gap duration in frames (default: 5).")
    args = parser.parse_args()

    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    try:
        class_labels_dict = eval(args.class_labels)
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels: {e}")
        return
    
    #Added min_bout_duration, and max_gap_duration here
    csv_file_path = analyze_detections(args.output_folder, class_labels_dict, csv_output_folder, args.video_name, args.min_bout_duration, args.max_gap_duration, args.frame_rate)
    if csv_file_path:
        save_analysis_to_excel(csv_file_path, args.output_folder, class_labels_dict, args.frame_rate, args.video_name)
        plot_general_analysis(csv_file_path, class_labels_dict, args.frame_rate, args.output_folder, args.video_name)

if __name__ == "__main__":
    main()