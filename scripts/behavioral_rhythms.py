import os
import numpy as np
import pandas as pd
import argparse
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import re
import shutil  # For copying files

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

def parse_peak_times(peak_times_str, max_value=1500):
    """Parses peak times from a string, handles errors."""
    if not isinstance(peak_times_str, str) or not peak_times_str.strip():
        return []
    try:
        cleaned_str = re.sub(r'(np\.array|array|\(|\)|np\.float64|\[|\]|\s+)', '', peak_times_str)
        if not cleaned_str:
            return []
        peak_times = []
        for part in cleaned_str.split(','):
            part = part.strip()
            if part:
                try:
                    peak_times.append(float(part))
                except ValueError:
                    print(f"  Invalid float value: '{part}'")
        filtered_peak_times = [time for time in peak_times if time <= max_value]
        return filtered_peak_times
    except Exception as e:
        print(f"  Error parsing peak times: {e}, Input: {peak_times_str}")
        return []


def analyze_rhythm_files(rhythm_files_path, output_folder, max_value=1500):
    """Analyzes multiple rhythm files, generates histograms and stats."""
    os.makedirs(output_folder, exist_ok=True)

    all_data = {"Exploration": [], "Jump": [], "Rearing": [], "WallHugging": [], "Grooming": []}  # Use your actual behaviors
    valid_behaviors = set(all_data.keys())
    processed_peak_times = set()  # Track (behavior, time) for duplicates

    for filename in os.listdir(rhythm_files_path):
        if not filename.endswith("_rhythms.xlsx"):
            continue

        filepath = os.path.join(rhythm_files_path, filename)
        try:
            df = pd.read_excel(filepath)
        except (FileNotFoundError, Exception) as e:
            print(f"Error reading {filename}: {e}")
            continue

        video_name = os.path.basename(filename).replace("_rhythms.xlsx", "")
        try:
            df['Peak Times (seconds)'] = df['Peak Times (seconds)'].astype(str).apply(
                lambda x: parse_peak_times(x, max_value)
            )
        except KeyError:
            print(f"  'Peak Times (seconds)' column not found in {filename}. Skipping.")
            continue

        for index, row in df.iterrows():
            try:
                behavior = row['Behavior']
                peak_times = row['Peak Times (seconds)']

                if behavior not in valid_behaviors:
                    print(f"    Behavior '{behavior}' not recognized. Skipping.")
                    continue
                if not peak_times:
                    continue
                for time in peak_times:
                    if (behavior, time) not in processed_peak_times:
                        all_data[behavior].append(time)
                        processed_peak_times.add((behavior, time))
            except KeyError as e:
                print(f"    Missing column in row {index}: {e}. Skipping.")
            except Exception as e:
                print(f"    Error processing row {index}: {e}. Skipping.")

    for behavior, all_times in all_data.items():
        if not all_times:
            print(f"No data for behavior: {behavior}")
            continue

        # Combined Histogram
        plt.figure(figsize=(12, 6))
        plt.hist(all_times, bins='auto', edgecolor='black', alpha=0.7)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.title(f"Combined Time Distribution for {behavior} (All Videos)")
        plt.grid(True)
        output_path = os.path.join(output_folder, f"{behavior}_combined_histogram.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Histogram saved: {output_path}")

        # Statistical Summary
        stats_summary = perform_statistical_tests(all_times)
        summary_file = os.path.join(output_folder, f"{behavior}_statistical_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"Statistical Summary for {behavior}\n\n")
            f.write(f"Count: {len(all_times)}\n")
            f.write(f"Mean: {np.mean(all_times):.2f}\n")
            f.write(f"Median: {np.median(all_times):.2f}\n")
            f.write(f"Std Dev: {np.std(all_times):.2f}\n")
            f.write(f"Min: {np.min(all_times):.2f}\n")
            f.write(f"Max: {np.max(all_times):.2f}\n\n")
            f.write("Shapiro-Wilk Normality Test:\n")
            f.write(f"  Statistic: {stats_summary['shapiro'][0]:.3f}\n")
            f.write(f"  P-value: {stats_summary['shapiro'][1]:.3f}\n")
            if stats_summary['shapiro'][1] > 0.05:
                f.write("  Normally distributed (fail to reject H0).\n")
            else:
                f.write("  Not normally distributed (reject H0).\n")
        print(f"Summary saved: {summary_file}")

def perform_statistical_tests(data):
    """Performs Shapiro-Wilk normality test."""
    results = {}
    if len(data) >= 3:
        try:
            results['shapiro'] = shapiro(data)
        except Exception as e:
            print(f"    Shapiro-Wilk error: {e}")
            results['shapiro'] = (np.nan, np.nan)
    else:
        print("    Not enough data for Shapiro-Wilk (n < 3).")
        results['shapiro'] = (np.nan, np.nan)
    return results

def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Detect behavioral rhythms and perform analysis.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.")
    parser.add_argument("--video_name", required=True, help="Video name.")
    parser.add_argument("--prominence", type=float, default=1.0, help="Prominence for peak detection.")
    parser.add_argument("--max_time", type=float, default=1500.0, help="Maximum time value to consider.")
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

    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found. Run general_analysis.py first.")
        return

    # Single-video rhythm detection
    rhythm_results = calculate_behavioral_rhythms(csv_file_path, class_labels_dict, args.frame_rate, args.prominence)
    if rhythm_results:
        save_rhythms_to_excel(rhythm_results, args.output_folder, args.video_name)
    
    # Create a dedicated folder for combined rhythm analysis
    rhythm_excel_folder = os.path.join(args.output_folder, "rhythm_excel")
    os.makedirs(rhythm_excel_folder, exist_ok=True)

    # Copy the Excel file to the dedicated folder
    source_excel_path = os.path.join(args.output_folder, f"{args.video_name}_behavioral_rhythms.xlsx")
    destination_excel_path = os.path.join(rhythm_excel_folder, f"{args.video_name}_behavioral_rhythms.xlsx")
    if os.path.exists(source_excel_path):
        try:
            shutil.copy(source_excel_path, destination_excel_path)
            print(f"Copied rhythm Excel file to: {destination_excel_path}")
        except Exception as e:
            print(f"Error copying file: {e}")

            # Multi-video analysis (if other files are present)
            analyze_rhythm_files(rhythm_excel_folder, args.output_folder, args.max_time)

if __name__ == "__main__":
    main()