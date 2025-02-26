import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import re

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

    all_data = {"Exploration": [], "Jump": [], "Rear": [], "Wall-Rearing": [], "Grooming": []}  # Use your actual behaviors
    valid_behaviors = set(all_data.keys())
    processed_peak_times = set()  # Track (behavior, time) for duplicates

    for filename in os.listdir(rhythm_files_path):
        if not filename.endswith("_behavioral_rhythms.xlsx"):
            continue

        filepath = os.path.join(rhythm_files_path, filename)
        try:
            df = pd.read_excel(filepath)
        except (FileNotFoundError, Exception) as e:
            print(f"Error reading {filename}: {e}")
            continue

        video_name = os.path.basename(filename).replace("_behavioral_rhythms.xlsx", "")
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
    """Main function to parse arguments and run multi-video analysis."""
    parser = argparse.ArgumentParser(description="Perform aggregated behavioral rhythm analysis across multiple videos.")
    parser.add_argument("--rhythm_folder", required=True, help="Path to the folder containing rhythm Excel files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for aggregated results.")
    parser.add_argument("--max_time", type=float, default=1500.0, help="Maximum time value to consider.")
    args = parser.parse_args()

    analyze_rhythm_files(args.rhythm_folder, args.output_folder, args.max_time)

if __name__ == "__main__":
    main()