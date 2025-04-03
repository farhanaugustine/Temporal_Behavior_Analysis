import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, anderson, kstest, jarque_bera
import re

def parse_peak_times(peak_times_str, max_value=2000):
    """Parses peak times from a string, handles errors, and filters by max_value."""
    if not isinstance(peak_times_str, str) or not peak_times_str.strip():
        return []

    try:
        peak_times = [float(match) for match in re.findall(r'-?\d+\.?\d*', peak_times_str)]
        filtered_peak_times = [time for time in peak_times if time <= max_value]
        return filtered_peak_times
    except Exception as e:
        print(f"  Error parsing peak times: {e}, Input: {peak_times_str}")
        return []

def analyze_rhythm_files(rhythm_files_path, output_folder, max_value=2000, dpi=600):
    """Analyzes multiple rhythm files, generates histograms and stats."""
    os.makedirs(output_folder, exist_ok=True)

    all_data = {"Exploration": [], "Jump": [], "Rear": [], "Wall-Rearing": [], "Grooming": []}
    valid_behaviors = set(all_data.keys())
    processed_peak_times = set()

    output_messages = [] # List to collect messages for logging

    for filename in os.listdir(rhythm_files_path):
        if not filename.endswith("_behavioral_rhythms.xlsx"):
            continue

        filepath = os.path.join(rhythm_files_path, filename)
        try:
            df = pd.read_excel(filepath)
            if df.empty:
                message = f"Skipping file {filename}: DataFrame is empty."
                print(message)
                output_messages.append(message)
                continue
        except (FileNotFoundError, Exception) as e:
            message = f"Error reading {filename}: {e}"
            print(message)
            output_messages.append(message)
            continue

        video_name = os.path.basename(filename).replace("_behavioral_rhythms.xlsx", "")
        try:
            df['Peak Times (seconds)'] = df['Peak Times (seconds)'].astype(str).apply(
                lambda x: parse_peak_times(x, max_value)
            )
        except KeyError:
            message = f"  'Peak Times (seconds)' column not found in {filename}. Skipping."
            print(message)
            output_messages.append(message)
            continue

        for index, row in df.iterrows():
            try:
                behavior = row['Behavior']
                peak_times = row['Peak Times (seconds)']

                if behavior not in valid_behaviors:
                    message = f"    Behavior '{behavior}' not recognized. Skipping."
                    print(message)
                    output_messages.append(message)
                    continue
                if not peak_times:
                    continue
                for time in peak_times:
                    if (behavior, time, video_name) not in processed_peak_times:
                        all_data[behavior].append(time)
                        processed_peak_times.add((behavior, time, video_name))

            except KeyError as e:
                message = f"    Missing column in row {index}: {e}. Skipping."
                print(message)
                output_messages.append(message)
            except Exception as e:
                message = f"    Error processing row {index}: {e}. Skipping."
                print(message)
                output_messages.append(message)


    hist_paths = []
    summary_paths = []
    for behavior, all_times in all_data.items():
        if not all_times:
            message = f"No data for behavior: {behavior}"
            print(message)
            output_messages.append(message)
            continue

        # Combined Histogram
        plt.figure(figsize=(12, 6))
        plt.hist(all_times, bins='auto', edgecolor='black', alpha=0.7)
        plt.xlabel("Time (seconds)",font=16)
        plt.ylabel("Frequency", font=16)
        plt.title(f"Combined Time Distribution for {behavior} (All Videos)",font=20)
        plt.grid(True, alpha=0.4)
        output_path = os.path.join(output_folder, f"{behavior}_combined_histogram.png")
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        message = f"Histogram saved: {output_path}"
        print(message)
        output_messages.append(message)
        hist_paths.append(output_path)


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

            alpha = 0.05
            f.write("Normality Tests:\n")

            if 'shapiro' in stats_summary:
                f.write("  Shapiro-Wilk Test:\n")
                f.write(f"    Statistic: {stats_summary['shapiro'][0]:.3f}\n")
                f.write(f"    P-value: {stats_summary['shapiro'][1]:.3f}\n")
                if len(all_times) > 5000:
                    f.write("    Note: Shapiro-Wilk may be unreliable for N > 5000.\n")
                if stats_summary['shapiro'][1] > alpha:
                    f.write(f"    Interpretation: Fail to reject null (normal distribution).\n")
                else:
                    f.write(f"    Interpretation: Reject null (not normally distributed).\n")

            if 'anderson' in stats_summary:
                f.write("  Anderson-Darling Test:\n")
                f.write(f"    Statistic: {stats_summary['anderson'][0]:.3f}\n")
                f.write(f"    Critical Values: {stats_summary['anderson'][1]}\n")
                f.write(f"    Significance Levels: {stats_summary['anderson'][2]}\n")
                reject = False
                for i in range(len(stats_summary['anderson'][2])):
                    if stats_summary['anderson'][0] > stats_summary['anderson'][1][i]:
                        reject = True
                    else:
                        reject = False
                        break
                if reject:
                    f.write("    Interpretation: Reject null (not normally distributed).\n")
                else:
                    f.write("    Interpretation: Fail to reject null (normal distribution).\n")

            if 'kstest' in stats_summary:
                f.write("  Kolmogorov-Smirnov Test:\n")
                f.write(f"    Statistic: {stats_summary['kstest'][0]:.3f}\n")
                f.write(f"    P-value: {stats_summary['kstest'][1]:.3f}\n")
                if stats_summary['kstest'][1] > alpha:
                    f.write(f"    Interpretation: Fail to reject null (normal distribution).\n")
                else:
                    f.write(f"    Interpretation: Reject null (not normally distributed).\n")

            if 'jarque_bera' in stats_summary:
                f.write("  Jarque-Bera Test:\n")
                f.write(f"    Statistic: {stats_summary['jarque_bera'][0]:.3f}\n")
                f.write(f"    P-value: {stats_summary['jarque_bera'][1]:.3f}\n")
                if stats_summary['jarque_bera'][1] > alpha:
                    f.write(f"    Interpretation: Fail to reject null (normal distribution).\n")
                else:
                    f.write(f"    Interpretation: Reject null (not normally distributed).\n")
            f.write("\nNote: For large datasets (N > 5000), consider Anderson-Darling or visual checks.\n")
        message = f"Summary saved: {summary_file}"
        print(message)
        output_messages.append(message)
        summary_paths.append(summary_file)


    return "\n".join(output_messages) # Combine all messages into single string


def perform_statistical_tests(data):
    """Performs Shapiro-Wilk, Anderson-Darling, KS, and Jarque-Bera tests."""
    results = {}
    if len(data) >= 3:
        # Shapiro-Wilk
        try:
            results['shapiro'] = shapiro(data)
            if len(data) > 5000:
                print(f"    Warning: Shapiro-Wilk test may be unreliable for N > 5000 (N={len(data)}).")
        except Exception as e:
            print(f"    Shapiro-Wilk error: {e}")
            results['shapiro'] = (np.nan, np.nan)

        # Anderson-Darling
        try:
            anderson_result = anderson(data)
            results['anderson'] = (anderson_result.statistic, anderson_result.critical_values, anderson_result.significance_level)
        except Exception as e:
            print(f"    Anderson-Darling error: {e}")
            results['anderson'] = (np.nan, np.nan, np.nan)

        # Kolmogorov-Smirnov (against normal distribution)
        try:
            results['kstest'] = kstest(data, 'norm')
        except Exception as e:
            print(f"    Kolmogorov-Smirnov error: {e}")
            results['kstest'] = (np.nan, np.nan)

        # Jarque-Bera
        try:
            results['jarque_bera'] = jarque_bera(data)
        except Exception as e:
            print(f"   Jarque-Bera error: {e}")
            results['jarque_bera'] = (np.nan, np.nan)
    else:
        print("    Not enough data for normality tests (n < 3).")
        results['shapiro'] = (np.nan, np.nan)
        results['anderson'] = (np.nan, np.nan, np.nan)
        results['kstest'] = (np.nan, np.nan)
        results['jarque_bera'] = (np.nan, np.nan)

    return results

def main_analysis(rhythm_folder, output_folder, max_time=2000.0, dpi=600): # Modified to main_analysis, keyword args, defaults
    """Main function to run multi-video rhythm analysis."""
    if not os.path.isdir(rhythm_folder):
        return f"Error: Rhythm folder not found: {rhythm_folder}" # Error string

    output_msg = analyze_rhythm_files(rhythm_folder, output_folder, max_time, dpi) # Call analysis function
    return output_msg # Return combined output messages


if __name__ == "__main__":
    # Example for direct testing:
    rhythm_folder_path = "path/to/your/rhythm_excel_folder"  # Replace with a real path
    output_folder_path = "path/to/your/output_folder"       # Replace with a real path
    max_time_val = 1500.0
    dpi_val = 300

    # Dummy Args class for testing
    class Args:
        def __init__(self, rhythm_folder, output_folder, max_time, dpi):
            self.rhythm_folder = rhythm_folder
            self.output_folder = output_folder
            self.max_time = max_time
            self.dpi = dpi

    test_args = Args(rhythm_folder_path, output_folder_path, max_time_val, dpi_val)

    # Simulate command-line execution (for original main) - not needed for GUI
    # main(test_args) 

    # Call main_analysis directly with keyword arguments for testing:
    output_message = main_analysis(rhythm_folder=test_args.rhythm_folder, 
                                  output_folder=test_args.output_folder, 
                                  max_time=test_args.max_time, 
                                  dpi=test_args.dpi)
    print(output_message) # Print output for direct testing