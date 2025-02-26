import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.stats import shapiro, anderson, kstest, jarque_bera
import re

def parse_peak_times(peak_times_str, max_value=2000):
    """Parses peak times from a string, handles errors, and filters by max_value."""
    if not isinstance(peak_times_str, str) or not peak_times_str.strip():
        return []

    try:
        # Extract all floating-point numbers from the string
        peak_times = [float(match) for match in re.findall(r'-?\d+\.?\d*', peak_times_str)]
        # Filter by max_value
        filtered_peak_times = [time for time in peak_times if time <= max_value]
        return filtered_peak_times
    except Exception as e:
        print(f"  Error parsing peak times: {e}, Input: {peak_times_str}")
        return []

def analyze_rhythm_files(rhythm_files_path, output_folder, max_value=2000, dpi=600):  # Add dpi argument
    """Analyzes multiple rhythm files, generates histograms and stats."""
    os.makedirs(output_folder, exist_ok=True)

    all_data = {"Exploration": [], "Jump": [], "Rear": [], "Wall-Rearing": [], "Grooming": []}  # Use your actual behaviors
    valid_behaviors = set(all_data.keys())
    processed_peak_times = set()  # Track (behavior, time, video) for duplicates

    for filename in os.listdir(rhythm_files_path):
        if not filename.endswith("_behavioral_rhythms.xlsx"):
            continue

        filepath = os.path.join(rhythm_files_path, filename)
        try:
            df = pd.read_excel(filepath)
             # --- Check for empty DataFrame ---
            if df.empty:
                print(f"Skipping file {filename}: DataFrame is empty.")
                continue
        except (FileNotFoundError, Exception) as e:
            print(f"Error reading {filename}: {e}")
            continue

        video_name = os.path.basename(filename).replace("_behavioral_rhythms.xlsx", "") #Get video name.
        try:
            df['Peak Times (seconds)'] = df['Peak Times (seconds)'].astype(str).apply(
                lambda x: parse_peak_times(x, max_value) #Use max_value here
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
                    if (behavior, time, video_name) not in processed_peak_times: # Include video_name
                        all_data[behavior].append(time)
                        processed_peak_times.add((behavior, time, video_name)) # Include video_name

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
        plt.grid(True, alpha=0.4)  # Set grid alpha here
        output_path = os.path.join(output_folder, f"{behavior}_combined_histogram.png")
        plt.savefig(output_path, dpi=dpi)  # Save with specified DPI
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

            # --- Normality Test Results (All Tests) WITH INTERPRETATION ---
            f.write("Normality Tests:\n")
            alpha = 0.05 # Set alpha

            if 'shapiro' in stats_summary:
                f.write("  Shapiro-Wilk Test:\n")
                f.write(f"    Statistic: {stats_summary['shapiro'][0]:.3f}\n")
                f.write(f"    P-value: {stats_summary['shapiro'][1]:.3f}\n")
                if len(all_times) > 5000:
                    f.write("    Note: Shapiro-Wilk may be unreliable for N > 5000.\n")
                if stats_summary['shapiro'][1] > alpha:
                    f.write(f"    Interpretation: Fail to reject the null hypothesis (data may be normally distributed).\n")
                else:
                    f.write(f"    Interpretation: Reject the null hypothesis (data is likely not normally distributed).\n")


            if 'anderson' in stats_summary:
                f.write("  Anderson-Darling Test:\n")
                f.write(f"    Statistic: {stats_summary['anderson'][0]:.3f}\n")
                f.write(f"    Critical Values: {stats_summary['anderson'][1]}\n")
                f.write(f"    Significance Levels: {stats_summary['anderson'][2]}\n")
                # Find the most appropriate significance level
                reject = False
                for i in range(len(stats_summary['anderson'][2])):
                    if stats_summary['anderson'][0] > stats_summary['anderson'][1][i]:
                        reject = True
                    else:
                        reject = False
                        break # Stop once we do not reject.
                if reject:
                    f.write("    Interpretation: Reject the null hypothesis (data is likely not normally distributed).\n")
                else:
                    f.write("    Interpretation: Fail to reject the null hypothesis (data may be normally distributed).\n")

            if 'kstest' in stats_summary:
                f.write("  Kolmogorov-Smirnov Test:\n")
                f.write(f"    Statistic: {stats_summary['kstest'][0]:.3f}\n")
                f.write(f"    P-value: {stats_summary['kstest'][1]:.3f}\n")
                if stats_summary['kstest'][1] > alpha:
                    f.write(f"    Interpretation: Fail to reject the null hypothesis (data may be normally distributed).\n")
                else:
                    f.write(f"    Interpretation: Reject the null hypothesis (data is likely not normally distributed).\n")

            if 'jarque_bera' in stats_summary:
                f.write("  Jarque-Bera Test:\n")
                f.write(f"    Statistic: {stats_summary['jarque_bera'][0]:.3f}\n")
                f.write(f"    P-value: {stats_summary['jarque_bera'][1]:.3f}\n")
                if stats_summary['jarque_bera'][1] > alpha:
                    f.write(f"    Interpretation: Fail to reject the null hypothesis (data may be normally distributed).\n")
                else:
                    f.write(f"    Interpretation: Reject the null hypothesis (data is likely not normally distributed).\n")
            f.write("\nRecommendation: For large sample sizes (N > 5000), consider the Anderson-Darling test or visual inspection (histogram, Q-Q plot) in addition to Shapiro-Wilk.\n")


        print(f"Summary saved: {summary_file}")


def perform_statistical_tests(data):
    """Performs Shapiro-Wilk, Anderson-Darling, KS, and Jarque-Bera tests."""
    results = {}
    if len(data) >= 3:
        # Shapiro-Wilk
        try:
            results['shapiro'] = shapiro(data)
            if len(data) > 5000:
                print(f"    Warning: Shapiro-Wilk test may be unreliable for sample sizes > 5000 (N={len(data)}).")
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
            results['kstest'] = kstest(data, 'norm')  # Compare against normal dist.
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

def main():
    """Main function to parse arguments and run multi-video analysis."""
    parser = argparse.ArgumentParser(description="Perform aggregated behavioral rhythm analysis across multiple videos.")
    parser.add_argument("--rhythm_folder", required=True, help="Path to the folder containing rhythm Excel files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for aggregated results.")
    parser.add_argument("--max_time", type=float, default=2000.0, help="Maximum time value to consider.")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figures (default: 600).")  # Add DPI argument
    args = parser.parse_args()

    analyze_rhythm_files(args.rhythm_folder, args.output_folder, args.max_time, args.dpi) # Pass max_time and dpi

if __name__ == "__main__":
    main()