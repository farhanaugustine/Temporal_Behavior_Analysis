import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.stats import kruskal

def create_combined_fft_analysis(all_data, output_folder):
    """Performs combined FFT analysis and statistical tests."""

    frequency_groups = []
    power_groups = []
    video_names = []

    for video, data in all_data.items():
        video_names.append(video)
        frequency_groups.append(data['frequencies'])
        power_groups.append(data['powers'])

    # --- Perform Kruskal-Wallis Test (with checks) ---
    kruskal_frequency_results = None
    kruskal_power_results = None

    # Check for sufficient data and variability *before* calling kruskal
    if len(frequency_groups) > 1 and all(len(group) > 0 for group in frequency_groups) and any(len(set(group)) > 1 for group in frequency_groups):
        try:
            kruskal_frequency_results = kruskal(*frequency_groups)
        except ValueError as e:
            print(f"Error in Kruskal-Wallis (frequency): {e}")
    else:
        print("Insufficient or non-variable data for Kruskal-Wallis test on frequency.")

    if len(power_groups) > 1 and all(len(group) > 0 for group in power_groups) and any(len(set(group)) > 1 for group in power_groups):
        try:
            kruskal_power_results = kruskal(*power_groups)
        except ValueError as e:
            print(f"Error in Kruskal-Wallis (power): {e}")
    else:
        print("Insufficient or non-variable data for Kruskal-Wallis test on power.")
    # --- Plotting (Combined Bar Plots)---
    #Prepare Data for Plotting
    freq_means = [np.mean(group) if len(group) > 0 else np.nan for group in frequency_groups]
    power_means = [np.mean(group) if len(group) > 0 else np.nan for group in power_groups]

    # Frequency
    plt.figure(figsize=(12, 6))
    plt.bar(video_names, freq_means, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Video')
    plt.ylabel('Mean Dominant Frequency (Hz)')
    plt.title('Combined Dominant Frequency Across Videos')
    plt.grid(True, axis='y', alpha=0.5)

    if kruskal_frequency_results:
        plt.text(0.05, 0.95, f"Kruskal-Wallis: H={kruskal_frequency_results.statistic:.2f}, p={kruskal_frequency_results.pvalue:.3f}",
                 transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(output_folder, "combined_frequency_plot.png")
    plt.savefig(output_path)
    plt.close()
    print("Combined frequency plot saved.")

    # Power
    plt.figure(figsize=(12, 6))
    plt.bar(video_names, power_means, edgecolor='black', alpha=0.7, color='lightcoral')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Video')
    plt.ylabel('Mean Dominant Power')
    plt.title('Combined Dominant Power Across Videos')
    plt.grid(True, axis='y', alpha=0.5)

    if kruskal_power_results:
        plt.text(0.05, 0.95, f"Kruskal-Wallis: H={kruskal_power_results.statistic:.2f}, p={kruskal_power_results.pvalue:.3f}",
                 transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(output_folder, "combined_power_plot.png")
    plt.savefig(output_path)
    plt.close()
    print("Combined power plot saved.")

    # --- Statistical Summary ---
    with open(os.path.join(output_folder, "statistical_summary.txt"), "w") as f:
        f.write("Statistical Summary (All Behaviors Combined)\n\n")

        f.write("Dominant Frequency Analysis:\n")
        if kruskal_frequency_results:
            f.write(f"  Kruskal-Wallis Test:\n")
            f.write(f"    H-statistic: {kruskal_frequency_results.statistic:.2f}\n")
            f.write(f"    p-value: {kruskal_frequency_results.pvalue:.3f}\n")
            if kruskal_frequency_results.pvalue < 0.05:
                f.write("    Significant difference found between videos (alpha=0.05).\n") # Clarified alpha
            else:
                f.write("    No significant difference found between videos.\n")
        else:
            f.write("  Kruskal-Wallis Test: Not performed (insufficient or invalid data).\n")

        f.write("\nDominant Power Analysis:\n")
        if kruskal_power_results:
            f.write(f"  Kruskal-Wallis Test:\n")
            f.write(f"    H-statistic: {kruskal_power_results.statistic:.2f}\n")
            f.write(f"    p-value: {kruskal_power_results.pvalue:.3f}\n")
            if kruskal_power_results.pvalue < 0.05:
                f.write("    Significant difference found between videos (alpha=0.05).\n") # Clarified alpha
            else:
                f.write("    No significant difference found between videos.\n")
        else:
            f.write("  Kruskal-Wallis Test: Not performed (insufficient or invalid data).\n")

       # Descriptive stats
        f.write("\nDescriptive Statistics (Dominant Frequency):\n")
        freq_all = [val for sublist in frequency_groups for val in sublist]  # Flatten the list
        if freq_all:  # Check if not empty
          f.write(pd.Series(freq_all).describe().to_string() + "\n")
        else:
          f.write("No frequency data available.\n")

        f.write("\nDescriptive Statistics (Dominant Power):\n")
        power_all = [val for sublist in power_groups for val in sublist]  # Flatten the list
        if power_all:  # Check if not empty
            f.write(pd.Series(power_all).describe().to_string() + "\n")
        else:
            f.write("No power data available.\n")

def main():
    parser = argparse.ArgumentParser(description="Perform aggregated FFT analysis across multiple videos.")
    parser.add_argument("--fft_folder", required=True, help="Path to the folder containing FFT Excel files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for aggregated results.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    all_data = {}
    for filename in os.listdir(args.fft_folder):
        if filename.endswith("_fft_analysis.xlsx"):
            filepath = os.path.join(args.fft_folder, filename)
            try:
                df = pd.read_excel(filepath)
                video_name = filename.replace("_fft_analysis.xlsx", "")
                all_data[video_name] = {'frequencies': [], 'powers': []}

                for behavior in df['Behavior'].unique():
                    behavior_data = df[df['Behavior'] == behavior]
                    if not behavior_data.empty:
                        frequency = behavior_data['Dominant Frequency (Hz)'].iloc[0]
                        power = behavior_data['Dominant Power'].iloc[0]
                        if pd.notna(frequency):
                            all_data[video_name]['frequencies'].append(frequency)
                        if pd.notna(power):
                            all_data[video_name]['powers'].append(power)
            except (FileNotFoundError, Exception) as e:  # Catch FileNotFoundError explicitly
                print(f"Error processing {filename}: {e}")
                continue

    create_combined_fft_analysis(all_data, args.output_folder)

if __name__ == "__main__":
    main()