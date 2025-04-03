import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal

def create_combined_fft_analysis(all_data, output_folder):
    """Performs combined FFT analysis and statistical tests."""

    frequency_groups = []
    power_groups = []
    video_names = []

    output_messages = [] # List to collect messages

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
            message_kruskal_freq_error = f"Error in Kruskal-Wallis (frequency): {e}"
            print(message_kruskal_freq_error)
            output_messages.append(message_kruskal_freq_error)
    else:
        message_insufficient_freq_data = "Insufficient or non-variable data for Kruskal-Wallis test on frequency."
        print(message_insufficient_freq_data)
        output_messages.append(message_insufficient_freq_data)

    if len(power_groups) > 1 and all(len(group) > 0 for group in power_groups) and any(len(set(group)) > 1 for group in power_groups):
        try:
            kruskal_power_results = kruskal(*power_groups)
        except ValueError as e:
            message_kruskal_power_error = f"Error in Kruskal-Wallis (power): {e}"
            print(message_kruskal_power_error)
            output_messages.append(message_kruskal_power_error)
    else:
        message_insufficient_power_data = "Insufficient or non-variable data for Kruskal-Wallis test on power."
        print(message_insufficient_power_data)
        output_messages.append(message_insufficient_power_data)

    # --- Plotting (Combined Bar Plots)---
    freq_means = [np.mean(group) if len(group) > 0 else np.nan for group in frequency_groups]
    power_means = [np.mean(group) if len(group) > 0 else np.nan for group in power_groups]

    # Frequency plot
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
    frequency_plot_path = os.path.join(output_folder, "combined_frequency_plot.png")
    plt.savefig(frequency_plot_path)
    plt.close()
    message_freq_plot_saved = "Combined frequency plot saved."
    print(message_freq_plot_saved)
    output_messages.append(message_freq_plot_saved)


    # Power plot
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
    power_plot_path = os.path.join(output_folder, "combined_power_plot.png")
    plt.savefig(power_plot_path)
    plt.close()
    message_power_plot_saved = "Combined power plot saved."
    print(message_power_plot_saved)
    output_messages.append(message_power_plot_saved)

    # --- Statistical Summary ---
    summary_file_path = os.path.join(output_folder, "statistical_summary.txt")
    with open(summary_file_path, "w") as f:
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
    message_stats_saved = f"Statistical summary saved to: {summary_file_path}"
    print(message_stats_saved)
    output_messages.append(message_stats_saved)

    return output_messages # Return all messages


def main_analysis(fft_folder, output_folder): # Keyword arguments
    """Main function to run aggregated FFT analysis across multiple videos."""

    if not os.path.isdir(fft_folder):
        return f"Error: FFT folder not found: {fft_folder}" # Return error string

    os.makedirs(output_folder, exist_ok=True)

    all_data = {}
    output_messages = [] # Collect messages from file processing

    for filename in os.listdir(fft_folder):
        if filename.endswith("_fft_analysis.xlsx"):
            filepath = os.path.join(fft_folder, filename)
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
                message_file_error = f"Error processing {filename}: {e}"
                print(message_file_error)
                output_messages.append(message_file_error)
                continue

    analysis_messages = create_combined_fft_analysis(all_data, output_folder) # Get messages from analysis
    output_messages.extend(analysis_messages) # Extend main messages list

    return "\n".join(output_messages) # Return combined messages


if __name__ == "__main__":
    # Example for direct testing:
    fft_folder_path = "path/to/your/fft_excel_folder" # Replace
    output_folder_path = "path/to/your/output_folder" # Replace

    class Args: # Dummy Args class for testing (not needed for GUI)
        def __init__(self, fft_folder, output_folder):
            self.fft_folder = fft_folder
            self.output_folder = output_folder

    test_args = Args(fft_folder_path, output_folder_path)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(fft_folder=test_args.fft_folder, 
                                  output_folder=test_args.output_folder)
    print(output_message) # Print output for direct test