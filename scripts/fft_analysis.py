import os
import numpy as np
import pandas as pd
from scipy.fft import fft
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import glob
import shutil  # For copying files

def calculate_fft_analysis(csv_file_path, class_labels, frame_rate=30):
    """Analyzes frequency components using FFT (single video)."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    fft_results = {}
    for class_label in class_labels.values():
        signal = np.array([1 if label == class_label else 0 for label in df['Class Label']])
        N = len(signal)
        if N == 0:  # Handle empty signals
            fft_results[class_label] = (0, 0)
            continue
        yf = fft(signal)
        xf = np.fft.fftfreq(N, 1 / frame_rate)
        positive_frequencies = xf[xf >= 0]
        positive_magnitudes = np.abs(yf[xf >= 0])
        if len(positive_frequencies) > 0:
           peak_index = np.argmax(positive_magnitudes[1:]) + 1
           dominant_frequency = positive_frequencies[peak_index]
           dominant_power = positive_magnitudes[peak_index]
        else:
            dominant_frequency = 0
            dominant_power = 0

        fft_results[class_label] = (dominant_frequency, dominant_power)
    return fft_results

def save_fft_to_excel(fft_results, output_folder, video_name):
    """Saves FFT results to Excel."""
    if fft_results:
      excel_path = os.path.join(output_folder, f"{video_name}_fft_analysis.xlsx")
      try:
          rows = []
          for class_label, (dominant_frequency, dominant_power) in fft_results.items():
              rows.append({"Behavior": class_label, "Dominant Frequency (Hz)": dominant_frequency, "Dominant Power": dominant_power})
          df_fft = pd.DataFrame(rows)
          df_fft.to_excel(excel_path, sheet_name="FFT Analysis", index=False)
          print(f"FFT analysis saved to: {excel_path}")
      except Exception as e:
          print(f"Error saving FFT analysis to Excel: {e}")

def create_fft_plots(filepath, output_folder):
    """Reads FFT data from Excel, generates individual plots, and prepares data for combined analysis."""
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except Exception as e:
        print(f"Error reading excel file: {e}")
        return None  # Return None on error

    filename = os.path.splitext(os.path.basename(filepath))[0]
    # Get video name
    video_name = filename.replace("_fft_analysis", "")
    video_columns = df.columns[1:].tolist() #This is for cases where there is information from more than one video in the excel file, which is not the case now.
    behaviors = df['Behavior'].unique().tolist()

    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare data for combined analysis. This is key.
    combined_data = {}
    #In this case we have only one video
    combined_data[video_name] = {'frequencies': [], 'powers': []}
    for behavior in behaviors:
        # find the row that correspond to the given behavior
        filtered_df = df[(df['Behavior'] == behavior)]
        
        if not filtered_df.empty:
            # Access values directly, no need for .values[0]
            frequency = filtered_df['Dominant Frequency (Hz)'].iloc[0] if 'Dominant Frequency (Hz)' in filtered_df.columns else None #iloc added
            power = filtered_df['Dominant Power'].iloc[0] if 'Dominant Power' in filtered_df.columns else None #iloc added
            #check not nan before adding:
            if pd.notna(frequency):
                    combined_data[video_name]['frequencies'].append(frequency)
            if pd.notna(power):
                combined_data[video_name]['powers'].append(power)

        #individual plots.
        filtered_df = df[df['Behavior'] == behavior]
        if not filtered_df.empty:
            frequencies = filtered_df['Dominant Frequency (Hz)'].values.astype(float)
            powers = filtered_df['Dominant Power'].values.astype(float)

            bar_width = 0.4
            x = np.arange(1) # x will have one element

            plt.figure(figsize=(8, 6)) #Adjusted figure size
            plt.bar(x, frequencies, bar_width, edgecolor='black', alpha=0.7, color='skyblue')
            plt.xticks(x, [video_name], rotation=45, ha='right')  # Use video_name
            plt.xlabel('Video')
            plt.ylabel('Dominant Frequency (Hz)')
            plt.title(f'Dominant Frequency for {behavior} in {video_name}') #added filename
            plt.grid(True, axis='y', alpha=0.5)
            plt.tight_layout()

            output_path = os.path.join(output_folder, f"{video_name}_{behavior}_frequency_plot.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved for Frequency of {behavior} in {video_name}")

            plt.figure(figsize=(8, 6))  #Adjusted figure size
            plt.bar(x, powers, bar_width, edgecolor='black', alpha=0.7, color='lightcoral')
            plt.xticks(x, [video_name], rotation=45, ha='right')  # Use video_name
            plt.xlabel('Video')
            plt.ylabel('Dominant Power')
            plt.title(f'Dominant Power for {behavior} in {video_name}')#added filename
            plt.grid(True, axis='y', alpha=0.5)
            plt.tight_layout()

            output_path = os.path.join(output_folder, f"{video_name}_{behavior}_power_plot.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved for Power of {behavior} in {video_name}")

    return combined_data


def create_combined_fft_analysis(all_data, output_folder):
    """Performs combined FFT analysis and statistical tests."""

    # --- Statistical Analysis (Kruskal-Wallis) ---
    frequency_groups = []
    power_groups = []
    video_names = []

    # Restructure data for Kruskal-Wallis: Each *video* is a group.
    for video, data in all_data.items():
        video_names.append(video)  # Keep track of video names
        frequency_groups.append(data['frequencies'])
        power_groups.append(data['powers'])
    
    # --- Perform Kruskal-Wallis Test ---
    kruskal_frequency_results = None
    kruskal_power_results = None
    try:
        if len(frequency_groups) > 1 and all(len(group) > 0 for group in frequency_groups):  # Check for enough groups and non-empty
            kruskal_frequency_results = kruskal(*frequency_groups)
    except ValueError as e:
        print(f"Error in Kruskal-Wallis (frequency): {e}.  Likely, all values within a group are identical.")
    try:
      if len(power_groups) > 1 and all(len(group) > 0 for group in power_groups):  # Check for enough groups and non-empty groups
        kruskal_power_results = kruskal(*power_groups)
    except ValueError as e:
        print(f"Error in Kruskal-Wallis (power): {e}.  Likely, all values within a group are identical.")

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
            if kruskal_frequency_results.pvalue < 0.05: # added
                f.write("    Significant difference found between videos.\n")
            else:
                f.write("    No significant difference found between videos.\n")
        else:
            f.write("  Kruskal-Wallis Test: Not performed (insufficient or invalid data).\n")

        f.write("\nDominant Power Analysis:\n")
        if kruskal_power_results:
            f.write(f"  Kruskal-Wallis Test:\n")
            f.write(f"    H-statistic: {kruskal_power_results.statistic:.2f}\n")
            f.write(f"    p-value: {kruskal_power_results.pvalue:.3f}\n")
            if kruskal_power_results.pvalue < 0.05: # added
                f.write("    Significant difference found between videos.\n")
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
    parser = argparse.ArgumentParser(description="Perform FFT analysis and generate plots.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.")
    parser.add_argument("--video_name", required=True, help="Video name.")
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

    # Single-video analysis
    fft_results = calculate_fft_analysis(csv_file_path, class_labels_dict, args.frame_rate)
    if fft_results:
        save_fft_to_excel(fft_results, args.output_folder, args.video_name)

    # Prepare for multi-video analysis
    excel_output_folder = os.path.join(args.output_folder, "fft_excel")
    os.makedirs(excel_output_folder, exist_ok=True)
    source_excel_path = os.path.join(args.output_folder, f"{args.video_name}_fft_analysis.xlsx")
    destination_excel_path = os.path.join(excel_output_folder, f"{args.video_name}_fft_analysis.xlsx")

    if os.path.exists(source_excel_path):
        try:
            shutil.copy(source_excel_path, destination_excel_path)
            print(f"Copied FFT Excel file to: {destination_excel_path}")
        except Exception as e:
            print(f"Error copying file: {e}")

        # Perform multi-video analysis and plotting
        all_data = {}
        combined_file_data = create_fft_plots(destination_excel_path, args.output_folder)
        if combined_file_data:
              all_data.update(combined_file_data)
        create_combined_fft_analysis(all_data, args.output_folder)

    else:
        print(f"Error: {source_excel_path} not found.")

if __name__ == "__main__":
    main()