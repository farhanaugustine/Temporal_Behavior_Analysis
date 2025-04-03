import os
import numpy as np
import pandas as pd
from scipy.fft import fft
import matplotlib.pyplot as plt
import ast  

def calculate_fft_analysis(csv_file_path, class_labels, frame_rate=30):
    """Analyzes frequency components using FFT (single video)."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return "Error: CSV file not found or empty", None # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}", None # Return error string

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
            peak_index = np.argmax(positive_magnitudes[1:]) + 1  # Find peak (excluding 0 Hz)
            dominant_frequency = positive_frequencies[peak_index]
            dominant_power = positive_magnitudes[peak_index]
        else:
            dominant_frequency = 0
            dominant_power = 0

        fft_results[class_label] = (dominant_frequency, dominant_power)
    return fft_results, None # Return fft_results, and None for error string


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
            return f"FFT analysis saved to: {excel_path}" # Return success message
        except Exception as e:
            return f"Error saving FFT analysis to Excel: {e}" # Return error message
    return "No FFT data to save." # Return if no data


def create_fft_plots(excel_file_path, output_folder): # Changed arg name for clarity
    """Reads FFT data from Excel, generates individual plots."""
    try:
        df = pd.read_excel(excel_file_path, engine='openpyxl') # Use excel_file_path
    except Exception as e:
        return f"Error reading excel file: {e}", []  # Return error string and empty list

    filename = os.path.splitext(os.path.basename(excel_file_path))[0] # Use excel_file_path
    video_name = filename.replace("_fft_analysis", "")
    behaviors = df['Behavior'].unique().tolist()

    plot_paths = [] # List to collect plot paths

    for behavior in behaviors:
        filtered_df = df[df['Behavior'] == behavior]
        if not filtered_df.empty:
            frequencies = filtered_df['Dominant Frequency (Hz)'].values.astype(float)
            powers = filtered_df['Dominant Power'].values.astype(float)

            bar_width = 0.4
            x = np.arange(1)

            # Frequency plot
            plt.figure(figsize=(8, 6))
            plt.bar(x, frequencies, bar_width, edgecolor='black', alpha=0.7, color='skyblue')
            plt.xticks(x, [video_name], rotation=45, ha='right')
            plt.xlabel('Video')
            plt.ylabel('Dominant Frequency (Hz)')
            plt.title(f'Dominant Frequency for {behavior} in {video_name}')
            plt.grid(True, axis='y', alpha=0.5)
            plt.tight_layout()

            frequency_plot_path = os.path.join(output_folder, f"{video_name}_{behavior}_frequency_plot.png")
            plt.savefig(frequency_plot_path)
            plt.close()
            message_freq_plot = f"Plot saved for Frequency of {behavior} in {video_name}"
            print(message_freq_plot)
            plot_paths.append(frequency_plot_path)

            # Power plot
            plt.figure(figsize=(8, 6))
            plt.bar(x, powers, bar_width, edgecolor='black', alpha=0.7, color='lightcoral')
            plt.xticks(x, [video_name], rotation=45, ha='right')
            plt.xlabel('Video')
            plt.ylabel('Dominant Power')
            plt.title(f'Dominant Power for {behavior} in {video_name}')
            plt.grid(True, axis='y', alpha=0.5)
            plt.tight_layout()

            power_plot_path = os.path.join(output_folder, f"{video_name}_{behavior}_power_plot.png")
            plt.savefig(power_plot_path)
            plt.close()
            message_power_plot = f"Plot saved for Power of {behavior} in {video_name}"
            print(message_power_plot)
            plot_paths.append(power_plot_path)

    return None, plot_paths # No error string if successful, return plot paths


def main_analysis(output_folder, class_labels, frame_rate, video_name): # Keyword args
    """Main function to run single-video FFT analysis."""

    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Return error string

    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        return f"Error: {csv_file_path} not found. Run general_analysis.py first." # Return error string

    fft_results, fft_error = calculate_fft_analysis(csv_file_path, class_labels, frame_rate) # Capture potential error
    if fft_error: # If calculate_fft_analysis returned an error string
        return fft_error # Return error string to GUI

    excel_output_msg = save_fft_to_excel(fft_results, output_folder, video_name)
    plot_error, plot_paths = create_fft_plots(os.path.join(output_folder, f"{video_name}_fft_analysis.xlsx"), output_folder) # Capture plot paths and error

    output_messages = [msg for msg in [excel_output_msg] if msg is not None] # Start with excel message
    plot_output_messages = [f"Plot saved: {path}" for path in plot_paths] # Format plot paths to messages
    output_messages.extend(plot_output_messages) # Add plot messages

    if plot_error: # If create_fft_plots returned an error string
        output_messages.insert(0, plot_error) # Prepend plot error to output messages


    return "\n".join(output_messages) if output_messages else "FFT analysis completed. No specific output messages."


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