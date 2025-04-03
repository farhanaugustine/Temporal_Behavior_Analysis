import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import ast  

def calculate_time_lagged_cross_correlation(csv_file_path, class_labels, max_lag_frames=150):
    """Calculates time-lagged cross-correlation, returns data for plotting."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return "Error: CSV file not found or empty", None, None, None # Return error string
    except Exception as e:
        return f"Error reading CSV: {e}", None, None, None # Return error string

    all_labels = df['Class Label'].tolist()
    cross_correlations = {}

    for (class1, class2) in itertools.combinations(class_labels.values(), 2):
        signal1 = np.array([1 if label == class1 else 0 for label in all_labels])
        signal2 = np.array([1 if label == class2 else 0 for label in all_labels])
        lags = np.arange(-max_lag_frames, max_lag_frames + 1)
        xcorr = []

        for lag in lags:
            if (len(signal1) - abs(lag) > 0 and len(signal2) - abs(lag) > 0):
                if np.std(signal1[:len(signal1) - abs(lag)]) > 0 and np.std(signal2[abs(lag):]) > 0:
                    xcorr.append(np.corrcoef(signal1[:len(signal1) - abs(lag)], signal2[abs(lag):])[0][1])
                else:
                    xcorr.append(0) 
            else:
                xcorr.append(0) 

        xcorr_time_lagged = {f"lag_{lag}": cross_corr for lag, cross_corr in zip(lags, xcorr)}
        cross_correlations[(class1, class2)] = xcorr_time_lagged 
    return cross_correlations, lags, signal1, signal2 

def save_cross_correlation_to_excel(cross_corr, output_folder, video_name):
    """Saves cross-correlation results to Excel."""
    if cross_corr:
        excel_path = os.path.join(output_folder, f"{video_name}_cross_correlation.xlsx")
        try:
            rows = []
            for (class1, class2), lags_corr in cross_corr.items():
                for lag, corr in lags_corr.items():
                    rows.append({"Behavior 1": class1, "Behavior 2": class2, "Lag": lag, "Cross Correlation": corr})
            df_cross_corr = pd.DataFrame(rows)
            df_cross_corr.to_excel(excel_path, sheet_name="Time Lagged Cross Correlation", index=False)
            return f"Cross-correlation results saved to: {excel_path}" # Return success message
        except Exception as e:
            return f"Error saving cross-correlation results to Excel: {e}" # Return error message
    return "No cross-correlation data to save." # Return if no data


def plot_cross_correlation(cross_corr_results, lags, output_folder, video_name):
    """Plots cross-correlation for each behavior pair."""

    if not cross_corr_results:
        return "No cross-correlation results to plot." # Return message

    plot_paths = [] # List to collect plot paths
    for (class1, class2), xcorr_time_lagged in cross_corr_results.items():
        correlations = list(xcorr_time_lagged.values())
        plt.figure(figsize=(10, 6))
        plt.plot(lags, correlations)
        plt.title(f"Cross-Correlation: {class1} vs {class2} - {video_name}")
        plt.xlabel("Lag (Frames)")
        plt.ylabel("Cross-Correlation Coefficient")
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)  
        plt.axvline(0, color='black', linewidth=0.5)  
        plt.tight_layout()

        plot_filename = f"{video_name}_{class1}_vs_{class2}_cross_correlation.png"
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        message = f"Cross-correlation plot saved to: {plot_path}"
        print(message)
        plot_paths.append(plot_path)
    return "\n".join([f"Plot saved: {path}" for path in plot_paths]) if plot_paths else "No plots generated." # Return paths or no plots message


def main_analysis(output_folder, class_labels, frame_rate, video_name, max_lag_frames=150): # Keyword args, default lag
    """Main function to run time-lagged cross-correlation analysis."""

    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Return error string

    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        return f"Error: {csv_file_path} not found. Run general_analysis.py first." # Error string

    cross_corr_results, lags, signal1, signal2 = calculate_time_lagged_cross_correlation(
        csv_file_path, class_labels, max_lag_frames
    )
    if isinstance(cross_corr_results, str): # Check if calculate_time_lagged_cross_correlation returned error string
        return cross_corr_results # Return the error string

    excel_output_msg = save_cross_correlation_to_excel(cross_corr_results, output_folder, video_name)
    plot_output_msg = plot_cross_correlation(cross_corr_results, lags, output_folder, video_name)

    output_messages = [msg for msg in [excel_output_msg, plot_output_msg] if msg is not None] # Collect non-None messages
    return "\n".join(output_messages) if output_messages else "Time-lagged cross-correlation analysis completed. No specific output messages."


if __name__ == "__main__":
    # Example for direct testing:
    output_folder_path = "path/to/your/output_folder" # Replace with real path
    class_labels_dict = {0: "Exploration", 1: "Grooming", 2: "Jump", 3: "Wall-Rearing", 4: "Rear"}
    frame_rate_val = 30
    video_name_val = "your_video_name" # Replace with real video name
    max_lag_frames_val = 200

    # Dummy Args class for testing (not needed for GUI)
    class Args:
        def __init__(self, output_folder, class_labels, frame_rate, video_name, max_lag_frames):
            self.output_folder = output_folder
            self.class_labels = str(class_labels) # Pass as string for direct test
            self.frame_rate = frame_rate
            self.video_name = video_name
            self.max_lag_frames = max_lag_frames

    test_args = Args(output_folder_path, class_labels_dict, frame_rate_val, video_name_val, max_lag_frames_val)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(output_folder=test_args.output_folder, 
                                  class_labels=class_labels_dict, # Pass dict directly
                                  frame_rate=test_args.frame_rate, 
                                  video_name=test_args.video_name,
                                  max_lag_frames=test_args.max_lag_frames)
    print(output_message) # Print output for direct test