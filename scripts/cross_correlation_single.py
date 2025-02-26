import os
import numpy as np
import pandas as pd
import argparse
import itertools
import matplotlib.pyplot as plt
import ast  # Import ast

def calculate_time_lagged_cross_correlation(csv_file_path, class_labels, max_lag_frames=150):
    """Calculates time-lagged cross-correlation, now returns data for plotting."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None, None, None, None #Return None for all.
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, None, None

    all_labels = df['Class Label'].tolist()
    cross_correlations = {}

    for (class1, class2) in itertools.combinations(class_labels.values(), 2):
        signal1 = np.array([1 if label == class1 else 0 for label in all_labels])
        signal2 = np.array([1 if label == class2 else 0 for label in all_labels])
        lags = np.arange(-max_lag_frames, max_lag_frames + 1)
        xcorr = []

        for lag in lags:
            if (len(signal1) - abs(lag) > 0 and len(signal2) - abs(lag) > 0):
                # Handle cases with zero variance
                if np.std(signal1[:len(signal1) - abs(lag)]) > 0 and np.std(signal2[abs(lag):]) > 0:
                    xcorr.append(np.corrcoef(signal1[:len(signal1) - abs(lag)], signal2[abs(lag):])[0][1])
                else:
                    xcorr.append(0)  # Or np.nan, depending on how you want to handle it
            else:
                xcorr.append(0)  # Handle cases where lag is too large

        xcorr_time_lagged = {f"lag_{lag}": cross_corr for lag, cross_corr in zip(lags, xcorr)}
        cross_correlations[(class1, class2)] = xcorr_time_lagged  # Store for excel
    return cross_correlations, lags, signal1, signal2 # Return lags for plotting.

def save_cross_correlation_to_excel(cross_corr, output_folder, video_name):
    """Saves cross-correlation results to Excel (same as before)."""
    if cross_corr:
        excel_path = os.path.join(output_folder, f"{video_name}_cross_correlation.xlsx")
        try:
            rows = []
            for (class1, class2), lags_corr in cross_corr.items():
                for lag, corr in lags_corr.items():
                    rows.append({"Behavior 1": class1, "Behavior 2": class2, "Lag": lag, "Cross Correlation": corr})
            df_cross_corr = pd.DataFrame(rows)
            df_cross_corr.to_excel(excel_path, sheet_name="Time Lagged Cross Correlation", index=False)
            print(f"Cross-correlation results saved to: {excel_path}")
        except Exception as e:
            print(f"Error saving cross-correlation results to Excel: {e}")

def plot_cross_correlation(cross_corr_results, lags, output_folder, video_name):
    """Plots the cross-correlation for each behavior pair."""

    if not cross_corr_results:
        print("No cross-correlation results to plot.")
        return

    for (class1, class2), xcorr_time_lagged in cross_corr_results.items():
        correlations = list(xcorr_time_lagged.values())
        plt.figure(figsize=(10, 6))
        plt.plot(lags, correlations)
        plt.title(f"Cross-Correlation: {class1} vs {class2} - {video_name}")
        plt.xlabel("Lag (Frames)")
        plt.ylabel("Cross-Correlation Coefficient")
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)  # Add a horizontal line at y=0
        plt.axvline(0, color='black', linewidth=0.5)  # Add a vertical line at x=0
        plt.tight_layout()

        plot_filename = f"{video_name}_{class1}_vs_{class2}_cross_correlation.png"
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Cross-correlation plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform time-lagged cross-correlation analysis for a single video.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.") #Keep frame_rate for consistency
    parser.add_argument("--video_name", required=True, help="Video name.")
    parser.add_argument("--max_lag_frames", type=int, default=150, help="Maximum lag in frames.")
    args = parser.parse_args()

    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)

    try:
        class_labels_dict = ast.literal_eval(args.class_labels)
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels: {e}")
        return

    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found.  Run general_analysis.py first.")
        return

    cross_corr_results, lags, signal1, signal2 = calculate_time_lagged_cross_correlation(
        csv_file_path, class_labels_dict, args.max_lag_frames
    )

    if cross_corr_results:
        save_cross_correlation_to_excel(cross_corr_results, args.output_folder, args.video_name)
        plot_cross_correlation(cross_corr_results, lags, args.output_folder, args.video_name)


if __name__ == "__main__":
    main()