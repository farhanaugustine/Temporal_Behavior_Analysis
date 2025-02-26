import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Import the 're' module


def calculate_combined_cross_correlation(all_data, output_folder, dpi=1200):
    """Calculates, plots, and returns combined cross-correlation data."""

    combined_data = {}  # Store combined data for heatmap

    for behavior_pair, video_data in all_data.items():
        combined_lags = []
        combined_correlations = []

        # Collect all lags and correlations
        for video_name, data in video_data.items():
            combined_lags.extend(data["lags"])
            combined_correlations.extend(data["correlations"])

        # --- FIX: Check if combined_lags is empty ---
        if not combined_lags:
            print(f"Warning: No lag data found for behavior pair '{behavior_pair}'. Skipping.")
            continue  # Skip this behavior pair if no data

        # Sort by lag
        sorted_indices = np.argsort(combined_lags)
        combined_lags = np.array(combined_lags)[sorted_indices]
        combined_correlations = np.array(combined_correlations)[sorted_indices]

        # Interpolation
        common_lags = np.arange(combined_lags.min(), combined_lags.max() + 1)
        interpolated_correlations = np.interp(
            common_lags, combined_lags, combined_correlations
        )

        # Averaging
        correlation_sums = {}
        correlation_counts = {}
        for lag, correlation in zip(common_lags, interpolated_correlations):
            if lag not in correlation_sums:
                correlation_sums[lag] = 0
                correlation_counts[lag] = 0
            correlation_sums[lag] += correlation
            correlation_counts[lag] += 1

        average_correlations = {
            lag: correlation_sums[lag] / correlation_counts[lag]
            for lag in correlation_sums
        }
        average_lags = list(average_correlations.keys())
        average_correlations_list = list(average_correlations.values())

        # Store for heatmap
        combined_data[behavior_pair] = {
            "lags": average_lags,
            "correlations": average_correlations_list,
        }

        # Plotting (same as before)
        plt.figure(figsize=(12, 6))
        plt.plot(average_lags, average_correlations_list, marker="o", linestyle="-", label="Combined")
        plt.xlabel("Lag (Frames)")
        plt.ylabel("Average Cross-Correlation Coefficient")
        plt.title(f"Combined Cross-Correlation for {behavior_pair}")
        plt.grid(True, alpha=0.5)  # Keep original alpha
        plt.legend(loc="upper right", fontsize=8)
        output_path = os.path.join(output_folder, f"{behavior_pair}_combined_cross_correlation.png")
        plt.savefig(output_path, dpi=dpi)  # Save with specified DPI
        plt.close()
        print(f"Combined cross-correlation plot saved for {behavior_pair}")

    return combined_data


def create_combined_correlation_heatmap(combined_data, output_folder, dpi=1200):
    """Creates a heatmap of the combined cross-correlations."""

    # --- FIX: Handle empty combined_data ---
    if not combined_data:
        print("Warning: No combined data to create heatmap.")
        return

    # Find the overall min and max lags across all behavior pairs
    all_lags = []
    for data in combined_data.values():
        all_lags.extend(data["lags"])
    min_lag = min(all_lags)
    max_lag = max(all_lags)

    # Create a common lag range
    common_lags = np.arange(min_lag, max_lag + 1)

    # Prepare data for the heatmap
    heatmap_data = []
    behavior_pairs = []

    for behavior_pair, data in combined_data.items():
        # Interpolate *again* to the *overall* common lags
        interpolated_correlations = np.interp(
            common_lags, data["lags"], data["correlations"]
        )
        heatmap_data.append(interpolated_correlations)
        behavior_pairs.append(behavior_pair)

    # Create the heatmap DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, index=behavior_pairs, columns=common_lags)

    # Plot the heatmap
    plt.figure(figsize=(14, 8))  # Adjust size as needed
    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        cbar_kws={"label": "Average Cross-Correlation"},
        linewidths=0.5,
        linecolor="black",
    )
    plt.title("Combined Cross-Correlation Heatmap")
    plt.xlabel("Lag (Frames)")
    plt.ylabel("Behavior Pairs")
    plt.tight_layout()

    output_path = os.path.join(output_folder, "combined_cross_correlation_heatmap.png")
    plt.savefig(output_path, dpi=dpi)  # Save with specified DPI
    plt.close()
    print(f"Combined cross-correlation heatmap saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Combine and visualize cross-correlation results from multiple videos.")
    parser.add_argument("--cross_corr_folder", required=True, help="Path to the folder containing cross-correlation Excel files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for combined results.")
    parser.add_argument("--dpi", type=int, default=1200, help="DPI for saved figures (default: 1200).") # Added DPI argument.
    args = parser.parse_args()

    print(f"Input folder: {args.cross_corr_folder}")
    print(f"Output folder: {args.output_folder}")

    os.makedirs(args.output_folder, exist_ok=True)

    all_data = {}

    for filename in os.listdir(args.cross_corr_folder):
        if filename.endswith("_cross_correlation.xlsx"):
            filepath = os.path.join(args.cross_corr_folder, filename)
            video_name = filename.replace("_cross_correlation.xlsx", "")
            print(f"Processing file: {filepath}, video_name: {video_name}")

            try:
                df = pd.read_excel(filepath)  # Read without dtype enforcement
                 # --- Check for empty DataFrame ---
                if df.empty:
                    print(f"Skipping file {filename}: DataFrame is empty.")
                    continue
                print(f"Successfully read file: {filename}")

                # --- Extract Lag Numbers using Regular Expressions ---
                df['Lag'] = df['Lag'].str.extract(r'lag_(-?\d+)').astype(int)
                print(f"DataFrame head after lag extraction:\n{df.head()}")
                print(f"DataFrame dtypes after lag extraction:\n{df.dtypes}")

                for index, row in df.iterrows():
                    behavior1 = row["Behavior 1"]
                    behavior2 = row["Behavior 2"]
                    lag = row["Lag"]  # Now a number
                    correlation = row["Cross Correlation"]
                    behavior_pair = f"{behavior1} vs {behavior2}"

                    if behavior_pair not in all_data:
                        all_data[behavior_pair] = {}
                    if video_name not in all_data[behavior_pair]:
                        all_data[behavior_pair][video_name] = {"lags": [], "correlations": []}

                    all_data[behavior_pair][video_name]["lags"].append(lag)
                    all_data[behavior_pair][video_name]["correlations"].append(correlation)

            except Exception as e:
                print(f"Error reading or processing file {filename}: {e}")
                continue

    print(f"all_data: {all_data}")
    combined_data = calculate_combined_cross_correlation(all_data, args.output_folder, args.dpi) # Pass DPI
    create_combined_correlation_heatmap(combined_data, args.output_folder, args.dpi) # Pass DPI

if __name__ == "__main__":
    main()