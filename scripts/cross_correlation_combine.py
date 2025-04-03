import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re  


def calculate_combined_cross_correlation(all_data, output_folder, dpi=1200):
    """Calculates, plots, and returns combined cross-correlation data."""

    combined_data = {}  

    output_messages = [] # List to collect messages

    for behavior_pair, video_data in all_data.items():
        combined_lags = []
        combined_correlations = []

        for video_name, data in video_data.items():
            combined_lags.extend(data["lags"])
            combined_correlations.extend(data["correlations"])

        if not combined_lags:
            message = f"Warning: No lag data for behavior pair '{behavior_pair}'. Skipping."
            print(message)
            output_messages.append(message)
            continue  

        sorted_indices = np.argsort(combined_lags)
        combined_lags = np.array(combined_lags)[sorted_indices]
        combined_correlations = np.array(combined_correlations)[sorted_indices]

        common_lags = np.arange(combined_lags.min(), combined_lags.max() + 1)
        interpolated_correlations = np.interp(
            common_lags, combined_lags, combined_correlations
        )

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

        combined_data[behavior_pair] = {
            "lags": average_lags,
            "correlations": average_correlations_list,
        }

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(average_lags, average_correlations_list, marker="o", linestyle="-", label="Combined")
        plt.xlabel("Lag (Frames)")
        plt.ylabel("Average Cross-Correlation Coefficient")
        plt.title(f"Combined Cross-Correlation for {behavior_pair}")
        plt.grid(True, alpha=0.5)  
        plt.legend(loc="upper right", fontsize=8)
        output_path = os.path.join(output_folder, f"{behavior_pair}_combined_cross_correlation.png")
        plt.savefig(output_path, dpi=dpi)  
        plt.close()
        message = f"Combined cross-correlation plot saved for {behavior_pair}"
        print(message)
        output_messages.append(message)

    return combined_data, output_messages # Return messages too


def create_combined_correlation_heatmap(combined_data, output_folder, dpi=1200):
    """Creates a heatmap of the combined cross-correlations."""

    if not combined_data:
        message = "Warning: No combined data to create heatmap."
        print(message)
        return message, [] # Return message and empty list of paths

    all_lags = []
    for data in combined_data.values():
        all_lags.extend(data["lags"])
    min_lag = min(all_lags)
    max_lag = max(all_lags)
    common_lags = np.arange(min_lag, max_lag + 1)

    heatmap_data = []
    behavior_pairs = []
    for behavior_pair, data in combined_data.items():
        interpolated_correlations = np.interp(
            common_lags, data["lags"], data["correlations"]
        )
        heatmap_data.append(interpolated_correlations)
        behavior_pairs.append(behavior_pair)

    heatmap_df = pd.DataFrame(heatmap_data, index=behavior_pairs, columns=common_lags)

    plt.figure(figsize=(14, 8))  
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
    plt.savefig(output_path, dpi=dpi)  
    plt.close()
    message = f"Combined cross-correlation heatmap saved to {output_path}"
    print(message)
    return message, [output_path] # Return message and path


def main_analysis(cross_corr_folder, output_folder, dpi=1200): # Keyword args, default dpi
    """Main function to combine and visualize cross-correlation results."""

    if not os.path.isdir(cross_corr_folder):
        return f"Error: Cross-correlation folder not found: {cross_corr_folder}" # Return error string

    output_message_list = [] # To collect messages from functions

    os.makedirs(output_folder, exist_ok=True)

    all_data = {}

    for filename in os.listdir(cross_corr_folder):
        if filename.endswith("_cross_correlation.xlsx"):
            filepath = os.path.join(cross_corr_folder, filename)
            video_name = filename.replace("_cross_correlation.xlsx", "")

            try:
                df = pd.read_excel(filepath)  
                if df.empty:
                    message = f"Skipping file {filename}: DataFrame is empty."
                    print(message)
                    output_message_list.append(message)
                    continue

                df['Lag'] = df['Lag'].str.extract(r'lag_(-?\d+)').astype(int)


                for index, row in df.iterrows():
                    behavior1 = row["Behavior 1"]
                    behavior2 = row["Behavior 2"]
                    lag = row["Lag"]  
                    correlation = row["Cross Correlation"]
                    behavior_pair = f"{behavior1} vs {behavior2}"

                    if behavior_pair not in all_data:
                        all_data[behavior_pair] = {}
                    if video_name not in all_data[behavior_pair]:
                        all_data[behavior_pair][video_name] = {"lags": [], "correlations": []}

                    all_data[behavior_pair][video_name]["lags"].append(lag)
                    all_data[behavior_pair][video_name]["correlations"].append(correlation)

            except Exception as e:
                message = f"Error processing file {filename}: {e}"
                print(message)
                output_message_list.append(message)
                continue


    combined_data, cc_plot_messages = calculate_combined_cross_correlation(all_data, output_folder, dpi) # Get messages back
    output_message_list.extend(cc_plot_messages) # Extend main list

    heatmap_message, heatmap_paths = create_combined_correlation_heatmap(combined_data, output_folder, dpi) #Get message and paths
    output_message_list.append(heatmap_message) # Add heatmap message

    return "\n".join(output_message_list) # Return combined messages


if __name__ == "__main__":
    # Example for direct testing:
    cross_corr_folder_path = "path/to/your/cross_correlation_excel_folder"  # Replace
    output_folder_path = "path/to/your/output_folder" # Replace
    dpi_value = 600

    # Dummy Args class for testing (not needed for GUI)
    class Args:
        def __init__(self, cross_corr_folder, output_folder, dpi):
            self.cross_corr_folder = cross_corr_folder
            self.output_folder = output_folder
            self.dpi = dpi

    test_args = Args(cross_corr_folder_path, output_folder_path, dpi_value)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Call main_analysis directly for testing:
    output_message = main_analysis(cross_corr_folder=test_args.cross_corr_folder, 
                                  output_folder=test_args.output_folder,
                                  dpi=test_args.dpi)
    print(output_message) # Print output for direct test