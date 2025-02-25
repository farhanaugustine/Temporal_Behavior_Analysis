import os
import numpy as np
import pandas as pd
import argparse
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from numpy import trapezoid
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re  # Import the 're' module


def benjamini_hochberg(p_values, alpha=0.05):
    """Applies the Benjamini-Hochberg procedure (FDR correction)."""
    p_values = np.array(p_values)
    ranked_p_values = np.argsort(p_values)
    m = len(p_values)
    adjusted_p_values = np.zeros(m)
    for i, rank in enumerate(ranked_p_values):
        adjusted_p_values[rank] = p_values[rank] * m / (i + 1)
    adjusted_p_values = np.minimum(adjusted_p_values, 1)  # Ensure p-values <= 1
    for i in range(m - 2, -1, -1):  # Corrected loop for proper adjustment
        adjusted_p_values[ranked_p_values[i]] = min(
            adjusted_p_values[ranked_p_values[i]], adjusted_p_values[ranked_p_values[i + 1]]
        )
    return adjusted_p_values

def calculate_time_lagged_cross_correlation(csv_file_path, class_labels, max_lag_frames=150, frame_rate=30):
    """Calculates time-lagged cross-correlation, now returns data for plotting."""
    try:
        df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: CSV file not found or empty: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

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

def calculate_combined_cross_correlation(all_data, output_folder):
    """Calculates, plots, and returns combined cross-correlation data."""

    combined_data = {}  # Store combined data for heatmap

    for behavior_pair, video_data in all_data.items():
        combined_lags = []
        combined_correlations = []

        # Collect all lags and correlations
        for video_name, data in video_data.items():
            combined_lags.extend(data["lags"])
            combined_correlations.extend(data["correlations"])

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
        plt.xlabel("Lag")
        plt.ylabel("Average Cross-Correlation Coefficient")
        plt.title(f"Combined Cross-Correlation for {behavior_pair}")
        plt.grid(True, alpha=0.5)
        plt.legend(loc="upper right", fontsize=8)
        output_path = os.path.join(output_folder, f"{behavior_pair}_combined_cross_correlation.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Combined cross-correlation plot saved for {behavior_pair}")

    return combined_data


def create_combined_correlation_heatmap(combined_data, output_folder):
    """Creates a heatmap of the combined cross-correlations."""

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
    plt.xlabel("Lag")
    plt.ylabel("Behavior Pairs")
    plt.tight_layout()

    output_path = os.path.join(output_folder, "combined_cross_correlation_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Combined cross-correlation heatmap saved to {output_path}")


def analyze_cross_correlation_files(
    excel_files_path,
    output_folder,
    correlation_column_name="Cross Correlation",
    metric="peak_correlation",
):
    """Analyzes cross-correlation files, with combined heatmap."""
    os.makedirs(output_folder, exist_ok=True)

    all_data = {}
    summary_data = {}
    stats_data = {}
    p_values = {}

    for filename in os.listdir(excel_files_path):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(excel_files_path, filename)

            try:
                df = pd.read_excel(filepath)
            except FileNotFoundError:
                print(f"File not found: {filepath}")
                continue
            except Exception as e:
                print(f"Error reading excel file: {e}")
                continue

            video_name = os.path.basename(filename).replace(".xlsx", "")
            df.columns = df.columns.str.strip()

            if correlation_column_name not in df.columns:
                print(
                    f"Warning: Column '{correlation_column_name}' not found in file '{filename}'. Trying to find a similar column..."
                )
                correlation_columns = [
                    col
                    for col in df.columns
                    if "correlat" in col.lower() or "correlation" in col.lower()
                ]
                if not correlation_columns:
                    print(
                        f"Error: No suitable correlation column found in file '{filename}'. Skipping this file."
                    )
                    continue
                else:
                    correlation_column_name = correlation_columns[0]
                    print(f"Using column '{correlation_column_name}' as the correlation column.")

            for index, row in df.iterrows():
                behavior1 = row["Behavior 1"]
                behavior2 = row["Behavior 2"]
                lag = row["Lag"]
                correlation = row[correlation_column_name]

                behavior_pair = f"{behavior1} vs {behavior2}"

                if behavior_pair not in all_data:
                    all_data[behavior_pair] = {}
                if video_name not in all_data[behavior_pair]:
                    all_data[behavior_pair][video_name] = {"lags": [], "correlations": []}

                try:
                    lag_value = int(re.search(r"lag_(-?\d+)", lag).group(1)) # Use re here.
                except:
                    print(
                        f"Warning: Could not parse lag value from '{lag}'. Skipping this data point."
                    )
                    continue

                all_data[behavior_pair][video_name]["lags"].append(lag_value)
                all_data[behavior_pair][video_name]["correlations"].append(correlation)

    # Calculate combined cross-correlations
    combined_data = calculate_combined_cross_correlation(all_data, output_folder)

    # Create combined heatmap
    create_combined_correlation_heatmap(combined_data, output_folder)

    for behavior_pair, video_data in all_data.items():
        if video_data:
            plt.figure(figsize=(12, 6))
            metric_values = []
            video_names = []
            for video_name, data in video_data.items():
                lags = data["lags"]
                correlations = data["correlations"]
                plt.plot(lags, correlations, marker="o", linestyle="-", label=video_name)

                if metric == "peak_correlation":
                    metric_value = max(correlations, key=abs)
                elif metric == "auc":
                    metric_value = trapezoid(correlations, lags)
                else:
                    print(f"Error: Invalid metric '{metric}'. Using peak correlation.")
                    metric_value = max(correlations, key=abs)

                metric_values.append(metric_value)
                video_names.append(video_name)

            plt.xlabel("Lag")
            plt.ylabel("Cross-Correlation Coefficient")
            plt.title(f"Cross-Correlation for {behavior_pair} Across Videos")
            plt.grid(True, alpha=0.5)
            plt.legend(loc="upper right", fontsize=8)

            output_path = os.path.join(
                output_folder, f"{behavior_pair}_cross_correlation_all_videos.png"
            )
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved for {behavior_pair}")

            summary_data[behavior_pair] = {
                "metric_values": dict(zip(video_names, metric_values))
            }
        else:
            print(f"No data found for behavior pair {behavior_pair}.")

    heatmap_data = {}
    for behavior_pair, data in summary_data.items():
        heatmap_data[behavior_pair] = data["metric_values"]

    heatmap_df = pd.DataFrame(heatmap_data).T

    # Perform statistical test (Friedman test)
    all_metric_values = []
    for behavior_pair, data in summary_data.items():
        all_metric_values.append(list(data["metric_values"].values()))

    if len(all_metric_values) > 0 and all(len(sublist) > 2 for sublist in all_metric_values):
        try:
            h_statistic, p_value = stats.friedmanchisquare(*all_metric_values)
            adjusted_p_value = benjamini_hochberg([p_value])[0]

            # --- FIX: Use scientific notation for p-values ---
            stats_data["all_pairs"] = {
                "test": "Friedman Test",
                "h_statistic": f"{h_statistic:.2f}",
                "p_value": f"{p_value:.3e}",  # Use scientific notation
                "fdr_adjusted_p_value": f"{adjusted_p_value:.3e}",  # Use scientific notation
                "significant": adjusted_p_value < 0.05,
                "medians": {},
            }

            # Calculate and store medians correctly
            for behavior_pair, data in summary_data.items():
                for video_name, metric_value in data["metric_values"].items():
                    stats_data["all_pairs"]["medians"][
                        f"{behavior_pair} - {video_name}"
                    ] = f"{np.median([metric_value]):.3f}"

            print(f"Friedman test results for all behavior pairs:")
            print(f"  H-statistic: {h_statistic:.2f}")
            print(f"  P-value: {p_value:.3e}")  # Display in scientific notation
            print(
                f"  FDR-adjusted P-value: {adjusted_p_value:.3e}"
            )  # Display in scientific notation

            if adjusted_p_value < 0.05:
                print(
                    f"  Significant difference in {metric} between behavior pairs (FDR-adjusted)"
                )
            else:
                print(
                    f"  No significant difference in {metric} between behavior pairs (FDR-adjusted)"
                )

            # --- ADDED: Check for identical ranks within groups ---
            for i, behavior_data in enumerate(all_metric_values):
                if np.allclose(behavior_data, behavior_data[0]):
                    behavior_pair = list(summary_data.keys())[i]
                    print(
                        f"WARNING: All metric values for behavior pair '{behavior_pair}' are identical.  This might lead to unreliable Friedman test results."
                    )

        except ValueError as e:
            print(
                f"Error performing Friedman test: {e}.  Likely not enough samples or all samples are identical."
            )
            stats_data["all_pairs"] = {
                "test": "Friedman Test (Error)",
                "h_statistic": "N/A",
                "p_value": "N/A",
                "fdr_adjusted_p_value": "N/A",
                "significant": False,
                "medians": "N/A",
            }
    else:
        stats_data["all_pairs"] = {
            "test": "Not Applicable",
            "h_statistic": "N/A",
            "p_value": "N/A",
            "fdr_adjusted_p_value": "N/A",
            "significant": False,
            "medians": "N/A",
        }
        print("Not enough data, or fewer than 3 videos per group, to perform the Friedman test.")

    annot_data = np.empty_like(heatmap_df, dtype=object)
    for i, behavior_pair in enumerate(heatmap_df.index):
        for j, video in enumerate(heatmap_df.columns):
            value = heatmap_df.iloc[i, j]
            if stats_data["all_pairs"]["significant"]:
                p = adjusted_p_value
                if p < 0.001:
                    annot_data[i, j] = f"{value:.2f}***"
                elif p < 0.01:
                    annot_data[i, j] = f"{value:.2f}**"
                elif p < 0.05:
                    annot_data[i, j] = f"{value:.2f}*"
                else:
                    annot_data[i, j] = f"{value:.2f}"
            else:
                annot_data[i, j] = f"{value:.2f}"

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_df,
        annot=annot_data,
        cmap="viridis",
        fmt="",
        annot_kws={"fontsize": 10},
        mask=heatmap_df.isnull(),
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    plt.title(f"{metric.replace('_', ' ').title()} Values for Behavior Pairs Across Videos")
    plt.xlabel("Videos")
    plt.ylabel("Behavior Pairs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    heatmap_output_path = os.path.join(output_folder, "metric_values_heatmap.png")  # Corrected path
    plt.savefig(heatmap_output_path)
    plt.close()
    print(f"Heatmap saved to {heatmap_output_path}")


    summary_output_path = os.path.join(output_folder, "cross_correlation_summary.txt")
    with open(summary_output_path, "w") as f:
        for behavior_pair, data in summary_data.items():
            f.write(f"Behavior Pair: {behavior_pair}\n")
            f.write(f"  {metric.replace('_', ' ').title()} Values:\n")
            for video, metric_value in data["metric_values"].items():
                f.write(f"    {video}: {metric_value:.3f}\n")
            f.write("\n")
    print(f"Summary data saved to {summary_output_path}")

    stats_output_path = os.path.join(output_folder, "cross_correlation_statistics.txt")
    with open(stats_output_path, "w") as f:
        for behavior_pair, data in stats_data.items():
            if behavior_pair == "all_pairs":
                f.write(f"Statistical Test: {data['test']}\n")
                f.write(f"  H-statistic: {data['h_statistic']}\n")
                f.write(f"  P-value: {data['p_value']}\n")
                f.write(f"  FDR-adjusted P-value: {data['fdr_adjusted_p_value']}\n")
                f.write(f"  Significant Difference: {data['significant']}\n")
                f.write(f"  Medians:\n")
                for median_key, median_value in data["medians"].items():
                    f.write(f"    {median_key}: {median_value}\n")
                f.write("\n")
            else:
                f.write(f"Behavior Pair: {behavior_pair}\n")
                f.write(f"  Statistical Test: Not Applicable\n")
                f.write("\n")
    print(f"Statistics data saved to {stats_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform time-lagged cross-correlation analysis.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.")
    parser.add_argument("--video_name", required=True, help="Video name.")
    parser.add_argument("--max_lag_frames", type=int, default=150, help="Maximum lag in frames.")
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
        print(f"Error: {csv_file_path} not found.  Run general_analysis.py first.")
        return

    cross_corr_results, lags, signal1, signal2 = calculate_time_lagged_cross_correlation(
        csv_file_path, class_labels_dict, args.max_lag_frames, args.frame_rate
    )

    if cross_corr_results:
        save_cross_correlation_to_excel(cross_corr_results, args.output_folder, args.video_name)
        plot_cross_correlation(cross_corr_results, lags, args.output_folder, args.video_name)
    #The following part of the code is for performing statistical analysis and further data visualization.
    #First, let's create a folder to store the cross_correlation excels.
    excel_output_folder = os.path.join(args.output_folder, "cross_correlation_excel")
    if not os.path.exists(excel_output_folder):
        os.makedirs(excel_output_folder)

    # Copy the cross_correlation excels to that folder
    source_excel_path = os.path.join(args.output_folder, f"{args.video_name}_cross_correlation.xlsx") #Original file
    destination_excel_path = os.path.join(excel_output_folder, f"{args.video_name}_cross_correlation.xlsx") #New destination
    if os.path.exists(source_excel_path):
        try:
            import shutil
            shutil.copy(source_excel_path, destination_excel_path)
            print(f"Copied cross-correlation Excel file to: {destination_excel_path}")
        except Exception as e:
            print(f"Error copying file: {e}")

    # Now, let's call your analysis function with the new folder for further analysis
    analyze_cross_correlation_files(
        excel_output_folder,
        args.output_folder,
        correlation_column_name="Cross Correlation",
        metric="peak_correlation",  # Or "auc", as needed
    )

if __name__ == "__main__":
    main()