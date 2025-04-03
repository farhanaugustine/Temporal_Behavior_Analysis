import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid  
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re
import ast

def benjamini_hochberg(p_values, alpha=0.05):
    """Applies the Benjamini-Hochberg procedure (FDR correction)."""
    p_values = np.array(p_values)
    ranked_p_values = np.argsort(p_values)
    m = len(p_values)
    adjusted_p_values = np.zeros(m)
    for i, rank in enumerate(ranked_p_values):
        adjusted_p_values[rank] = p_values[rank] * m / (i + 1)
    adjusted_p_values = np.minimum(adjusted_p_values, 1)  
    for i in range(m - 2, -1, -1):  
        adjusted_p_values[ranked_p_values[i]] = min(
            adjusted_p_values[ranked_p_values[i]], adjusted_p_values[ranked_p_values[i + 1]]
        )
    return adjusted_p_values

def analyze_cross_correlation_files(
    excel_files_path,
    output_folder,
    correlation_column_name="Cross Correlation",
    metric="peak_correlation",
):
    """Analyzes cross-correlation files, performs stats, and creates summary files."""
    os.makedirs(output_folder, exist_ok=True)

    all_data = {}  
    summary_data = {}  
    stats_data = {}
    p_values = {}

    output_messages = [] # List to collect messages

    for filename in os.listdir(excel_files_path):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(excel_files_path, filename)

            try:
                df = pd.read_excel(filepath)
                if df.empty:
                    message = f"Skipping file {filename}: DataFrame is empty."
                    print(message)
                    output_messages.append(message)
                    continue

            except FileNotFoundError:
                message = f"File not found: {filepath}"
                print(message)
                output_messages.append(message)
                continue
            except Exception as e:
                message = f"Error reading excel file: {e}"
                print(message)
                output_messages.append(message)
                continue

            video_name = os.path.basename(filename).replace(".xlsx", "")
            df.columns = df.columns.str.strip() 

            if correlation_column_name not in df.columns:
                message_warning = f"Warning: Column '{correlation_column_name}' not found in file '{filename}'. Trying to find a similar column..."
                print(message_warning)
                output_messages.append(message_warning)
                correlation_columns = [
                    col
                    for col in df.columns
                    if "correlat" in col.lower() or "correlation" in col.lower()
                ]
                if not correlation_columns:
                    message_error = f"Error: No suitable correlation column found in file '{filename}'. Skipping this file."
                    print(message_error)
                    output_messages.append(message_error)
                    continue
                else:
                    correlation_column_name = correlation_columns[0]
                    message_info = f"Using column '{correlation_column_name}' as the correlation column."
                    print(message_info)
                    output_messages.append(message_info)

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

                all_data[behavior_pair][video_name]["lags"].append(lag)
                all_data[behavior_pair][video_name]["correlations"].append(correlation)

    for behavior_pair, video_data in all_data.items():
        if video_data:
            metric_values = []
            video_names = []
            for video_name, data in video_data.items():
                lags = data["lags"]
                correlations = data["correlations"]
                if metric == "peak_correlation":
                    metric_value = max(correlations, key=abs)
                elif metric == "auc":
                    if not lags or not correlations:
                        message_warning = f"Warning: Empty lags or correlations for {behavior_pair} in video {video_name}.  Setting AUC to NaN."
                        print(message_warning)
                        output_messages.append(message_warning)
                        metric_value = np.nan
                    else:
                        metric_value = trapezoid(correlations, lags)
                else:
                    message_error_metric = f"Error: Invalid metric '{metric}'. Using peak correlation."
                    print(message_error_metric)
                    output_messages.append(message_error_metric)
                    metric_value = max(correlations, key=abs)  # Default to peak

                metric_values.append(metric_value)
                video_names.append(video_name)

            summary_data[behavior_pair] = {
                "metric_values": dict(zip(video_names, metric_values))
            }
        else:
            message_no_data = f"No data found for behavior pair {behavior_pair}."
            print(message_no_data)
            output_messages.append(message_no_data)

    all_metric_values = []
    for behavior_pair, data in summary_data.items():
        all_metric_values.append(list(data["metric_values"].values()))

    if len(all_metric_values) > 0 and all(len(sublist) > 2 for sublist in all_metric_values) and any(len(set(group)) > 1 for group in all_metric_values): 
        try:
            h_statistic, p_value = stats.friedmanchisquare(*all_metric_values)
            adjusted_p_value = benjamini_hochberg([p_value])[0]

            stats_data["all_pairs"] = {
                "test": "Friedman Test",
                "h_statistic": f"{h_statistic:.2f}",
                "p_value": f"{p_value:.3e}",  
                "fdr_adjusted_p_value": f"{adjusted_p_value:.3e}",  
                "significant": adjusted_p_value < 0.05,
                "medians": {},
            }

            for behavior_pair, data in summary_data.items():
                for video_name, metric_value in data["metric_values"].items():
                    stats_data["all_pairs"]["medians"][
                        f"{behavior_pair} - {video_name}"
                    ] = f"{np.median([metric_value]):.3f}"

            message_friedman_results = f"Friedman test results for all behavior pairs:\n  H-statistic: {h_statistic:.2f}\n  P-value: {p_value:.3e}\n  FDR-adjusted P-value: {adjusted_p_value:.3e}"
            print(message_friedman_results)
            output_messages.append(message_friedman_results)

            if adjusted_p_value < 0.05:
                message_significant = f"  Significant difference in {metric} between behavior pairs (FDR-adjusted)"
                print(message_significant)
                output_messages.append(message_significant)
            else:
                message_not_significant = f"  No significant difference in {metric} between behavior pairs (FDR-adjusted)"
                print(message_not_significant)
                output_messages.append(message_not_significant)


        except ValueError as e:
            message_friedman_error = f"Error performing Friedman test: {e}.  Likely not enough samples or all identical."
            print(message_friedman_error)
            output_messages.append(message_friedman_error)
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
        message_not_enough_data = "Not enough data, or fewer than 3 videos per group, or lack of variability to perform Friedman test."
        print(message_not_enough_data)
        output_messages.append(message_not_enough_data)

    # --- Debug Prints Added Before Saving ---
    print("Type of summary_data before saving:", type(summary_data))
    print("Value of summary_data before saving:", repr(summary_data))
    print("Type of stats_data before saving:", type(stats_data))
    print("Value of stats_data before saving:", repr(stats_data))
    # --- End Debug Prints ---


    # Save summary data
    summary_output_path = os.path.join(output_folder, "cross_correlation_summary.txt")
    with open(summary_output_path, "w") as f:
        for behavior_pair, data in summary_data.items(): # Error likely in this loop or the next
            f.write(f"Behavior Pair: {behavior_pair}\n")
            f.write(f"  {metric.replace('_', ' ').title()} Values:\n")
            for video, metric_value in data["metric_values"].items():
                f.write(f"    {video}: {metric_value:.3f}\n")
            f.write("\n")
    message_summary_saved = f"Summary data saved to {summary_output_path}"
    print(message_summary_saved)
    output_messages.append(message_summary_saved)


    # Save statistics data
    stats_output_path = os.path.join(output_folder, "cross_correlation_statistics.txt")
    with open(stats_output_path, "w") as f:
        for behavior_pair, data in stats_data.items(): # ... or in this loop
            if behavior_pair == "all_pairs":
                f.write(f"Statistical Test: {data['test']}\n")
                f.write(f"  H-statistic: {data['h_statistic']}\n")
                f.write(f"  P-value: {data['p_value']}\n")
                f.write(f"  FDR-adjusted P-value: {data['fdr_adjusted_p_value']}\n")
                f.write(f"  Significant Difference: {data['significant']}\n")
                f.write(f"  Medians:\n")
                for median_key, median_value in data["medians"].items(): # ... or specifically in this nested loop
                    f.write(f"    {median_key}: {median_value}\n")
                f.write("\n")
            else:
                f.write(f"Behavior Pair: {behavior_pair}\n")
                f.write(f"  Statistical Test: Not Applicable\n")
                f.write("\n")
    message_stats_saved = f"Statistics data saved to {stats_output_path}"
    print(message_stats_saved)
    output_messages.append(message_stats_saved)

    return "\n".join(output_messages) # Return all messages


def main_analysis(cross_corr_folder, output_folder, metric="peak_correlation"): # Keyword args, default metric
    """Main function to run statistical analysis on cross-correlation results."""

    if not os.path.isdir(cross_corr_folder):
        return f"Error: Cross-correlation folder not found: {cross_corr_folder}" # Return error string

    output_message = analyze_cross_correlation_files( # Capture messages
        cross_corr_folder,
        output_folder,
        correlation_column_name="Cross Correlation",
        metric=metric,
    )
    return output_message # Return combined messages


if __name__ == "__main__":
    # Example for direct testing:
    cross_corr_folder_path = "path/to/your/cross_correlation_excel_folder"  # Replace
    output_folder_path = "path/to/your/output_folder" # Replace
    metric_type = "peak_correlation" # or 'auc'

    # Dummy Args class (not needed for GUI)
    class Args:
        def __init__(self, cross_corr_folder, output_folder, metric):
            self.cross_corr_folder = cross_corr_folder
            self.output_folder = output_folder
            self.metric = metric

    test_args = Args(cross_corr_folder_path, output_folder_path, metric_type)

    # Simulate command-line execution (original main) - not needed for GUI
    # main(test_args)

    # Direct call to main_analysis for testing:
    output_message = main_analysis(cross_corr_folder=test_args.cross_corr_folder, 
                                  output_folder=test_args.output_folder,
                                  metric=test_args.metric)
    print(output_message) # Print output for direct test