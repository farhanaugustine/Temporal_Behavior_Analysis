import os
import numpy as np
import pandas as pd
import argparse
from scipy import stats
from numpy import trapezoid
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
    adjusted_p_values = np.minimum(adjusted_p_values, 1)  # Ensure p-values <= 1
    for i in range(m - 2, -1, -1):  # Corrected loop for proper adjustment
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

    all_data = {}  # Store the raw data (lags and correlations)
    summary_data = {}  # Store summary metric (peak or AUC) for each video
    stats_data = {} # Store the stats summary
    p_values = {}

    # Loop the files in the excel_file_path, extract data and put them in the all_data.
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
            df.columns = df.columns.str.strip() # Strip any leading/trailing whitespace in column names.

            # Try to find a correlation column.
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

            # Loop through the data in the file
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

    # Calculate the summary metric for each video and store it
    for behavior_pair, video_data in all_data.items():
        if video_data:
            metric_values = []
            video_names = []
            for video_name, data in video_data.items():
                lags = data["lags"]
                correlations = data["correlations"]
                # Calculate the metric (peak or AUC).
                if metric == "peak_correlation":
                    metric_value = max(correlations, key=abs)
                elif metric == "auc":
                    metric_value = trapezoid(correlations, lags)
                else:
                    print(f"Error: Invalid metric '{metric}'. Using peak correlation.")
                    metric_value = max(correlations, key=abs)

                metric_values.append(metric_value)
                video_names.append(video_name)

            summary_data[behavior_pair] = {
                "metric_values": dict(zip(video_names, metric_values))
            }
        else:
            print(f"No data found for behavior pair {behavior_pair}.")

    # Perform statistical test (Friedman test)
    all_metric_values = []
    for behavior_pair, data in summary_data.items():
        all_metric_values.append(list(data["metric_values"].values()))
    #Check the requirements to perform Friedman Test
    if len(all_metric_values) > 0 and all(len(sublist) > 2 for sublist in all_metric_values):
        try:
            h_statistic, p_value = stats.friedmanchisquare(*all_metric_values)
            adjusted_p_value = benjamini_hochberg([p_value])[0]

            # Store results in stats_data
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

            # Check for identical ranks within groups.
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
            # If not applicable, return N/A
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

    #Save the statistical summary into a file
    summary_output_path = os.path.join(output_folder, "cross_correlation_summary.txt")
    with open(summary_output_path, "w") as f:
        for behavior_pair, data in summary_data.items():
            f.write(f"Behavior Pair: {behavior_pair}\n")
            f.write(f"  {metric.replace('_', ' ').title()} Values:\n")
            for video, metric_value in data["metric_values"].items():
                f.write(f"    {video}: {metric_value:.3f}\n")
            f.write("\n")
    print(f"Summary data saved to {summary_output_path}")

    #Save the statistical analysis
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
    parser = argparse.ArgumentParser(description="Perform statistical analysis on combined cross-correlation results.")
    parser.add_argument("--cross_corr_folder", required=True, help="Path to the folder containing cross-correlation Excel files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for statistical analysis results.")
    parser.add_argument("--metric", type=str, default="peak_correlation", choices=["peak_correlation", "auc"],
                        help="Metric to use for statistical analysis: 'peak_correlation' (default) or 'auc'.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    analyze_cross_correlation_files(
        args.cross_corr_folder,
        args.output_folder,
        correlation_column_name="Cross Correlation",
        metric=args.metric,
    )

if __name__ == "__main__":
    main()