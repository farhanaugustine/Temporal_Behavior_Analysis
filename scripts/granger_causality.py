# scripts/granger_causality.py
import os
import pandas as pd
import argparse
import itertools
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

def calculate_granger_causality(csv_file_path, class_labels, max_lag_frames = 150):
   """
   Calculates Granger causality between all combinations of behaviors

   Args:
        csv_file_path: Path to the CSV file containing frame data.
        class_labels: A list or dictionary of class labels.
        max_lag_frames: The maximum lag to consider (in frames).
   Returns:
       A dictionary where keys are behavior pairs and value is the granger causality test result.

   """
   try:
      df = pd.read_csv(csv_file_path)
   except FileNotFoundError:
      print(f"File not found: {csv_file_path}")
      return None
   except Exception as e:
      print(f"Error reading csv file: {e}")
      return None
    
   granger_causalities = {}
   for (class1, class2) in itertools.combinations(class_labels.values(), 2):
       
      # Create boolean time series for each class
      ts1 = pd.Series([1 if label == class1 else 0 for label in df['Class Label']])
      ts2 = pd.Series([1 if label == class2 else 0 for label in df['Class Label']])
      
      #Combine both series
      combined_series = pd.DataFrame({'ts1': ts1, 'ts2': ts2})
      
      # Perform granger causality test with the maximum lag available.
      try:
          granger_result = grangercausalitytests(combined_series[['ts2', 'ts1']], maxlag=max_lag_frames, verbose=False)
          f_stats_p_value = granger_result[max_lag_frames][0]['ssr_ftest'][1] #Gets the p-value from the test for the maximum lag.
      except ValueError:
          f_stats_p_value = None #If there are less data points than the lag, then this test is not performed.

      granger_causalities[(class1, class2)] = f_stats_p_value

   return granger_causalities

def save_granger_to_excel(granger_results, output_folder, video_name):
    """Saves Granger causality results to an Excel file."""
    if granger_results:
        excel_path = os.path.join(output_folder, f"{video_name}_granger_analysis.xlsx")
        try:
            df_granger = pd.DataFrame([{"Behavior 1": c1, "Behavior 2": c2, "Granger Causality P-value": p}
                                     for (c1, c2), p in granger_results.items()])
            df_granger.to_excel(excel_path, sheet_name="Granger Causality", index=False)
            print(f"Granger causality results saved to: {excel_path}")
        except Exception as e:
             print(f"Error writing to excel file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Perform Granger causality analysis.")
    parser.add_argument("--output_folder", required=True, help="Path to the folder containing YOLO output text files.")
    parser.add_argument("--class_labels", required=True, help="Class labels dictionary (as a string).")
    parser.add_argument("--frame_rate", required=True, type=int, help="Frame rate.")
    parser.add_argument("--video_name", required=True, help="The name of the video to process")
    args = parser.parse_args()

     #Create the csv output folder
    csv_output_folder = os.path.join(args.output_folder, "csv_output")
    if not os.path.exists(csv_output_folder):
       os.makedirs(csv_output_folder)
    
    try:
        class_labels_dict = eval(args.class_labels)
        if not isinstance(class_labels_dict, dict):
            raise ValueError("Class labels must be a dictionary.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid class labels format: {e}")
        return
    
    #Generate the csv file path
    csv_file_path = os.path.join(csv_output_folder, f"{args.video_name}_analysis.csv")

    # Perform Granger causality analysis if the csv file exist
    if os.path.exists(csv_file_path):
        granger_results = calculate_granger_causality(csv_file_path, class_labels_dict)
        save_granger_to_excel(granger_results, args.output_folder, args.video_name) #Save the data
    else:
        print(f"Error, {args.video_name}_analysis.csv file not found. Please run the general analysis first") #Error message if the csv file does not exist


if __name__ == "__main__":
    main()