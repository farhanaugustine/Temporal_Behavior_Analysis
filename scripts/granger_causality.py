# scripts/granger_causality.py
import os
import pandas as pd
import itertools
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")
import ast

def calculate_granger_causality(csv_file_path, class_labels, max_lag_frames = 150):
   """
   Calculates Granger causality between all combinations of behaviors
   """
   try:
      df = pd.read_csv(csv_file_path)
   except FileNotFoundError:
      return f"File not found: {csv_file_path}", None # Return error string
   except Exception as e:
      return f"Error reading csv file: {e}", None # Return error string
    
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
          f_stats_p_value = granger_result[max_lag_frames][0]['ssr_ftest'][1] #Gets p-value from test for max lag.
      except ValueError:
          f_stats_p_value = None #If less data points than lag, test not performed.

      granger_causalities[(class1, class2)] = f_stats_p_value

   return None, granger_causalities # Return None for error, and results


def save_granger_to_excel(granger_results, output_folder, video_name):
    """Saves Granger causality results to Excel."""
    if granger_results:
        excel_path = os.path.join(output_folder, f"{video_name}_granger_analysis.xlsx")
        try:
            df_granger = pd.DataFrame([{"Behavior 1": c1, "Behavior 2": c2, "Granger Causality P-value": p}
                                     for (c1, c2), p in granger_results.items()])
            df_granger.to_excel(excel_path, sheet_name="Granger Causality", index=False)
            return f"Granger causality results saved to: {excel_path}" # Return success message
        except Exception as e:
             return f"Error writing to excel file: {e}" # Return error message
    return "No Granger causality data to save." # Return if no data


def main_analysis(output_folder, class_labels, frame_rate, video_name): # Keyword args
    """Main function to run Granger causality analysis."""

     #Create the csv output folder
    csv_output_folder = os.path.join(output_folder, "csv_output")
    os.makedirs(csv_output_folder, exist_ok=True)
    
    if not isinstance(class_labels, dict):
        return "Error: Class labels must be a dictionary." # Return error string
    
    #Generate the csv file path
    csv_file_path = os.path.join(csv_output_folder, f"{video_name}_analysis.csv")

    # Perform Granger causality analysis if csv file exists
    if not os.path.exists(csv_file_path):
        return f"Error: {video_name}_analysis.csv file not found. Run general_analysis first." #Error string

    error_granger, granger_results = calculate_granger_causality(csv_file_path, class_labels) # Get potential error and results
    if error_granger: # If calculate_granger_causality returned an error string
        return error_granger # Return error string to GUI

    excel_output_msg = save_granger_to_excel(granger_results, output_folder, video_name) # Save and get message

    output_messages = [msg for msg in [excel_output_msg] if msg is not None] # Collect non-None messages
    return "\n".join(output_messages) if output_messages else "Granger causality analysis completed. No specific output messages." # Return combined messages


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