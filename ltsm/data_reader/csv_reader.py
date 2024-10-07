import os
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
)

def transform_csv(input_file: str) -> pd.DataFrame:
    """
    This function reads the CSV file, deletes the first row and the first column, 
    replaces the time with 0, 1, 2 sequence, and returns the transformed DataFrame.
    
    Args:
        input_file(str): CSV file path

    Returns:
        pd.DataFrame: transformed DataFrame
    """
    try:
        df = pd.read_csv(input_file, header=None)
        df = df.drop(index=0)
        if df.shape[1] > 1:  
            df = df.drop(columns=[df.columns[0]])
        df = df.fillna(0)  # deal with possible NaN values
        df_transposed = df.T 
        df_transposed.columns = range(len(df_transposed.columns))

        return df_transposed.reset_index(drop=True) # reset index to start from 0
    except FileNotFoundError as e:
        logging.error(f"Error: File not found - {input_file}")
        raise e  

    except pd.errors.EmptyDataError as e:
        logging.error(f"Error: The file is empty or invalid - {input_file}")
        raise e

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise e

def transform_csv_dataset(input_folder: str, output_folder: str):
    """
    Iterates through all the CSV files in the input folder and 
    converts each one, saving it to the output folder.

    Args:
        input_folder(str): path to the folder containing the CSV files to be transformed
        output_folder(str): output folder path

    Returns:
        list of transformed DataFrames
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    rtn_data = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            try:
                df_transformed = transform_csv(input_file)
                df_transformed.to_csv(output_file, index=False)
                rtn_data.append(df_transformed)
                logging.info(f"DK CSV transform finished. Output saved to {output_file}")
            except Exception as e:
                print("here")
                logging.error(f"Processing {input_file} , have error: {e}")
                raise e
    return rtn_data

                
if __name__ == '__main__':
    input_folder = './datasets/DK/'
    output_folder = './datasets/DK_transformed/'
    transform_csv_dataset(input_folder, output_folder)
