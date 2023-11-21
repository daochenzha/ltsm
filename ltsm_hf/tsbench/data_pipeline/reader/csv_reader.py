import numpy as np
import pandas as pd
from distutils.util import strtobool
from datetime import datetime

class Reader():
    def __init__(self):
        pass

    def fetch(self):
        # input: path
        # output: DataFrame
        pass


# class MonashReader(Reader):
#     def __init__(self, data_path):
#         super().__init__()
#         self.data_path = data_path

#     def fetch(self):
#         # input: path
#         # output: DataFrame
#         df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = self._tsf_to_dataframe(self.data_path)

#         def dropna(x):
#             return x[~np.isnan(x)]
#         timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
#         return timeseries

#     def _tsf_to_dataframe(self, data_path: str, 
#                           replace_missing_vals_with="NaN", 
#                           value_column_name="series_value"):
#         col_names = []
#         col_types = []
#         all_data = {}
#         line_count = 0
#         frequency = None
#         forecast_horizon = None
#         contain_missing_values = None
#         contain_equal_length = None
#         found_data_tag = False
#         found_data_section = False
#         started_reading_data_section = False
#         with open(data_path, "r", encoding="cp1252") as file:
#             for line in file:
#                 # Strip white space from start/end of line
#                 line = line.strip()
#                 if line:
#                     if line.startswith("@"):  # Read meta-data
#                         if not line.startswith("@data"):
#                             line_content = line.split(" ")
#                             if line.startswith("@attribute"):
#                                 if (
#                                     len(line_content) != 3
#                                 ):  # Attributes have both name and type
#                                     raise Exception("Invalid meta-data specification.")

#                                 col_names.append(line_content[1])
#                                 col_types.append(line_content[2])
#                             else:
#                                 if (
#                                     len(line_content) != 2
#                                 ):  # Other meta-data have only values
#                                     raise Exception("Invalid meta-data specification.")

#                                 if line.startswith("@frequency"):
#                                     frequency = line_content[1]
#                                 elif line.startswith("@horizon"):
#                                     forecast_horizon = int(line_content[1])
#                                 elif line.startswith("@missing"):
#                                     contain_missing_values = bool(
#                                         strtobool(line_content[1])
#                                     )
#                                 elif line.startswith("@equallength"):
#                                     contain_equal_length = bool(strtobool(line_content[1]))

#                         else:
#                             if len(col_names) == 0:
#                                 raise Exception(
#                                     "Missing attribute section. Attribute section must come before data."
#                                 )

#                             found_data_tag = True
#                     elif not line.startswith("#"):
#                         if len(col_names) == 0:
#                             raise Exception(
#                                 "Missing attribute section. Attribute section must come before data."
#                             )
#                         elif not found_data_tag:
#                             raise Exception("Missing @data tag.")
#                         else:
#                             if not started_reading_data_section:
#                                 started_reading_data_section = True
#                                 found_data_section = True
#                                 all_series = []

#                                 for col in col_names:
#                                     all_data[col] = []

#                             full_info = line.split(":")

#                             if len(full_info) != (len(col_names) + 1):
#                                 raise Exception("Missing attributes/values in series.")

#                             series = full_info[len(full_info) - 1]
#                             series = series.split(",")

#                             if len(series) == 0:
#                                 raise Exception(
#                                     "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
#                                 )

#                             numeric_series = []

#                             for val in series:
#                                 if val == "?":
#                                     numeric_series.append(replace_missing_vals_with)
#                                 else:
#                                     numeric_series.append(float(val))

#                             if numeric_series.count(replace_missing_vals_with) == len(
#                                 numeric_series
#                             ):
#                                 raise Exception(
#                                     "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
#                                 )

#                             all_series.append(pd.Series(numeric_series).array)

#                             for i in range(len(col_names)):
#                                 att_val = None
#                                 if col_types[i] == "numeric":
#                                     att_val = int(full_info[i])
#                                 elif col_types[i] == "string":
#                                     att_val = str(full_info[i])
#                                 elif col_types[i] == "date":
#                                     att_val = datetime.strptime(
#                                         full_info[i], "%Y-%m-%d %H-%M-%S"
#                                     )
#                                 else:
#                                     raise Exception(
#                                         "Invalid attribute type."
#                                     )  # Currently, the code supports only numeric, string and date types. Extend this as required.

#                                 if att_val is None:
#                                     raise Exception("Invalid attribute value.")
#                                 else:
#                                     all_data[col_names[i]].append(att_val)

#                     line_count = line_count + 1

#             if line_count == 0:
#                 raise Exception("Empty file.")
#             if len(col_names) == 0:
#                 raise Exception("Missing attribute section.")
#             if not found_data_section:
#                 raise Exception("Missing series information under data section.")

#             all_data[value_column_name] = all_series
#             print(all_data)
#             loaded_data = pd.DataFrame(all_data)

#             return (
#                 loaded_data,
#                 frequency,
#                 forecast_horizon,
#                 contain_missing_values,
#                 contain_equal_length,
#             )

class ReaderFlightDelayPrediction(Reader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        #self.fetch()

    def fetch(self):
        # input: path
        # output: DataFrame
        df = self._csv_to_dataframe(self.data_path)
        return df

    def _csv_to_dataframe(self, data_path: str, 
                          replace_missing_vals_with="NaN", 
                          value_column_name="series_value"):

        loaded_data=pd.read_csv(data_path, header=0, sep=',', engine='python')
        loaded_data.drop(columns=['OP_UNIQUE_CARRIER','OP_CARRIER','TAIL_NUM','ORIGIN','DEST','DEP_TIME_BLK'],inplace=True)
        all_series=[]
        numeric_series=[]
        for i in range(loaded_data.shape[0]):
            numeric_series=loaded_data.iloc[i,:].values
            all_series.append(pd.Series(numeric_series).array)
        return all_series
    
class ReaderFlightStatusPrediction(Reader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        #self.fetch()

    def fetch(self):
        # input: path
        # output: DataFrame
        df = self._csv_to_dataframe(self.data_path)
        return df

    def _csv_to_dataframe(self, data_path: str, 
                          replace_missing_vals_with="NaN", 
                          value_column_name="series_value"):

        loaded_data=pd.read_csv(data_path, header=0, sep=',', engine='python')
        loaded_data.drop(columns=['FlightDate','Airline','Origin','Dest','Cancelled','Diverted',
                                  'Marketing_Airline_Network','Operated_or_Branded_Code_Share_Partners',
                                  'IATA_Code_Marketing_Airline','Operating_Airline','IATA_Code_Operating_Airline',
                                  'Tail_Number','OriginCityName','OriginState','OriginStateName',
                                  'DestCityName','DestState','DestStateName','DepTimeBlk',
                                  'ArrTimeBlk'],inplace=True)
        all_series=[]
        numeric_series=[]
        for i in range(loaded_data.shape[0]):
            numeric_series=loaded_data.iloc[i,:].values
            all_series.append(pd.Series(numeric_series).array)
        return all_series

class ReaderM5(Reader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        #self.fetch()

    def fetch(self):
        # input: path
        # output: DataFrame
        df = self._csv_to_dataframe(self.data_path)
        return df

    def _csv_to_dataframe(self, data_path: str, 
                          replace_missing_vals_with="NaN", 
                          value_column_name="series_value"):

        loaded_data=pd.read_csv(data_path, header=0, sep=',', engine='python')
        loaded_data.drop(columns=['id','item_id','dept_id','cat_id','store_id','state_id'],inplace=True)
        all_series=[]
        numeric_series=[]
        for i in range(loaded_data.shape[0]):
            numeric_series=loaded_data.iloc[i,:].values
            all_series.append(pd.Series(numeric_series).array)
        return all_series
    
class ReaderIllness(Reader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        #self.fetch()

    def fetch(self):
        # input: path
        # output: DataFrame
        df = self._csv_to_dataframe(self.data_path)
        return df

    def _csv_to_dataframe(self, data_path: str, 
                          replace_missing_vals_with="NaN", 
                          value_column_name="series_value"):

        loaded_data=pd.read_csv(data_path, header=0, sep=',', engine='python')
        loaded_data.drop(columns=['date'],inplace=True)
        all_series=[]
        numeric_series=[]
        for i in range(loaded_data.shape[0]):
            numeric_series=loaded_data.iloc[i,:].values
            all_series.append(pd.Series(numeric_series).array)
        return all_series
    
class ReaderSuperstore(Reader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        #self.fetch()

    def fetch(self):
        # input: path
        # output: DataFrame
        df = self._csv_to_dataframe(self.data_path)
        return df

    def _csv_to_dataframe(self, data_path: str, 
                          replace_missing_vals_with="NaN", 
                          value_column_name="series_value"):

        loaded_data=pd.read_csv(data_path, header=0, sep=',', engine='python')
        data=loaded_data.iloc[:,17]
        all_series=[]
        numeric_series=[]
        for i in range(data.shape[0]):
            numeric_series=data.iloc[i,:].values
            all_series.append(pd.Series(numeric_series).array)
        return all_series

class ReaderTemperatureReading(Reader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        #self.fetch()

    def fetch(self):
        # input: path
        # output: DataFrame
        df = self._csv_to_dataframe(self.data_path)
        return df

    def _csv_to_dataframe(self, data_path: str, 
                          replace_missing_vals_with="NaN", 
                          value_column_name="series_value"):

        loaded_data=pd.read_csv(data_path, header=0, sep=',', engine='python')
        data=loaded_data.iloc[:,3]
        all_series=[]
        numeric_series=[]
        for i in range(data.shape[0]):
            numeric_series=data.iloc[i,:].values
            all_series.append(pd.Series(numeric_series).array)
        return all_series

class ReaderPower(Reader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        #self.fetch()

    def fetch(self):
        # input: path
        # output: DataFrame
        df = self._csv_to_dataframe(self.data_path)
        return df

    def _csv_to_dataframe(self, data_path: str, 
                          replace_missing_vals_with="NaN", 
                          value_column_name="series_value"):

        loaded_data=pd.read_csv(data_path, header=0, sep=',', engine='python')
        data=loaded_data.iloc[:,1]
        all_series=[]
        numeric_series=[]
        for i in range(data.shape[0]):
            numeric_series=data.iloc[i,:].values
            all_series.append(pd.Series(numeric_series).array)
        return all_series
    
reader_dict = {
    "FlightDelayPrediction": ReaderFlightDelayPrediction,
    "FlightStatusPrediction": ReaderFlightStatusPrediction,
    "m5": ReaderM5,
    "illness": ReaderIllness,
    "superstore": ReaderSuperstore,
    "temperature": ReaderTemperatureReading,
    "power": ReaderPower
}
