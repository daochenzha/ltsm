import unittest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from ltsm.data_reader.csv_reader import transform_csv, transform_csv_dataset  
from tests.data_reader.csv_reader_DK_testcases import DK_TestCases_input, DK_TestCases_Output

# python -m unittest tests.data_reader.test_csv_reader  # run this command to test csv_reader.py

class TestCSVTransform(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_transform_csv(self, mock_read_csv):
        """ This test case tests the transform_csv function with different inputs.
        """
        dfs = DK_TestCases_input()
        dfs_expected= DK_TestCases_Output()
        mock_read_csv.side_effect = dfs
        file_paths = ["file" + str(i) + ".csv" for i in range(1, len(dfs) + 1)]

        for file_path, expected_df in zip(file_paths, dfs_expected):
            result_df = transform_csv(file_path)
            pd.testing.assert_frame_equal(result_df, expected_df)

       
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')  # Mock to_csv function
    @patch('ltsm.data_reader.csv_reader.transform_csv')  # Mock the transform_csv function
    def test_transform_csv_folder(self, mock_transform_csv, mock_to_csv, mock_makedirs, mock_path_exists, mock_listdir):
        """ This test case tests the transform_csv_dataset function with different inputs.
        """
        mock_path_exists.side_effect = lambda path: path == './input_folder'  
        mock_listdir.return_value = ['file1.csv', 'file2.csv']
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_transform_csv.return_value = mock_df

        transform_csv_dataset('./input_folder', './output_folder')

        mock_makedirs.assert_called_once_with('./output_folder')
        self.assertEqual(mock_transform_csv.call_count, 2)

        expected_output_calls = [
            call(os.path.join('./output_folder', 'file1.csv'), index=False),
            call(os.path.join('./output_folder', 'file2.csv'), index=False)
        ]
        mock_to_csv.assert_has_calls(expected_output_calls, any_order=True)

if __name__ == '__main__':
    unittest.main()
