import unittest
from unittest.mock import patch, MagicMock
import database_connector as db_connector
import pandas as pd

class TestDatabaseConnector(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing table setup with corrected datetime format
        self.input_df = pd.DataFrame({
            'Updated Time': ['06/30/2023 19:01:24', '06/30/2023 19:03:04','06/30/2024 19:03:04'],
            'Top Temperature': [61.71, 69.21, 87.31],
            'Motor Temperature': [98.34, 98.11, 99.22]
        })
        self.database = "test_database"
        self.table_name = "test_table"

    @patch('database_connector.create_connection')
    def test_create_connection(self, mock_create_connection):
        # Mock successful connection
        mock_conn = MagicMock()
        mock_create_connection.return_value = mock_conn

        # Test the connection function
        conn = db_connector.create_connection("localhost", 6030)
        self.assertIsNotNone(conn)
        mock_create_connection.assert_called_once_with("localhost", 6030)

    @patch('database_connector.create_connection')
    def test_setup_database(self, mock_create_connection):
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_create_connection.return_value = mock_conn

        # Call the function to be tested
        db_connector.setup_database(mock_conn, self.database)

        # Check if the correct SQL command was executed
        mock_cursor.execute.assert_called_once_with(f"CREATE DATABASE IF NOT EXISTS {self.database}")

    @patch('database_connector.create_connection')
    def test_setup_tables(self, mock_create_connection):
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_create_connection.return_value = mock_conn

        # Call the function to be tested
        db_connector.setup_tables(mock_conn, self.database, self.table_name, self.input_df)

        # Check if the correct SQL commands were executed
        mock_cursor.execute.assert_any_call(f"USE {self.database}")
        expected_schema = "(ts TIMESTAMP, Top_Temperature FLOAT, Motor_Temperature FLOAT)"
        mock_cursor.execute.assert_any_call(f"CREATE TABLE IF NOT EXISTS {self.table_name} {expected_schema}")

    @patch('database_connector.create_connection')
    @patch('database_connector.pd.read_csv')
    def test_insert_data_from_csv(self, mock_read_csv, mock_create_connection):
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_create_connection.return_value = mock_conn

        # Mock reading CSV
        mock_read_csv.return_value = self.input_df

        # Update the format of datetime parsing to match the CSV data
        db_connector.insert_data_from_csv(mock_conn, self.database, "dummy_path.csv", self.table_name)

        # Check if setup_tables was called
        mock_cursor.execute.assert_any_call(f"USE {self.database}")
        expected_schema = "(ts TIMESTAMP, Top_Temperature FLOAT, Motor_Temperature FLOAT)"
        mock_cursor.execute.assert_any_call(f"CREATE TABLE IF NOT EXISTS {self.table_name} {expected_schema}")

        # Check if data insertion commands were executed
        self.assertTrue(mock_cursor.execute.called)
        self.assertEqual(mock_cursor.execute.call_count, len(self.input_df)+3)

if __name__ == '__main__':
    unittest.main()