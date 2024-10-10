import taosws
import pandas as pd
import os

# input data path
datapath = "input_data"

# output data path
output_folder = 'output'
# database name
database = "time_series_demo"

# create_connection() function to connect to the database.
def create_connection(host, port):
    conn = None

    try:
        conn = taosws.connect(
            user="root",
            password="taosdata",
            host=host,
            port=port,
        )
        print(f"Connected to {host}:{port} successfully.")
        return conn
    except Exception as err:
        print(f"Failed to connect to {host}:{port}, ErrMessage: {err}")
        raise err


def setup_database(conn, database):
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        print("Database time_series_demo set up successfully.")
    except Exception as err:
        print(f"Error setting up database: {err}")
        raise err


def setup_tables(conn, database, table_name, df):
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE {database}")
        # Dynamically create schema based on CSV column names
        columns = df.columns
        schema_columns = ["ts TIMESTAMP"]
        for column in columns[1:]:
            schema_columns.append(f"{column.replace(' ', '_')} FLOAT")

        schema = f"({', '.join(schema_columns)})"
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} {schema}")
        print(f"Table {table_name} set up successfully with schema: {schema}")
    except Exception as err:
        print(f"Error setting up database or table {table_name}: {err}")
        raise err


def insert_data_from_csv(conn, database, csv_file, table_name):
    try:
        cursor = conn.cursor()
        df = pd.read_csv(csv_file)
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="%m/%d/%Y %H:%M:%S")

        setup_tables(conn, database, table_name, df)
        # Dynamically create SQL INSERT queries based on CSV data
        for _, row in df.iterrows():
            values = [f"'{row[df.columns[0]]}'"] + [str(row[col]) for col in df.columns[1:]]
            cursor.execute(f"USE {database}")
            cursor.execute(f"INSERT INTO {table_name} VALUES ({', '.join(values)})")

        print(f"Data from {csv_file} inserted into {table_name}.")
    except Exception as err:
        print(f"Error inserting data from {csv_file} into {table_name}: {err}")
        raise err


def retrieve_data_to_csv(conn, database, table_name, output_file):
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE {database}")
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()

        # Retrieve column names dynamically
        cursor.execute(f"DESCRIBE {table_name}")
        columns = [desc[0] for desc in cursor.fetchall()]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"Data from {table_name} saved to {output_file}.")
    except Exception as err:
        print(f"Error retrieving data from {table_name}: {err}")
        raise err


def main():
    host="35.153.211.255"
    port=6041

    conn = create_connection(host, port)

    if conn:
        try:
            setup_database(conn, database)
            csv_files = [os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith('.csv')]
            # Generate table names by removing ".csv" from filenames
            tables = [os.path.splitext(os.path.basename(csv_file))[0] for csv_file in csv_files]

            for csv_file, table_name in zip(csv_files, tables):
                insert_data_from_csv(conn, database, csv_file, table_name)

            # Output folder

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for table_name in tables:
                output_file = os.path.join(output_folder, f"{table_name}.csv")
                retrieve_data_to_csv(conn, database, table_name, output_file)

        finally:
            conn.close()


if __name__ == "__main__":
    main()
