from ltsm.data_reader.csv_reader import CSVReader,transform_csv, transform_csv_dataset
import pytest
from pandas.api.types import is_float_dtype
import os
import pandas as pd
import numpy as np

def test_csv_reader_NA(tmp_path):
    d = tmp_path / "na_test.csv"
    d.write_text("0,1,2,3,4,5,6\n1.,,2.,,,5.,5.\n2.,,4.,,6.,8.,\n1.,1.,1.,1.,,1.,1.\n")
    csv_reader = CSVReader(str(d))
    df = csv_reader.fetch()
    assert df.iloc[0, 1] == 1.5
    assert df.iloc[0, 3] == 3
    assert df.iloc[0, 4] == 4
    assert df.iloc[1, 1] == 3
    assert df.iloc[1, 3] == 5
    assert df.iloc[1, 6] == 8
    assert df.iloc[2, 4] == 1
    assert df.isna().to_numpy().sum() == 0

def test_csv_reader_columns(tmp_path):
    d = tmp_path / "col_names_test.csv"
    d.write_text("0,1,2.2,2016-08-25 15:32:00,2016-08-25 15:33:00,LABEL,05-13-2023\n,1.,1.,1.,1.,1.,1.,hi\n")
    csv_reader = CSVReader(str(d))
    df = csv_reader.fetch()
    assert len(df.columns) == 4
    assert (df.columns.values == ["0","1","2016-08-25 15:32:00","2016-08-25 15:33:00"]).all()
    for col in df.columns:
        assert is_float_dtype(df[col])

def test_invalid_csv_path():
    data_path = "invalid/path/to/data.csv"
    csv_reader = CSVReader(data_path)
    with pytest.raises(FileNotFoundError):
        csv_reader.fetch()

def test_empty_csv(tmp_path):
    d = tmp_path / "empty.csv"
    d.write_text("")
    csv_reader = CSVReader(str(d))
    with pytest.raises(ValueError):
        csv_reader.fetch()

def test_improper_csv(tmp_path):
    d = tmp_path / "improper.csv"
    d.write_text('''0,1,2,3,4,5\n,0.,1.,2,3,4,"5''')
    csv_reader = CSVReader(str(d))
    with pytest.raises(ValueError):
        csv_reader.fetch()

def test_is_datetime(tmp_path):
    d = tmp_path / "dummy.csv"
    csv_reader = CSVReader(str(d))
    assert csv_reader._CSVReader__is_datetime("2001-01-01") == True
    assert csv_reader._CSVReader__is_datetime("NotADate") == False

@pytest.fixture
def setup_csv_data(mocker):
    # 准备输入的 DataFrame 和期望的输出
    dfs = [
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In'],
            ['6/30/2023 19:01:24', 0, 1, 98.34],
            ['6/30/2023 19:03:04', 0, 3, 98.93],
            ['6/30/2023 19:04:44', 0, 2, 97.90],
            ['6/30/2023 19:06:22', 2, 3, 98.37],
            ['6/30/2023 19:08:03', 3, 1, 98.37]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out'],
            ['6/30/2023 19:09:43', 1, 2, 98.43, 109.31],
            ['6/30/2023 19:11:23', 1, 3, 97.64, 109.18],
            ['6/30/2023 19:13:02', 1, 4, np.nan, 109.18]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp'],
            ['6/30/2023 19:01:24', 61.71, 102.75],
            ['6/30/2023 19:03:04', 69.21, 103.19],
            ['6/30/2023 19:04:44', 71.35, 103.34],
            ['6/30/2023 19:06:22', 68.14, 102.60],
            ['6/30/2023 19:08:03', 83.47, 103.26],
            ['6/30/2023 19:09:43', 63.14, 103.85]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out'],
            ['6/30/2023 19:01:24', 61.71, 102.75, np.nan, 109.73],
            ['6/30/2023 19:03:04', np.nan, 103.19, 98.93, 109.73],
            ['6/30/2023 19:04:44', 71.35, 103.34, np.nan, 108.87],
            ['6/30/2023 19:06:22', np.nan, 102.60, 98.37, np.nan]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out', 'Additional Column'],
            ['6/30/2023 19:01:24', 61.71, 102.75, 98.34, 109.73, 1],
            ['6/30/2023 19:03:04', 69.21, 103.19, 98.93, 109.73, 2],
            ['6/30/2023 19:04:44', 71.35, 103.34, 97.90, 108.87, 3],
            ['6/30/2023 19:06:22', 68.14, 102.60, 98.37, 109.48, 4],
            ['6/30/2023 19:08:03', 83.47, 103.26, 98.37, 109.44, 5]
        ])
    ]

    dfs_expected = [
        pd.DataFrame({
            0: [0, 1, 98.34],
            1: [0, 3, 98.93],
            2: [0, 2, 97.90],
            3: [2, 3, 98.37],
            4: [3, 1, 98.37]
        }),
        pd.DataFrame({
            0: [1, 2, 98.43, 109.31],
            1: [1, 3, 97.64, 109.18],
            2: [1, 4, 0, 109.18]
        }),
        pd.DataFrame({
            0: [61.71, 102.75],
            1: [69.21, 103.19],
            2: [71.35, 103.34],
            3: [68.14, 102.60],
            4: [83.47, 103.26],
            5: [63.14, 103.85]
        }),
        pd.DataFrame({
            0: [61.71, 102.75, 0, 109.73],
            1: [0, 103.19, 98.93, 109.73],
            2: [71.35, 103.34, 0, 108.87],
            3: [0, 102.6, 98.37, 0]
        }),
        pd.DataFrame({
            0: [61.71, 102.75, 98.34, 109.73, 1],
            1: [69.21, 103.19, 98.93, 109.73, 2],
            2: [71.35, 103.34, 97.9, 108.87, 3],
            3: [68.14, 102.6, 98.37, 109.48, 4],
            4: [83.47, 103.26, 98.37, 109.44, 5]
        })
    ]

    return dfs, dfs_expected


def test_transform_csv(setup_csv_data, mocker):
    dfs, dfs_expected = setup_csv_data
    for i, (input_df, expected_df) in enumerate(zip(dfs, dfs_expected)):
        mocker.patch('pandas.read_csv', return_value=input_df)
        result_df = transform_csv(f'file{i+1}.csv')
        #assert (result_df.iloc[0, :] == range(len(result_df.columns))).all(), "Time sequence conversion failed." Q to ask
        try:
            pd.testing.assert_frame_equal(result_df, expected_df,
                                      check_dtype=False,)
        except AssertionError as e:
            raise AssertionError("Data transformation did not produce the expected output.") from e


def test_transform_csv_folder(mocker):
    """ This test case tests the transform_csv_dataset function with different inputs. """
    
    mocker.patch('os.path.exists', side_effect=lambda path: path == './input_folder')
    mocker.patch('os.listdir', return_value=['file1.csv', 'file2.csv'])
    
    mock_transform_csv = mocker.patch('ltsm.data_reader.csv_reader.transform_csv')
    mock_to_csv = mocker.patch('pandas.DataFrame.to_csv')
    mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mock_transform_csv.return_value = mock_df
    mock_makedirs = mocker.patch('os.makedirs')

    transform_csv_dataset('./input_folder', './output_folder')
    mock_makedirs.assert_called_once_with('./output_folder')
    assert mock_transform_csv.call_count == 2  # check if transform_csv is called twice

    expected_calls = [
        mocker.call(os.path.join('./output_folder', 'file1.csv'), index=False),
        mocker.call(os.path.join('./output_folder', 'file2.csv'), index=False)
    ]
    mock_to_csv.assert_has_calls(expected_calls, any_order=True)