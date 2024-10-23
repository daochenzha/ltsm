from ltsm.data_reader.csv_reader import CSVReader
import pytest
from pandas.api.types import is_float_dtype

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