import data_preprocessing

def test_str_split():
    assert data_preprocessing.str_split(" Dever 3:01, 1234")  == ("3:01", 1234.0)
