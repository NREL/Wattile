import pytest
import json
import pathlib
import entry_point as epb


TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"

@pytest.fixture
def config_for_tests():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    return configs
    
    
def test_create_input_dataframe(config_for_tests, tmpdir):
    # patch configs and create temporary, unquie output file
    config_for_tests["results_dir"] = tmpdir / "train_results"

    # creat data frame
    data = epb.create_input_dataframe(config_for_tests)

    excepted_data_columns = []
    # add weather columns
    for pred in config_for_tests["weather_include"]:
        excepted_data_columns += [f"{pred}_{m}" for m in ["max", "min", "mean"]]
        excepted_data_columns += [f"{pred}_{m}_lag{l + 1}" for m in ["max", "min", "mean"] for l in range(config_for_tests["window"])]

    # add year and hour columns
    excepted_data_columns += [
        f"{f}_{t}_{m}"
        for t in ["MOY", "HOD"] 
        for f in ["cos", "sin"] 
        for m in ["max", "min", "mean"]
    ]
    excepted_data_columns += [
        f"{f}_{t}_{m}_lag{l + 1}" 
        for t in ["MOY", "HOD"] 
        for f in ["cos", "sin"] 
        for m in ["max", "min", "mean"] 
        for l in range(config_for_tests["window"])
    ]

    # add week columns
    excepted_data_columns += [
        f"DOW_binary_reg_{i}_{m}" 
        for i in range(0,7) 
        for m in ["max", "min", "mean"]
    ]
    excepted_data_columns += [
        f"DOW_binary_reg_{i}_{m}_lag{l + 1}" 
        for i in range(0,7) 
        for m in ["max", "min", "mean"] 
        for l in range(config_for_tests["window"])
    ]

    # add target var
    excepted_data_columns.append(config_for_tests["target_var"])

    assert set(data.columns) == set(excepted_data_columns)
    assert data.shape == (660, len(excepted_data_columns))
