import wandb
import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--downloaded_raw_data", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    # Download input artifact. This script uses this particular
    # version
    data_path = run.use_artifact(
        request.config.option.downloaded_raw_data).file()
    if data_path is None:
        pytest.fail("Provide the --csv option on the command line")
    df = pd.read_csv(data_path)
    return df
