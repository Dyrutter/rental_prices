import wandb
import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    # Download input artifact. This script uses this particular
    # version
    data_path = run.use_artifact(
        request.config.option.sample_artifact).file()
    if data_path is None:
        pytest.fail("Provide the --csv option on the command line")
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    data_path = run.use_artifact(
        request.config.option.reference_artifact).file()
    if data_path is None:
        pytest.fail("Provide the --ref option on the command line")
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold
    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")
    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price
    if min_price is None:
        pytest.fail("You must provide min_price")
    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price
    if max_price is None:
        pytest.fail("You must provide max_price")
    return float(max_price)
