import pytest


def pytest_addoption(parser):
    parser.addoption("--runml", action="store_true", default=False,
                     help="run tests marked `ml` (needs torch/transformers)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runml"):
        return
    skip = pytest.mark.skip(reason="need --runml and torch/transformers installed")
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip)
