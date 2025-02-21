import pytest


@pytest.fixture(scope="session", autouse=True)
def init():
    pass
