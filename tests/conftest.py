import os

import pytest


@pytest.fixture(scope="session")
def rom_path():
    path = os.environ.get("NETRIS_ROM")
    if not path or not os.path.exists(path):
        pytest.skip("set NETRIS_ROM to run tests that need the ROM")
    return path
