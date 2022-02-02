import sys
from pathlib import Path

def pytest_sessionstart(session):
    # TODO: Restructure and Remove
    project_dir = Path().absolute()
    sys.path.insert(0, str(project_dir))
