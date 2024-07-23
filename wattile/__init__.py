import toml
from os import path

b_path = path.dirname(__file__)
proj_path = path.abspath(path.join(b_path, '..', 'pyproject.toml'))

with open(proj_path, 'r') as f:
    config = toml.load(f)
    version = config['tool']['poetry']['version']

