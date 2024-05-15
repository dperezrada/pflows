import subprocess
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Run mypy
subprocess.run(['mypy', '.'], check=True)
print('mypy completed successfully')

# Run black
# black  --config pyproject.toml  src
subprocess.run(['black', '--config', 'pyproject.toml', 'src'], check=True)

# Run pylint
subprocess.run(['pylint', 'src'], check=True)
print('pylint completed successfully')

# Run pytest
subprocess.run(['pytest', '--cov=.', '--cov-report=html', '-p', 'no:checkdocs', 'tests/'], check=True)
print('pytest completed successfully')