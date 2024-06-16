import subprocess
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Run black
# black  --config pyproject.toml  src
try:
    subprocess.run(['black', '--config', 'pyproject.toml', 'src'], check=True)
    print('black completed successfully')
except subprocess.CalledProcessError as e:
    print('black failed')

# Run pylint
try:
    subprocess.run(['pylint', 'src'], check=True)
    print('pylint completed successfully')
except subprocess.CalledProcessError as e:
    print('pylint failed')

# Run mypy
try:
    subprocess.run(['mypy', '.'], check=True)
    print('mypy completed successfully')
except subprocess.CalledProcessError as e:
    print('mypy failed')


# Run pytest
try:
    subprocess.run(['pytest', '--cov=.', '--cov-report=html', '-p', 'no:checkdocs', 'tests/'], check=True)
    print('pytest completed successfully')
except subprocess.CalledProcessError as e:
    print('pytest failed')