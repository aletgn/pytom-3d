#!/bin/bash

# Set the path to your .whl file
WHL_FILE=./dist/pytom3d-0.0.2-py3-none-any.whl

# Set the desired path for the virtual environment
VENV_DIR=$HOME/pythonEnv/simulation

# Specify the Python version (change this to your desired version)
# system-wide 3.8.10
#PYTHON_VERSION=3.6.5
PYTHON_VERSION=3.8.10

# Check if the virtual environment directory exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create a new virtual environment
virtualenv --python="$PYTHON_VERSION" "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install the wheel file using pip
pip3 install "$WHL_FILE"

# Display a message indicating successful setup
echo "Virtual environment created and package installed successfully."

# Optionally, deactivate the virtual environment
#deactivate
