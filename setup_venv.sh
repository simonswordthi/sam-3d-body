#!/bin/bash
# Script to set up a virtual environment on Linux and macOS

# Check if Python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install it to continue."
    exit
fi

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# For Linux
if [ "$OSTYPE" == "linux-gnu" ]; then
    source venv/bin/activate
# For macOS
elif [ "$OSTYPE" == "darwin" ]; then
    source venv/bin/activate
else
    echo "Unsupported OS. Please activate the virtual environment manually."
    exit
fi

echo "Virtual environment setup complete. To activate it, run 'source venv/bin/activate'"