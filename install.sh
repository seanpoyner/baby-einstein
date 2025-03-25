#!/bin/bash

# Install script for the Baby Einstein project

echo "Starting installation..."

# Update package list and install Python3 and pip if necessary
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install python3 python3-pip -y
    echo "Python3 installed."
else
    echo "Python3 is already installed."
fi

# Install virtualenv if not present
if ! command -v virtualenv &> /dev/null
then
    echo "Installing virtualenv..."
    pip3 install virtualenv
    echo "virtualenv installed."
else
    echo "virtualenv is already installed."
fi

# Create a virtual environment
if [ ! -d "env" ]; then
    echo "Creating a virtual environment..."
    virtualenv env
    echo "Virtual environment created."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Install the required packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
    echo "All packages installed."
else
    echo "requirements.txt file not found!"
    exit 1
fi

# Confirmation message
echo "Installation completed successfully! Run the application using 'python albert/main.py'."

# Deactivate the virtual environment
deactivate
echo "Virtual environment deactivated."
