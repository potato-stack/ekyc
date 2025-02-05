# Use the full official Python 3.12 image (instead of the slim version)
FROM python:3.12-slim

# Install necessary system dependencies like OpenGL and Glib
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /ekyc

# Copy only the requirements.txt first to cache pip dependencies
COPY requirements.txt /ekyc/

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /ekyc

# Specify the command to run your application
CMD ["python", "main.py"]

EXPOSE 8000
