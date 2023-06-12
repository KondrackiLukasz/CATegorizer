# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y gfortran && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y libgl1-mesa-glx && \
    pip install --trusted-host pypi.python.org -r requirements.txt

# Install uvicorn
RUN pip install uvicorn

# Expose port 8080 for the FastAPI server
EXPOSE 8080

# Define the command that starts the FastAPI server
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
