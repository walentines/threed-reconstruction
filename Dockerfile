# Use a Miniconda base image with CUDA (if needed)
FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /ml

# Install required system packages (CUDA for GPU support)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone your repository
RUN git clone https://github.com/walentines/threed-reconstruction.git .

# Create a Conda environment from the environment.yml file
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate Conda environment and ensure it is available in the container
SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]

# Expose the port for FastAPI
EXPOSE 8000

# Run the FastAPI server
CMD ["conda", "run", "-n", "your_env_name", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
