# Use an official CUDA base image with Jupyter pre-installed
FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter Notebook
RUN pip3 install --no-cache-dir jupyter

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Add project files
COPY /bajkogenerator /app/bajkogenerator
COPY README.md /app/README.md

# Add startup script
COPY /docker/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Expose the Jupyter and Gradio ports
EXPOSE 8888
EXPOSE 7860

# Set environment variables
ENV CUDA_VISIBLE_DEVICES all

# Start Jupyter Notebook
ENTRYPOINT ["/app/startup.sh"]
