# 🏗️ Base Image: CPU-only version of PyTorch
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install missing system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Make sure Flask listens on all interfaces
ENV PYTHONUNBUFFERED=1

# Run the Flask app
CMD ["python", "app.py"]
