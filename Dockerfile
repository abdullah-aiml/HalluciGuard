# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Hugging Face requires the app to run as user '1000'
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY --chown=user . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the startup script executable
RUN chmod +x run.sh

# Expose the port Hugging Face looks for
EXPOSE 7860

# Command to run the dual-server script
CMD ["./run.sh"]