#!/bin/bash

# Start the FastAPI backend in the background
echo "Starting FastAPI Backend..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# Wait a few seconds to let the ML model load into memory
sleep 10 

# Start the Streamlit frontend on Hugging Face's required port
echo "Starting Streamlit Frontend..."
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0