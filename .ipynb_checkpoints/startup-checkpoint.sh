#!/bin/bash

# Lancer FastAPI en arrière-plan
gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:80

# Lancer Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
