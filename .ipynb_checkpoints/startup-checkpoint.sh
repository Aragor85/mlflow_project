#!/bin/bash

# FastAPI tourne en fond
gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 &

# Streamlit est l'interface principale (donc sur le port 80)
streamlit run streamlit_app.py --server.port 80 --server.address 0.0.0.0
