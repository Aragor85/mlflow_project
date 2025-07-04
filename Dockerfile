# Utilise une image Python l�g�re
FROM python:3.10-slim

# D�finir le r�pertoire de travail
WORKDIR /app

# Copier les fichiers dans l'image Docker
COPY . .

# Installer les d�pendances
RUN pip install --upgrade pip && pip install -r requirements.txt

# Donner les permissions d'ex�cution au script de d�marrage
RUN chmod +x startup.sh

# Exposer uniquement le port 80 pour Azure
EXPOSE 80

# Lancer le script de d�marrage (FastAPI + Streamlit)
CMD ["./startup.sh"]

