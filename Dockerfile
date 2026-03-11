# 1. Utiliser une image Python légère
FROM python:3.11-slim

# 2. Définir le répertoire de travail
WORKDIR /app

# 3. Copier les fichiers de dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code source et le modèle
# (Assure-toi que ton dossier src/ et models/ sont bien copiés)
COPY src/ ./src/
COPY models/ ./models/

# --- LA LIGNE MAGIQUE ---
ENV PYTHONPATH=/app/src
# ------------------------
    
# 5. Exposer le port sur lequel FastAPI va tourner
EXPOSE 8000

# 6. Lancer l'API avec uvicorn
# On utilise 0.0.0.0 pour que l'API soit accessible hors du container
CMD ["uvicorn", "src.diamonds.api:app", "--host", "0.0.0.0", "--port", "8000"]