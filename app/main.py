# .github/workflows/analyse_sentiments_module-7.yml

name: Build and deploy Python app to Azure Web App - Module-7

on:
  push:
    branches:
      - analyse_sentiments
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Prepare artifact for deployment
        run: |
          mkdir deploy
          # Copier le code de l'API
          cp -r app deploy/
          # Copier le fichier des dépendances
          cp requirements.txt deploy/
          cd deploy
          # Créer l’archive zip qui sera déployée
          zip -r ../release.zip .
  
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: python-app
          path: release.zip

  deploy:
    runs-on: ubuntu-latest
    needs: build
    permissions:
      contents: read
      id-token: write

    steps:
      - name: Download artifact
        uses: actions/download-artifact@v4
        with:
          name: python-app

      - name: Unzip artifact
        run: |
          unzip release.zip -d deploy_content

      - name: Azure Login
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      - name: Deploy to Azure Web App
        uses: azure/webapps-deploy@v3
        with:
          app-name: Module-7
          slot-name: production
          publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
          package: deploy_content

      - name: Configure Startup Command
        uses: azure/CLI@v1
        with:
          azcliversion: latest
          inlineScript: |
            az webapp config set \
              --name Module-7 \
              --resource-group ${{ secrets.AZURE_RESOURCE_GROUP }} \
              --startup-file "gunicorn app.main:app --workers 1 --bind=0.0.0.0:8000"
