# Analyse de Sentiments grâce au Deep Learning avec l'approche MLOps

> Cet article est disponible en ligne : [xxxxxxxxxxxxx](xxxxxxxxxxxxxxxx)

![Les sentiments a travers les Tweet](images/Tweet.png)

*Cet article a été rédigé dans le cadre du projet : Réalisez une analyse de sentiments grâce au Deep Learning du parcours [AI Engineer](https://openclassrooms.com/fr/paths/795-ai-engineer). Les données utilisées sont issues du jeu de données open source [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140). Le code source complet est disponible sur [(https://github.com/Aragor85/mlflow_projectGitHub)]*

> 🎓 OpenClassrooms • Parcours [AI Engineer](https://openclassrooms.com/fr/paths/795-ai-engineer) | 👋 *Étudiant* : Djamel FERGUEN

![API: Analyse des sentiments a travers les Tweet](images/Tweet.png)


## 🌐 Contexte et problématique métier 

Ce projet s'inscrit dans un scénario professionnel où j'interviens en tant qu'ingénieur IA chez MIC (Marketing Intelligence Consulting), entreprise de conseil spécialisée sur les problématiqus de marketing digital.

Notre client,  **Air Paradis** (compagnie aérienne), souhaite **anticiper les bad buzz sur les réseaux sociaux**. La mission consiste à développer un produit IA permettant de prédire le sentiment associé à un tweet, afin d'améliorer son image de marque en ligne.

## ⚡ Mission

> Développer un modèle d'IA permettant de prédire le sentiment associé à un tweet.

Créer un prototype fonctionnel d'un modèle d'analyse de sentiments pour tweets selon trois approches différentes :

1. **Modèle simple** : Approche classique (régression logistique,Randomforest,LightGBM) pour une prédiction rapide
2. **Modèle avancé** : Utilisation de réseaux de neurones profonds avec différents word embeddings ( USE, Bidirectional_LSTM et BERT)
3. **Modèle avancé BERT** : Le modèle BERT est bien intégré dans le projet. Cependant, en raison de limitations matérielles (notamment l'absence de GPU et une configuration uniquement sur CPU), l'entraînement s'est avéré extrêmement lent. Face à un temps de calcul estimé à 10 heures par Epoch, j'ai décidé d'interrompre l'exécution du modèle

Cette mission implique également la mise en place d'une **démarche MLOps complète pour le deploiment sur le Cloud** :

- Utilisation de **MLFlow pour le tracking des expérimentations et le stockage des modèles**.
- Création d'un **pipeline de déploiement continu (Git + Github + plateforme Cloud Azure)**.
- Intégration de **tests unitaires automatisés**.
- Mise en place d'un **suivi de performance du modéle en production** via Azure A[pplication Insight](https://learn.microsoft.com/fr-fr/azure/azure-monitor/app/app-insights-overview).

## 🔧 Environnement technique

- **Distribution** : Anaconda ver. XX.XX
- **Langages** : Python ver. 3.10
- **Bibliothèques ML/DL** : Scikit-learn, TensorFlow/Keras, Transformers (BERT),  **Ajoute USE LSTM,......**
- **MLOps** : MLFlow, Git, GitHub Actions
- **Backend** : FastAPI
- **Frontend** : Next.js / React   
- **Monitoring** : Azure Application Insight
- **Traitement texte** : NLTK, Word Embeddings

## 🏛️ Structure du projet

```
📦 mlflow_project/
┣━━ 📂 app/
┃   ┣━━ 📂 model/                                   # Backend API de prédiction
┃       ┗━━ 📃 analyse_sentiments_module-7.yml      # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃       ┗━━ 📃 analyse_sentiments_module-7.yml      # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃       ┗━━ 📃 analyse_sentiments_module-7.yml      # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃       ┗━━ 📃 analyse_sentiments_module-7.yml      # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃       ┗━━ 📃 analyse_sentiments_module-7.yml      # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights

┣━━ 📂 .github/
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights

┣━━ 📂 data/
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┣━━ 📂 docs/
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┣━━ 📂 images/
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┣━━ 📂 mlruns/
┃   ┣━━ 📂 0/                                       # Backend API de prédiction
┃       ┗━━ 📂 frontend/                            # Application Next.js
┃       ...
        ┗━━ 📂 frontend/                            # Application Next.js
┃       
┣━━ 📂 models/
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
    ...  
┃   ┗━━ 📃 analyse_sentiments_module-7.yml          # Guide de suivi des feedback utilisateur et des alertes avec Azure Application insights
┗━━ 📂 notebooks/                                   # Notebooks Jupyter pour l'analyse et modèles
    ┣━━ 📝 01_Analyse_exploratoire.ipynb            # Exploration et visualisation des données
    
┗━━ 📝 04_Modele_BERT.ipynb                         # DistilBERT pour analyse de sentiment
┗━━ 📝 04_Modele_BERT.ipynb                         # DistilBERT pour analyse de sentiment
┗━━ 📝 04_Modele_BERT.ipynb                         # DistilBERT pour analyse de sentiment
┗━━ 📝 04_Modele_BERT.ipynb                         # DistilBERT pour analyse de sentiment
┗━━ 📝 04_Modele_BERT.ipynb                         # DistilBERT pour analyse de sentiment

## 📔 Notebooks du projet

- [📊 Notebook 1 : Analyse exploratoire des données]  link to notebook

## 🧭 Guides

- Help pour utilisation de l'API !!!! 

## 📑 Méthodologie et données

### Le jeu de données Sentiment140

Pour ce projet, nous avons utilisé le jeu de données open source Sentiment140, qui contient 1,6 million de tweets annotés (négative ou positive). Ce dataset comprend six champs principaux :

- **target** : la polarité du tweet (0 = négatif, 1 = positif)
- **ids** : l'identifiant du tweet
- **date** : la date du tweet
- **flag** : une requête éventuelle
- **user** : l'utilisateur ayant posté le tweet
- **text** : le contenu textuel du tweet

J'ai choisi de réduire la taille du dataset a 16 000 tweets pour la suite du projet (configuration materiéls).

!!!!  reduction de la taille du dataset 

### Analyse exploratoire des données Sentiment140

Notre analyse exploratoire a révélé des caractéristiques distinctives importantes entre les tweets positifs et négatifs :

- XX%  de tweets positifs
- XX%  de tweets négatifs

équilibrés pas de smote 

### Prétraitement des données textuelles

Un petit paragraphe pour décrire et surtout vérification ce que j'ai fait dans le premier Notebook (stratégie de prétraitement en 3 ou 4 points clés) :   

## 🧠 Approches de modélisation

Pour répondre à la demande d'Air Paradis, nous avons développé et comparé 5 approches de modélisation distinctes, de la plus simple à la plus avancée.

### Modèle classique
- Logistic regression
- Randomforest
- LightGBM


Notre première approche s'est basée sur des techniques classiques de machine learning, combinant une vectorisation du texte avec un classifieur traditionnel :

1. **Vectorisation** : transformation des textes en représentations numériques via TF-IDF (Term Frequency-Inverse Document Frequency)
2. **Classification** : utilisation d'un Randomforest,LightGBM ou Régression Logistique pour prédire le sentiment

Cette approche présente plusieurs avantages :
- Rapidité d'entraînement et d'inférence
- Faible empreinte mémoire
- Bonne interprétabilité des résultats

Malgré sa simplicité, ce modèle a atteint une précision (accuracy) de XX% sur notre jeu de test, ce qui constitue une base solide pour la détection de sentiments.

### Modèles avancé (réseaux de neurones avec word embeddings)

- USE
- Bidirectional_LSTM
- distilbert-base-uncased

Pour notre deuxième approche, nous avons exploré les techniques de deep learning avec des embeddings de mots et des réseaux de neurones récurrents :

Un petit paragraphe pour décrire le prétraitement en 3 ou 4 points clés) :   

Faut-il ajouter quelques morceau de code des differents models ? 


L'architecture de notre modèle LSTM comprend :

Un petit descrptif avec graphe Accuracy et loss ( Test 10 epoch et non pas 4 comme dans mlflow UI )
Ajoute courbe d'apprentissage voir mlruns 

**L'architecture de notre modèle USE comprend** :

**L'architecture de notre modèle BERT comprend** :

Les résultats de l'entraînement montrent une progression constante avec de l'accuracyes/48j9lz9bh84os9nkp2bz.png)

Cette approche plus sophistiquée nous a permis d'atteindre une précision de 81,8% sur l'ensemble de validation, avec un score de 85,2% sur le jeu d'entraînement, surpassant ainsi le modèle simple.

### Modèle BERT (approche transformer)

Pour notre troisième approche, nous avons exploré l'état de l'art en NLP en utilisant BERT (Bidirectional Encoder Representations from Transformers) :

1. **Modèle pré-entraîné** : nous avons utilisé DistilBERT, une version allégée et distillée de BERT, pour réduire les coûts de calcul tout en maintenant des performances élevées
2. **Fine-tuning** : nous avons affiné le modèle sur notre jeu de données spécifique d'analyse de sentiments

Pour cette approche, nous avons utilisé le modèle `DistilBertForSequenceClassification` de la bibliothèque Hugging Face, qui est spécifiquement conçu pour les tâches de classification de séquences textuelles :

```S
```


### Comparaison des performances des modèles

Voici un récapitulatif des performances obtenues avec nos différentes approches :

| Modèle | Précision (Accuracy) | F1-Score | Temps d'entraînement | Taille du modèle |
|--------|----------------------|----------|---------------------|-----------------|
| Régression Logistique + TF-IDF | xx,xx% | xx,xx | xx secondes | ~xx MB |
| Randomforest + TF-IDF | xx,xx% | xx,xx | xx secondes | ~xx MB |
| LightGBM + TF-IDF | xx,xx% | xx,xx | xx secondes | ~xx MB |
| USE | xx,xx% | xx,xx | xx secondes (GPU) | ~xx MB |
| Bidirectional_LSTM | xx,xx% | xx,xx | xx min | ~xx MB |
| BERT | --% | -- | -- | ~--- MB |

Pour le déploiement en production, nous avons retenu le modèle **USE**, qui offre le meilleur compromis entre performance et ressources requises.  et plus adapté à un déploiement sur une infrastructure Cloud gratuite.

## ⚙️ Mise en œuvre du MLOps

### Principes du MLOps

**Le MLOps (Machine Learning Operations) est une méthodologie qui vise à standardiser et à automatiser le cycle de vie des modèles de machine learning**, de leur développement à leur déploiement en production. Pour ce projet, nous avons mis en œuvre plusieurs principes clés du MLOps :

1. **Reproductibilité** : environnement de développement versionné et documenté
2. **Automatisation** : pipeline de déploiement continu
3. **Monitoring** : suivi des performances du modèle en production
4. **Amélioration continue** : collecte de feedback et réentraînement périodique

Cette approche nous a permis de créer une solution robuste et évolutive pour Air Paradis.

### Tracking des expérimentations avec MLFlow

Pour assurer une gestion efficace des expérimentations, nous avons utilisé [MLFlow](https://mlflow.org/docs/latest/index.html), un outil open-source spécialisé dans le **suivi et la gestion des modèles de machine learning** :

1. **Tracking des métriques** : pour chaque expérimentation, nous avons enregistré automatiquement les paramètres du modèle, les métriques de performance (accuracy, F1-score, précision, rappel) et les artefacts générés
2. **Centralisation des modèles** : tous les modèles entraînés ont été stockés de manière centralisée avec leurs métadonnées
3. **Visualisation** : l'interface utilisateur de MLFlow nous a permis de comparer visuellement les différentes expérimentations

![photo mlflow UI avec adresse local 127.0.](images/xxxx.png)

Cette approche nous a permis de tracer l'évolution de nos modèles et de sélectionner le plus performant pour le déploiement.

## 💻 Interface utilisateur

### Architecture de l'application

Pour l'interfacage j'ai choisi FastAPI en Backend ( pourquoi ? voir dans la recherche d'info word ) :

![Page /docs du serveur FastAPI](images/ printscreen FastAPI.png)

- **Backend (FastAPI)** :
   - API REST exposant le modèle d'analyse de sentiments
   - Endpoints pour la prédiction individuelle et par lots
   - Système de feedback et de monitoring
   - Téléchargement automatique des artefacts du modèle depuis MLFlow



## 🔄 Pipeline de déploiement continu

Pour automatiser le déploiement de notre modèle, nous avons mis en place un **pipeline CI/CD (Intégration Continue / Déploiement Continu)** avec les composants suivants :

1. **Versionnement du code** : utilisation de Git pour le contrôle de version
2. **GitHub Actions** : automatisation des tests et du déploiement à chaque push sur la branche (analyse_sentiments)
3. **Déploiement sur Azure** : plateforme Cloud pour héberger notre API de prédiction de sentiments

### Tests unitaires automatisés

Pour garantir la fiabilité de notre solution, nous avons implémenté des **tests unitaires automatisés** couvrant les aspects critiques :

1. **Test du endpoint** : Vérifie que l'API répond correctement avec un code 200 et confirme que le statut retourné est "ok". Le modèle est chargé correctement.
2. **Test du endpoint de prédiction** : S'assure que l'API traite correctement les requêtes POST sur `/predict`, accepte un texte à analyser et renvoie un résultat contenant les champs "sentiment".

![photo test API et anacondapowershell et si besoin mettre le lien du realise ](images/xxxx.png)

### GitHub Actions 

Le déploiement est entièrement automatisé grâce à **GitHub Actions** :

1. **Déclenchement** : À chaque commit/push sur la branche(analyse_sentiments), GitHub Actions lance le workflow.
2. **Tests automatisés** : Le workflow exécute tous les tests unitaires.
3. **Déploiement conditionnel** : Uniquement si les tests réussissent, l'application est déployée automatiquement sur Azure .[Test API ](https://module-7-bgg7hvanhddthjh4.canadacentral-01.azurewebsites.net/docs)

#### Création du workflow GitHub Actions

Pour la création du workflow GitHub Actions, nous créons un fichier `.github/workflows/heroku-deploy.yml` à la racine dont voici le contenu :
'''Mettre le code .yml important !!!!
'''
#### Configuration des secrets GitHub

Le workflow **GitHub Actions** a besoin d'accéder aux **variables d'environnement**. Nous avons donc renseigner les "secrets" nécessaires. Dans notre dépôt GitHub, nous allons dans "Settings" > "Secrets and variables" > "Actions", puis nous cliquons sur "New repository secret". Nous ajoutons les secrets suivants:

![photo "New repository secret" dans Github](images/xxxx.png)


### Déploiement sur Azure

Pour le déploiement de notre solution, nous avons choisi [Azure](https://azure.microsoft.com/) pour plusieurs raisons :

1. **Plan gratuit** : conforme à la demande de limiter les coûts pour ce prototype
2. **Intégration avec GitHub** : facilite le déploiement continu avec GitHub Actions
3. **Scalabilité** : possibilité d'évoluer si le projet est approuvé pour la production
4. **Région Europe** : conformité avec les exigences de localisation des données

#### Configuration Azure

Notre application utilise les fichiers de configuration suivants pour Azure :

- **Procfile** : `gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000`
- **runtime.txt** : `python-3.10`
- **requirements.txt** : Liste de toutes les dépendances nécessaires

Les variables d'environnement sur Azure incluent :
- `MLFLOW_TRACKING_URI` : URI du serveur MLflow
- `RUN_ID` : Identifiant du run MLflow du modèle déployé
- `INSTRUMENTATION_KEY` : Clé pour Azure voir xxxxxxx.yml

### Exemple d'exécution et déploiement réussis

La capture d'écran suivante indique les **tests ont été passés avec succès** et que le déploiement est réussi sur **Azure**.

![Capture d'écran d'un run GitHub Actions](images/xxxx.png)

### Avantages de notre pipeline CI/CD

Notre pipeline de déploiement continu offre plusieurs avantages significatifs :

1. **Automatisation du deploiment** : chaque modification poussée sur GitHub déclenche automatiquement les étapes de test, de packaging, et de déploiement de l'API FastAPI contenant le modèle d'analyse de sentiments.
2. **Fiabilité grace aux tests automatisés** : Les tests unitaires garantissent la validité du code à chaque mise à jour.
3. **Traçabilité** : Chaque déploiement est associé a un commit Git précis pour faciliter le suivi et evolutions du modéle
4. **Feedback rapide pour les developpeurs** : en cas d'erreur des tests ou du deploiment, une notification est envoyé pour corrigier rapidement le bug


## 📡 Suivi de la performance en production

### Suivi des performances avec Azure Application Insights

Afin de surveiller le comportement de notre modèle en production, nous avons intégré Azure Application Insights, un outil puissant d’analyse des performances. Cette solution nous offre :

  **Une télémétrie automatisée** : collecte en temps réel des métriques de performance de l’API.

  **Des événements personnalisés** : enregistrement spécifique des actions ou erreurs liées aux prédictions du modèle.

  **Des tableaux de bord interactifs** : pour visualiser et analyser les performances sur la durée.

Cette intégration nous donne une vue complète et en temps réel du comportement de notre modèle.

### Collecte de feedback utilisateur

Dans le cadre de notre démarche MLOps, nous avons mis en place un système structuré de retour utilisateur permettant d’évaluer la justesse des prédictions :

**Interface de validation** : chaque utilisateur peut confirmer ou infirmer la prédiction générée par le modèle.

**Collecte détaillée** : enregistrement du tweet, de la prédiction du modèle et la correction utilisateur si nécessaire.

**Stockage unifié** : l’ensemble des retours est centralisé dans Azure Application Insights, facilitant l’analyse et l’amélioration continue du modèle.

Pour consulter les **feedbacks de tweets incorrectement prédits**, il suffit d'exécuter la commande suivante : 

```kusto
xxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxxx
```

![Feedbacks de tweets incorrectement prédits ](images/dans-Applicationinsight-capture-image.png)

Cette méthode permet de **constituer progressivement une base d'exemples difficiles à traiter**. Ces tweets mal classifiés sont très utiles car ils révèlent **les faiblesses spécifiques du modèle**. En les collectant systématiquement, on construit **un jeu de données ciblé sur les erreurs** du modèle. Cette méthode s'inscrit dans une démarche **d’apprentissage actif (active learning)**. Elle est plus **efficace** qu’un simple ajout aléatoire de données, car elle concentre l’amélioration du modèle sur les cas **réellement problématiques**.

### Configuration des alertes automatiques

Nous avons mis en place un **système d'alertes automatiques** pour détecter les dérives de performance du modèle.
Une alerte est déclenchée si **trois erreurs de prédiction sont signalées en moins de 5 minutes**.
Lorsqu’une alerte est générée, **une notification par email** est envoyée aux responsables du projet.
Toutes les **alertes sont stocker** pour permettre une analyse a posteriori.
Ce système de **monitoring proactif** permet à l’équipe d’intervenir avant que les erreurs ne se multiplient.

![Capture de l'écran alertes de Azure Application Insights](images/dans-Applicationinsight-Alerte-capture-image.png)

Pour améliorer le modéle,il faut **définir une periode** pour analyser les **tweets mal classifiés** pour détecter des motifs récurrents.
Les exemples identifiés sont ensuite ajoutés au dataset d'entraînement pour **enrichir le modèle** en se basant sur les conversation concernant la compagnie Air Paradis.
Enfin, **un réentraînement et déploiement automatisé** via le pipeline CI/CD.



## Conclusion

### Résultats obtenus
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxx
### Perspectives d'évolution

xxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxx

### Avantages de l'utilisation des outils IA pour Air Paradis
xxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxx
xxxxxxxxxxx