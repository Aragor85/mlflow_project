 ```mermaid
classDiagram
    %% Classes principales
    class 1-DataLoader {
      +load_sentiment140()
      
    }
    class 2-ModelTrainer {
      +preprocess_tweet()
      +Modéle_classic()
      +train_use()
      +train_lstm()
      +train_bert()
    }
    class 3-MLflowTracker {
      +start_run()
      +log_params()
      +log_metrics()
      +log_model()
    }
    class 8-FastAPIApp {
      +/predict()
      +/feedback()
    }
    class 7-StreamlitApp {
      +display_form()
      +call_API()
      +collect_feedback()
    }
    class 6-AzureAppService {
      +pull_container()
      +start_container()
    }
    class 5-ACR {
      +store_image()
      +provide_image()
    }
    class 4-GitHubActions {
      +on_push_build()
      +docker_login_ACR()
      +build_and_push_image()
      +deploy_to_Azure()
    }
    class 9-AppInsights {
      +track_request()
      +track_trace()
      +query_logs()
    }

    %% Relations
    1-DataLoader --> 2-ModelTrainer : fournit données prétraitées
    2-ModelTrainer --> 3-MLflowTracker : enregistre runs
    8-FastAPIApp --> 2-ModelTrainer : charge modèle depuis artefact
    7-StreamlitApp --> 8-FastAPIApp : appelle /predict et /feedback
    8-FastAPIApp --> 9-AppInsights : enregistre traces
    4-GitHubActions --> 5-ACR : push image
    5-ACR --> 6-AzureAppService : fournit image
    6-AzureAppService --> 8-FastAPIApp : héberge API
    6-AzureAppService --> 7-StreamlitApp : héberge UI
    8-FastAPIApp <-- 7-StreamlitApp : intègre dans le même conteneur
```
 