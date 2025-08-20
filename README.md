# titanic-webapp
A web application deployment using Kubernetes and dockers.

This project containerizes a multi-service Python web application, deploying it to a Kubernetes cluster. 

# Project Overview

The system consists of four main services:

UI: A web app ui (Flask) for interacting with the system.

Processing: Handles preprocessing and data pipeline tasks.

Training: Runs machine learning model training using cleaned data.

Inference: Generate predictions using trained model.

# Features

Containerized Python services with Docker.

Kubernetes Deployments for scalability and resilience.

Shared Persistent Volume for inter-service file exchange.

ConfigMaps for externalizing configuration.

Probes for health monitoring.

Ingress option for external access. (requires domain name)

# Project structure:
```bash
titanic-webapp/
├───ui/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── templates
│       ├── display.html
│       ├── index.html
│       └── map.html
├───processing/
│   ├── processing.py
│   ├── processing_api.py
│   ├── Dockerfile
│   └── requirements.txt
├───training/
│   ├── training.py
│   ├── training_api.py
│   ├── Dockerfile
│   └── requirements.txt
├───inference/
│   ├── inference.py
│   ├── inference_api.py
│   ├── Dockerfile
│   └── requirements.txt
├───k8s/
│   ├── ui.yaml
│   ├── processing.yaml
│   ├── training.yaml
│   ├── inference.yaml
│   ├── shared-pvc.yaml
│   ├── training-config.yaml
│   └── ingress.yaml (requires domain name)
├───docker-compose.yml
└───README.md
```

# Instructions:
## Docker build:
Build all the four main images
```
docker build -t aiad-ui:latest ./ui
docker build -t aiad-training:latest ./training
docker build -t aiad-processing:latest ./processing
docker build -t aiad-inference:latest ./inference
```

## Kubernetes deploy:
Build all the k8s config files
```
kubectl apply -f k8s/shared-pvc
kubectl apply -f k8s/ui
kubectl apply -f k8s/training
kubectl apply -f k8s/processing
kubectl apply -f k8s/inference
kubectl apply -f k8s/training-config
kubectl apply -f k8s/ingress (optional)
```

## Data File Copy:
If needed, copy over the data files into the PVC:
```
kubectl cp ./shared_volume processing-79547dfb8b-xlrhn:/usr/src/app
```

## Web App:
Get url link through:
```
minikube service ui --url
```

## Scaling:
Manual:
```
kubectl scale deployment inference --replicas=5
```

Auto:
```
kubectl autoscale deployment inference --cpu-percent=50 --min=2 --max=10
```

## Rollouts and Rollbacks:
Rollout example code:
```
kubectl set image deployment/inference inference=myrepo/inference:2.0
kubectl rollout status deployment/inference
kubectl rollout history deployment/inference
```

Rollback:
```
kubectl rollout undo deployment/inference
```

## Enable Ingress (requires domain name):
```
minikube addons enable ingress
```