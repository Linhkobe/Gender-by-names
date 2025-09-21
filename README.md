# Gender by Name API 

## 1. About the project 
A FastAPI + TensorFlow web service to predict gender probabilities from first names (the neural network model was trained with french firt names dataset), packaged with Docker and deployed on Google Cloud Run. 

Here is the deployed link : https://gender-by-name-872658504688.europe-west1.run.app/ 

## 2. Features
- Predict gender (male/female from given names)
- Uses TensorFlow SavedModel + statistical priors for prediction
- RESTful API built with FastAPI
- Simple HTML front-end for quick testting
- Dockerized for portability
- Production deployment via Google Cloud Run

## 3. Project structure 
gender_by_name/
├── app/                  # FastAPI app
│   ├── main.py           # Entrypoint
│   ├── service.py        # Model + encoding logic
│   ├── templates/        # index.html (frontend)
│   └── static/           # CSS, favicon, etc.
├── model/
│   └── prod/
│       ├── exported_model/    # TensorFlow SavedModel
│       │   ├── saved_model.pb
│       │   └── variables/
│       ├── data_dpt_encode.csv
│       ├── prior_mean.json
│       └── prenom_encoder.pkl
├── requirements.txt
├── Dockerfile
└── README.md


## 4. Steps for local development 
1. Clone repository

```
git clone https://github.com/Linhkobe/Gender-by-names.git
cd gender_by_name
```

2. Install dependencies (Python 3.10+)

 ```
 python3 -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt
 ```

 3. Run API locally 

```
uvicorn app.main:app --reload --port 8000
```

4. Open http://localhost:8000 on browser to test

## 5. Docker

* Install Docker desktop as prerequisite

1. Build image

```
docker build -t gender-by-name:v1 .
```

2. Run container
```
docker run --rm -p 8000:8000 gender-by-name:v1
```

3. Test in browser with http://localhost:8000


# 6. Deployment to Google Cloud Run

* Create a project and get its ID via https://cloud.google.com/?hl=en as prerequisite 

1. Enable APIs

```
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

2. Create artifact registry repo

```
gcloud artifacts repositories create gender-repo \
  --repository-format=docker
  --location=europe-west1 \
  --description="Docker repo for gender-by-name"
```

3. Build & push image

```
gcloud auth configure-docker europe-west1-docker.pkg.dev
docker build -t europe-west1-docker.pkg.dev/<PROJECT_ID>/gender-repo/gender-by-name:v1 .
docker push europe-west1-docker.pkg.dev/<PROJECT_ID>/gender-repo/gender-by-name:v1
```

4. Deploy to Google Cloud Run

```
gcloud run deploy gender-by-name \
  --image europe-west1-docker.pkg.dev/<PROJECT_ID>/gender-repo/gender-by-name:v1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Go \
  --timeout 600 \
  --set-env-vars MODEL_DIR=/app/model/prod,VERSION=v1
```


