from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,computed_field
from typing import Annotated
import joblib
import pandas as pd
import mlflow
import mlflow.pyfunc
from contextlib import asynccontextmanager

class UserInput(BaseModel):
    time_spend_alone : Annotated[int,Field(...,ge=0,le=11,description='Time spend alone by the user (0–11)')]
    stage_fear : Annotated[bool,Field(...,description='Do the user have stage fear on not')]
    social_event_attendance : Annotated[int,Field(...,ge=0,le=10,description='Frequency of social events (0–10)')]
    going_outside : Annotated[int,Field(...,ge=0,le=7,description='Frequency of going outside (0–7)')]
    drained_after_socializing : Annotated[bool,Field(...,description='Feeling drained after socializing')]
    friends_circle_size : Annotated[int,Field(...,ge=0,le=15,description='Number of close friends (0–15)')]
    post_frequency: Annotated[int,Field(...,ge=0,le=10,description='Social media post frequency (0–10)')] 
    
mlflow.set_tracking_uri("https://dagshub.com/AkHiLdEvGoD/Personality-Prediction.mlflow")
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(model_name,stages=['Staging'])
    if not latest:
        latest = client.get_latest_versions(model_name,stages=['None'])
    return latest[0].version if latest else None

@asynccontextmanager
async def lifespan(app : FastAPI):
    model_name = 'my_model'
    model_version = get_latest_model_version(model_name)
    if not model_version:
        raise RuntimeError(f"No versions found for model: {model_name}")
    model_uri = f'models:/{model_name}/{model_version}'
    print(f'Loading model from {model_uri}')

    app.state.model = mlflow.pyfunc.load_model(model_uri)
    
    app.state.label_encoder = joblib.load('./local_S3/models/label_encoder.pkl')

    app.state.preprocessing_pipeline = joblib.load('./local_S3/models/preprocessing_pipeline.pkl')

    print('Model and encoders loaded')
    yield


app = FastAPI(title="Personality Prediction API",lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "up"}

@app.post('/predict')
async def predict(data:UserInput):
    try : 
        model = app.state.model
        preprocessing_pipeline = app.state.preprocessing_pipeline
        label_encoder = app.state.label_encoder

        offline_social_activity = data.social_event_attendance * data.going_outside
        df = pd.DataFrame({
            'Time_spent_Alone' : [data.time_spend_alone],
            'Stage_fear' : [data.stage_fear],
            'Social_event_attendance' : [data.social_event_attendance],
            'Going_outside':[data.going_outside],
            'Drained_after_socializing':[data.drained_after_socializing],
            'Friends_circle_size' : [data.friends_circle_size],
            'Post_frequency':[data.post_frequency],
            'Offline_social_activity' : [offline_social_activity]
        })

        processed_df = preprocessing_pipeline.transform(df)
        pred = model.predict(processed_df)

        decoded_pred = label_encoder.inverse_transform(pred)

        return JSONResponse(status_code=200, content={'Predicted Personality': decoded_pred[0]})

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Prediction error: {e}")

