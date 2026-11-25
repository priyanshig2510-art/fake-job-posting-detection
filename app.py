
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

model = joblib.load(BASE_DIR / "model.pkl")


class Job(BaseModel):
    title: str
    company_profile: str = " "
    description: str = " "
    requirements: str = " "

    employment_type: str = "Unknown"
    required_experience: str = "Unknown"
    required_education: str = "Unknown"

    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/predict")
def predict(job: Job):
    try:
        text = (
            f"{job.title} {job.company_profile} "
            f"{job.description} {job.requirements}"
        )

        row = pd.DataFrame([{
            "text": text,
            "employment_type": job.employment_type,
            "required_experience": job.required_experience,
            "required_education": job.required_education,
            "telecommuting": job.telecommuting,
            "has_company_logo": job.has_company_logo,
            "has_questions": job.has_questions
        }])

        prob = float(model.predict_proba(row)[0][1])
        prediction = int(prob > 0.5)

        return {
            "prediction": prediction,
            "probability": round(prob, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
