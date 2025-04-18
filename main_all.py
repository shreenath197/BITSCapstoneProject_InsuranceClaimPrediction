from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
from io import StringIO
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np

app = FastAPI(
    title="Insurance Claim Amount Prediction Model",
    description="API providing Insurance Claim Predictions from LinearRegression, DeepLearning, and XGBoost models"
)

@app.on_event("startup")
def load_models():
    global model_lr, model_dl, model_xgb
    model_lr = joblib.load("claim_pred_Linear_Regression.joblib")
    model_dl = joblib.load("claim_pred_Deep_Learning.joblib")
    model_xgb = joblib.load("claim_pred_XGBOOST.joblib")

@app.get("/")
def index():
    return {
        "message": "Upload a CSV to get predictions and accuracy from LinearRegression, DeepLearning, and XGBoost models."
    }

@app.post("/predict/csv/")
async def predict_with_all_models(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        s = str(contents, 'utf-8')
        df = pd.read_csv(StringIO(s))

        # Check if target column is present
        if 'Claim_amount' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Claim_amount' column for accuracy calculation.")

        X = df.drop(columns=['Claim_amount'])
        y_true = df['Claim_amount']

        predictions = {}
        models = {
            "LinearRegression": model_lr,
            "DeepLearning": model_dl,
            "XGBoost": model_xgb,
        }

        for name, model in models.items():
            y_pred = model.predict(X)

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            accuracy = 100 - (mape * 100)

            predictions[name] = {
                "Predictions": y_pred.tolist(),
                "MSE": mse,
                "RMSE": rmse,
                "R2_Score": r2,
                "MAPE (%)": mape * 100,
                "Accuracy (%) (approx)": accuracy
            }

        return {"results": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
