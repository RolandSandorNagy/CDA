from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

class ReportRequest(BaseModel):
    asset_id: str

@app.post("/generate_report")
def generate_report(request: ReportRequest):
    asset_id = request.asset_id
    manual_description = request.manual_description
    try:
        result = subprocess.run(
            ["python", "app_wrapper.py", asset_id, manual_description],
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "status": "success",
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")