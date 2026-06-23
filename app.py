from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import pandas as pd
import json
import io
import os

# ── Configure Gemini ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-API-Key")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(
    title="AI Data Analyst Agent API",
    description="Upload any CSV and get instant AI-powered analysis, insights, and recommendations.",
    version="1.0.0"
)

# ── Request Schema ────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    csv_data: str  # CSV content as string

# ── Helper Functions ──────────────────────────────────────────────
def get_basic_stats(df: pd.DataFrame) -> dict:
    stats = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        stats["numeric_summary"] = df[numeric_cols].describe().round(2).to_dict()
    return stats

def build_analysis_prompt(df: pd.DataFrame, question: str = None) -> str:
    stats = get_basic_stats(df)
    sample = df.head(5).to_string()

    prompt = f"""
You are an expert AI Data Analyst Agent. Analyze the dataset below and provide deep, actionable insights.

DATASET INFO:
- Rows: {stats['rows']}, Columns: {stats['columns']}
- Columns: {stats['column_names']}
- Data Types: {stats['dtypes']}
- Missing Values: {stats['missing_values']}

NUMERIC SUMMARY:
{json.dumps(stats.get('numeric_summary', {}), indent=2)}

SAMPLE DATA:
{sample}

{"USER QUESTION: " + question if question else ""}

Provide analysis in these sections:
1. DATASET SUMMARY - what this data is about
2. DATA QUALITY - missing values, issues
3. KEY INSIGHTS - at least 5 specific insights with numbers
4. CORRELATIONS - relationships between columns
5. RECOMMENDATIONS - at least 3 actionable business recommendations
6. ML OPPORTUNITIES - what models could be trained on this data

Be specific, use actual numbers, and make insights actionable.
"""
    return prompt

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AI Data Analyst Agent is running!",
        "endpoints": {
            "POST /analyze": "Upload a CSV file for full AI analysis",
            "POST /ask": "Ask a specific question about your CSV data",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Gemini 1.5 Flash", "version": "1.0.0"}

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Upload a CSV file and get full AI-powered analysis."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    stats = get_basic_stats(df)
    prompt = build_analysis_prompt(df)

    try:
        response = model.generate_content(prompt)
        ai_analysis = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    return JSONResponse(content={
        "filename": file.filename,
        "basic_stats": stats,
        "ai_analysis": ai_analysis
    })

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = "What are the key insights?"):
    """Upload CSV and ask a specific question about your data."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")

    prompt = build_analysis_prompt(df, question)

    try:
        response = model.generate_content(prompt)
        ai_answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    return JSONResponse(content={
        "question": question,
        "answer": ai_answer,
        "data_shape": {"rows": df.shape[0], "columns": df.shape[1]}
    })
