from google import genai
import pandas as pd
import json
import os
import sys

# ── Configure Gemini ──────────────────────────────────────────────
GEMINI_API_KEY = "Your-API-Key"  # Replace with your key
client = genai.Client(api_key=GEMINI_API_KEY)

# ── Core Analysis Functions ───────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    df = pd.read_csv(filepath)
    return df

def get_basic_stats(df: pd.DataFrame) -> dict:
    """Extract basic statistics from the dataframe."""
    stats = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "numeric_summary": {}
    }
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        stats["numeric_summary"] = desc.to_dict()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    stats["categorical_columns"] = cat_cols
    stats["categorical_unique_counts"] = {
        col: int(df[col].nunique()) for col in cat_cols
    }
    return stats

def ask_gemini(prompt: str) -> str:
    """Send prompt to Gemini and return response."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

def analyze_dataset(df: pd.DataFrame, user_question: str = None) -> dict:
    """Full AI-powered analysis of a dataset."""
    stats = get_basic_stats(df)
    sample_data = df.head(5).to_string()

    prompt = f"""
You are an expert Data Analyst AI Agent. Analyze the following dataset and provide deep, actionable insights.

DATASET OVERVIEW:
- Total Rows: {stats['rows']}
- Total Columns: {stats['columns']}
- Column Names: {stats['column_names']}
- Data Types: {stats['dtypes']}
- Missing Values: {stats['missing_values']}

NUMERIC SUMMARY:
{json.dumps(stats['numeric_summary'], indent=2)}

SAMPLE DATA (first 5 rows):
{sample_data}

{"USER QUESTION: " + user_question if user_question else ""}

Please provide a comprehensive analysis with the following sections:

1. DATASET SUMMARY
2. DATA QUALITY ISSUES
3. KEY INSIGHTS (at least 5 specific insights with numbers)
4. CORRELATIONS & RELATIONSHIPS
5. BUSINESS RECOMMENDATIONS (at least 3)
6. SUGGESTED NEXT STEPS & ML OPPORTUNITIES

Be specific, use actual numbers from the data, and make insights actionable.
"""
    ai_response = ask_gemini(prompt)
    return {"basic_stats": stats, "ai_analysis": ai_response}


# ── Main Demo ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("No CSV provided — using sample dataset for demo...\n")
        data = {
            "Age": [25, 32, 47, 51, 23, 38, 44, 29, 55, 31],
            "Salary": [35000, 52000, 78000, 95000, 28000, 61000, 83000, 41000, 105000, 49000],
            "Department": ["IT", "HR", "IT", "Finance", "IT", "HR", "Finance", "IT", "Finance", "HR"],
            "Experience_Years": [1, 5, 12, 18, 0, 8, 15, 3, 22, 6],
            "Performance_Score": [3.2, 4.1, 4.5, 3.8, 2.9, 4.3, 4.7, 3.5, 4.9, 4.0],
            "Left_Company": [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
        }
        df = pd.DataFrame(data)
        filepath = "sample_data.csv"
        df.to_csv(filepath, index=False)

    df = load_csv(filepath)

    print("=" * 60)
    print("       AI DATA ANALYST AGENT — ANALYSIS REPORT")
    print("=" * 60)
    print(f"File: {filepath}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print("=" * 60)
    print("\n Running AI analysis... (this may take 10-15 seconds)\n")

    result = analyze_dataset(df)

    print("AI ANALYSIS:")
    print("-" * 60)
    print(result["ai_analysis"])
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
