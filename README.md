# AI Data Analyst Agent

An agentic AI tool that takes any CSV dataset and automatically performs Exploratory Data Analysis (EDA), uncovers insights, and delivers business recommendations — powered by Google Gemini.

---

## What It Does

Upload any CSV → AI agent analyzes it end-to-end and returns:
- Dataset summary & data quality report
- 5+ key insights with actual numbers
- Column correlations & relationships
- Business recommendations
- ML model suggestions for the dataset

No manual EDA needed. Just upload and get insights.

---

## Why This Is Different

Most data analysis tools require you to write code or pick chart types manually. This agent reasons about your data like a senior analyst — it understands context, spots patterns, and tells you what matters.

---

## Tech Stack

- **AI:** Google Gemini 1.5 Flash (LLM)
- **Backend:** FastAPI + Uvicorn
- **Data Processing:** Pandas, NumPy
- **API:** REST endpoints with file upload support

---

## Project Structure

```
ai-data-analyst/
│
├── ai_analyst_agent.py     # Core agent logic + CLI demo
├── app.py                  # FastAPI REST API
├── requirements.txt        # Dependencies
└── README.md
```

---

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Gemini API Key
In `app.py` and `ai_analyst_agent.py`, replace:
```python
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```
with your actual key from [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Run CLI Demo
```bash
python ai_analyst_agent.py
# or with your own CSV:
python ai_analyst_agent.py your_data.csv
```

### 4. Start the API
```bash
uvicorn app:app --reload
```
API docs at: `http://localhost:8000/docs`

---

## API Endpoints

### `POST /analyze`
Upload any CSV → get full AI analysis.

**Request:** multipart/form-data with CSV file

**Response:**
```json
{
  "filename": "sales_data.csv",
  "basic_stats": {
    "rows": 1000,
    "columns": 8,
    "missing_values": {...}
  },
  "ai_analysis": "## Dataset Summary\n This dataset contains..."
}
```

### `POST /ask`
Upload CSV + ask a specific question about your data.

**Query param:** `?question=Which department has highest attrition?`

**Response:**
```json
{
  "question": "Which department has highest attrition?",
  "answer": "Based on the data, the IT department...",
  "data_shape": {"rows": 1000, "columns": 8}
}
```

### `GET /health`
```json
{"status": "healthy", "model": "Gemini 1.5 Flash", "version": "1.0.0"}
```

---

## Example Output

Input: Employee HR dataset (1000 rows, 10 columns)

```
DATASET SUMMARY
This HR dataset tracks employee demographics, performance, and attrition...

KEY INSIGHTS
1. Attrition rate is 16.1% — highest in Sales (24%) and HR (19%)
2. Employees with <2 years experience are 3x more likely to leave
3. Average salary gap between departments: $18,000
4. Performance scores negatively correlate with overtime hours (-0.42)
5. Remote workers show 31% lower attrition than office workers

RECOMMENDATIONS
1. Introduce retention bonuses for employees in years 1-2
2. Review overtime policies in high-attrition departments
3. Expand remote work options to reduce overall attrition
```

---

## Author

**Prathamesh Jadhav**  
[LinkedIn](https://linkedin.com/in/prathameshjadhav) | [GitHub](https://github.com/Prathamesh1230)
