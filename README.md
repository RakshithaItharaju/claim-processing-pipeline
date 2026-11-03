Claim Processing Pipeline

About the Project

This project processes PDF claim documents using FastAPI and LangGraph. It extracts important information and returns the results in JSON format.

Technologies Used

- Python
- FastAPI
- LangGraph
- LLM
- PDF Processing

Working

1. Upload a PDF claim document.
2. Segregator Agent classifies the pages.
3. ID Agent extracts patient details.
4. Discharge Summary Agent extracts medical details.
5. Itemized Bill Agent extracts billing details.
6. Aggregator combines all results.
7. Final JSON output is generated.

Document Types

- Claim Forms
- Identity Documents
- Itemized Bills
- Discharge Summaries
- Prescriptions
- Investigation Reports
- Cash Receipts
- Bank Details
- Other

API Endpoint

POST "/api/process"

Input:

- Claim ID
- PDF File

Output:

- Extracted information in JSON format.

How to Run

Install dependencies:

pip install -r requirements.txt

Start the application:

uvicorn main:app --reload

Author

Rakshitha Itharaju 
