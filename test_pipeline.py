import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.graph import build_graph
from app.mock_llm import MockLLM
from app.pdf_utils import extract_pages

SAMPLE = Path(__file__).parent / "fixtures" / "sample_claim.pdf"

EXPECTED_CLASSES = {
    "claim_forms": [1],
    "cheque_or_bank_details": [2],
    "identity_document": [3],
    "discharge_summary": [4],
    "prescription": [5],
    "investigation_report": [6, 11, 12],
    "cash_receipt": [7],
    "itemized_bill": [9, 10],
    "other": [8, 13, 14, 15, 16, 17, 18],
}


class SpyLLM(MockLLM):
    """Records which page numbers each extraction call was shown."""

    def __init__(self):
        self.seen: dict[str, list[int]] = {}

    def structured(self, system, user, schema):
        import re

        self.seen.setdefault(schema.__name__, []).extend(
            int(n) for n in re.findall(r"=== PAGE (\d+) ===", user)
        )
        return super().structured(system, user, schema)


@pytest.fixture(scope="module")
def run():
    pages = extract_pages(SAMPLE.read_bytes())
    llm = SpyLLM()
    state = build_graph(llm).invoke({"claim_id": "CLM-1", "pages": pages, "warnings": []})
    return state["result"], llm


def test_page_count_and_classification(run):
    result, _ = run
    assert result["page_count"] == 18
    assert result["page_classification"] == EXPECTED_CLASSES


def test_agents_only_see_their_pages(run):
    _, llm = run
    assert sorted(llm.seen["IdentityInfo"]) == [1, 3]
    assert sorted(llm.seen["DischargeInfo"]) == [4]
    assert sorted(llm.seen["BillExtraction"]) == [9, 10]


def test_identity(run):
    data = run[0]["identity"]["data"]
    assert data["patient_name"] in ("John Michael Smith", "John Michael Smith".title())
    assert data["date_of_birth"] == "1985-03-15"
    assert data["policy_number"] == "POL-987654321"
    assert {"type": "government ID", "number": "ID-987-654-321"} in data["id_numbers"]


def test_discharge(run):
    data = run[0]["discharge_summary"]["data"]
    assert data["admission_date"] == "2025-01-20"
    assert data["discharge_date"] == "2025-01-25"
    assert "Pneumonia" in data["diagnosis"]
    assert data["attending_physician"] == "Dr. Sarah Johnson, MD"


def test_bills_arithmetic(run):
    bills = run[0]["itemized_bills"]["data"]
    hospital, pharmacy = bills["bills"]
    assert hospital["item_count"] == 20 and hospital["items_total"] == 6113.00
    assert hospital["calculated_total"] == 6418.65 and hospital["total_matches_stated"] is True
    assert pharmacy["item_count"] == 12 and pharmacy["items_total"] == 216.30
    assert pharmacy["calculated_total"] == 206.35 and pharmacy["total_matches_stated"] is True
    assert bills["total_items_amount"] == 6329.30
    assert bills["total_amount"] == 6625.00
    assert not hospital["issues"] and not pharmacy["issues"]


def test_api_end_to_end(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        with SAMPLE.open("rb") as f:
            r = client.post("/api/process", data={"claim_id": "CLM-2024-789456"},
                            files={"file": ("claim.pdf", f, "application/pdf")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["claim_id"] == "CLM-2024-789456"
        assert body["itemized_bills"]["data"]["total_amount"] == 6625.00


def test_api_rejects_non_pdf(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app.main import app

    with TestClient(app) as client:
        r = client.post("/api/process", data={"claim_id": "x"},
                        files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
        assert r.status_code == 400
        r = client.post("/api/process", data={"claim_id": "bad id!"},
                        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
        assert r.status_code == 422
