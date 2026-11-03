"""Offline stand-in for the LLM (LLM_PROVIDER=mock).

Keyword/regex heuristics tuned to the sample claim packet. It exists so the whole graph
(routing, parallel agents, aggregation, arithmetic) can be run and tested with no API key.
It is NOT a substitute for the real model on unseen documents.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from .schemas import (
    BillDocument,
    BillExtraction,
    DischargeInfo,
    IdentityInfo,
    IDNumber,
    LineItem,
    PageLabel,
    SegregationResult,
)

T = TypeVar("T", bound=BaseModel)

_PAGE_SPLIT = re.compile(r"=== PAGE (\d+) ===\n")


def _split_pages(user: str) -> list[tuple[int, str]]:
    parts = _PAGE_SPLIT.split(user)
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]


def _iso(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip().rstrip(",")
    for fmt in ("%B %d, %Y", "%d-%b-%Y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _grab(pattern: str, text: str, flags: int = re.I | re.M) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def classify(text: str) -> tuple[str, str]:
    t = text.upper()[:300]  # headers only: body text can mention other document names
    rules = [
        ("DISCHARGE SUMMARY", "discharge_summary"),
        ("ITEMIZED HOSPITAL BILL", "itemized_bill"),
        ("PHARMACY & OUTPATIENT BILL", "itemized_bill"),
        ("MEDICAL CLAIM FORM", "claim_forms"),
        ("CHEQUE DETAILS", "cheque_or_bank_details"),
        ("BANK ACCOUNT DETAILS", "cheque_or_bank_details"),
        ("GOVERNMENT ID CARD", "identity_document"),
        ("CASH RECEIPT", "cash_receipt"),
        ("LABORATORY REPORT", "investigation_report"),
        ("COMPREHENSIVE METABOLIC PANEL", "investigation_report"),
        ("LIPID PANEL", "investigation_report"),
    ]
    for kw, label in rules:
        if kw in t:
            return label, f"contains '{kw.title()}'"
    if t.startswith("RX") and "PRESCRIPTION" in t:
        return "prescription", "Rx header with PRESCRIPTION section"
    return "other", "no known document header"


class MockLLM:
    name = "mock:heuristic"

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        if schema is SegregationResult:
            labels = []
            for page, text in _split_pages(user):
                doc_type, reason = classify(text)
                labels.append(PageLabel(page=page, doc_type=doc_type, reason=reason))
            return SegregationResult(labels=labels)  # type: ignore[return-value]
        if schema is IdentityInfo:
            return self._identity(user)  # type: ignore[return-value]
        if schema is DischargeInfo:
            return self._discharge(user)  # type: ignore[return-value]
        if schema is BillExtraction:
            return self._bills(user)  # type: ignore[return-value]
        raise NotImplementedError(schema)

    # ------------------------------------------------------------------ #
    def _identity(self, user: str) -> IdentityInfo:
        info = IdentityInfo()
        ids: list[IDNumber] = []
        for _, text in _split_pages(user):
            if "GOVERNMENT ID" in text.upper():
                info.patient_name = info.patient_name or (_grab(r"Full Name:\s*(.+)", text) or "").title() or None
                info.date_of_birth = _iso(_grab(r"Date of Birth:\s*(\S+)", text)) or info.date_of_birth
                info.gender = _grab(r"Gender:\s*(\w+)", text) or info.gender
                gid = _grab(r"ID Number:\s*(\S+)", text)
                if gid:
                    ids.append(IDNumber(type="government ID", number=gid))
                addr = _grab(r"Address:\s*(.+\n.+)", text)
                info.address = addr.replace("\n", ", ") if addr else info.address
            else:  # claim form
                info.patient_name = _grab(r"Patient Name:\s*(.+)", text) or info.patient_name
                info.date_of_birth = info.date_of_birth or _iso(_grab(r"Date of Birth:\s*(.+)", text))
                info.policy_number = _grab(r"Policy Number:\s*(\S+)", text) or info.policy_number
                info.claim_reference = _grab(r"Claim Reference:\s*(\S+)", text)
                info.phone = _grab(r"Contact Number:\s*(\S+)", text)
                info.email = _grab(r"Email:\s*(\S+)", text)
                info.insurer = _grab(r"^(.*Insurance Company)\s*$", text)
                if info.policy_number:
                    ids.append(IDNumber(type="policy number", number=info.policy_number))
                if info.claim_reference:
                    ids.append(IDNumber(type="claim reference", number=info.claim_reference))
        info.id_numbers = ids
        return info

    def _discharge(self, user: str) -> DischargeInfo:
        text = user
        meds_block = _grab(r"Discharge Medications:\s*(.+?)\nFollow-up:", text, re.I | re.S) or ""
        meds = [m.strip() for m in meds_block.splitlines() if m.strip()]
        return DischargeInfo(
            patient_name=_grab(r"^Name:\s*(.+)", text),
            hospital_name=_grab(r"^(CITY MEDICAL CENTER)\s*$", text),
            admission_date=_iso(_grab(r"Admission Date:\s*(.+)", text)),
            discharge_date=_iso(_grab(r"Discharge Date:\s*(.+)", text)),
            diagnosis=_grab(r"Admission Diagnosis:\s*(.+)", text),
            attending_physician=_grab(r"Attending Physician:\s*(.+)", text),
            condition_at_discharge=_grab(r"Condition at Discharge:\s*(.+)", text),
            discharge_medications=meds,
            follow_up=_grab(r"Follow-up:\s*(.+)", text),
        )

    def _bills(self, user: str) -> BillExtraction:
        money_re = re.compile(r"^\$[\d,]+\.\d\d$")
        date_re = re.compile(r"^\d\d/\d\d/\d\d$")
        bills = []
        for _, text in _split_pages(user):
            # pypdf puts every table cell on its own line: ... [date] desc [dosage] qty $rate $amount
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            items = []
            for i in range(2, len(lines) - 2):
                if not (lines[i].isdigit() and money_re.match(lines[i + 1]) and money_re.match(lines[i + 2])):
                    continue
                if date_re.match(lines[i - 2]):
                    date, desc = _iso(lines[i - 2]), lines[i - 1]
                else:  # pharmacy rows: name, dosage (or '-'), qty, price, total
                    date, desc = None, lines[i - 2]
                items.append(
                    LineItem(date=date, description=desc, quantity=float(lines[i]),
                             unit_price=_num(lines[i + 1][1:]), amount=_num(lines[i + 2][1:]))
                )

            def money(p: str) -> float | None:
                v = _grab(p, text)
                return _num(v) if v else None

            bills.append(
                BillDocument(
                    bill_number=_grab(r"(?:Bill|Invoice) Number:\s*(\S+)", text),
                    provider="City Medical Center",
                    bill_date=_iso(_grab(r"(?:Bill )?Date:\s*(.+)", text)),
                    patient_name=_grab(r"Patient(?: Name)?:\s*([A-Za-z ]+?)(?:\s*\(|$)", text),
                    items=items,
                    stated_subtotal=money(r"Subtotal:\s*\$([\d,]+\.\d\d)"),
                    discount=money(r"Discount \(\d+%\):\s*-\$([\d,]+\.\d\d)"),
                    tax=money(r"Tax \(\d+%\):\s*\$([\d,]+\.\d\d)"),
                    stated_total=money(r"(?:Total Amount|TOTAL DUE):\s*\$([\d,]+\.\d\d)"),
                )
            )
        return BillExtraction(bills=bills)
