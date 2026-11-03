"""Pydantic schemas shared by the segregator and the extraction agents."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal[
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

DOC_TYPES: list[str] = list(DocType.__args__)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Segregator
# --------------------------------------------------------------------------- #
class PageLabel(BaseModel):
    page: int = Field(description="1-based page number exactly as shown in the input")
    doc_type: DocType
    reason: str = Field(description="One short sentence explaining the choice")


class SegregationResult(BaseModel):
    labels: list[PageLabel]


# --------------------------------------------------------------------------- #
# ID agent
# --------------------------------------------------------------------------- #
class IDNumber(BaseModel):
    type: str = Field(description="e.g. government ID, policy number, MRN, patient ID")
    number: str


class IdentityInfo(BaseModel):
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    gender: Optional[str] = None
    id_numbers: list[IDNumber] = Field(default_factory=list)
    policy_number: Optional[str] = None
    insurer: Optional[str] = None
    claim_reference: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None


# --------------------------------------------------------------------------- #
# Discharge summary agent
# --------------------------------------------------------------------------- #
class DischargeInfo(BaseModel):
    patient_name: Optional[str] = None
    hospital_name: Optional[str] = None
    admission_date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    discharge_date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    diagnosis: Optional[str] = Field(None, description="Primary / admission diagnosis")
    other_diagnoses: list[str] = Field(default_factory=list)
    attending_physician: Optional[str] = None
    other_physicians: list[str] = Field(default_factory=list)
    condition_at_discharge: Optional[str] = None
    discharge_medications: list[str] = Field(default_factory=list)
    follow_up: Optional[str] = None


# --------------------------------------------------------------------------- #
# Itemized bill agent
# --------------------------------------------------------------------------- #
class LineItem(BaseModel):
    date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD if shown")
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: float = Field(description="Line total exactly as printed on the bill")


class BillDocument(BaseModel):
    bill_number: Optional[str] = None
    provider: Optional[str] = None
    bill_date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    patient_name: Optional[str] = None
    items: list[LineItem] = Field(default_factory=list)
    stated_subtotal: Optional[float] = None
    discount: Optional[float] = Field(None, description="Positive number if a discount is applied")
    tax: Optional[float] = None
    stated_total: Optional[float] = Field(None, description="Total as printed, before insurance split")


class BillExtraction(BaseModel):
    bills: list[BillDocument]
