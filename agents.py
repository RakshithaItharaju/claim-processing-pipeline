"""Segregator + extraction agents (the LangGraph node functions)."""
from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal

from .llm import LLM
from .schemas import (
    DOC_TYPES,
    BillExtraction,
    DischargeInfo,
    IdentityInfo,
    SegregationResult,
)

log = logging.getLogger(__name__)

# Which document types each extraction agent is allowed to see.
# claim_forms go to the ID agent because the policy number / claim reference
# that the ID agent must extract live on the claim form.
ROUTES: dict[str, list[str]] = {
    "id_agent": ["identity_document", "claim_forms"],
    "discharge_agent": ["discharge_summary"],
    "bill_agent": ["itemized_bill"],
}

SEG_BATCH_SIZE = 12          # pages per classification call
SEG_CHARS_PER_PAGE = 1500    # enough to recognise a page, keeps the call cheap
EXTRACT_CHARS_PER_PAGE = 12000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _page_block(page: dict, limit: int) -> str:
    text = page["text"][:limit] if page["text"] else "[NO EXTRACTABLE TEXT]"
    return f"=== PAGE {page['page']} ===\n{text}"


def _money(x: float | None) -> Decimal:
    return Decimal(str(round(x, 2))) if x is not None else Decimal("0")


def _f(d: Decimal) -> float:
    return float(d.quantize(Decimal("0.01")))


# --------------------------------------------------------------------------- #
# 1. Segregator (AI)
# --------------------------------------------------------------------------- #
SEG_SYSTEM = f"""You classify pages of a scanned health-insurance claim packet.
Label EVERY page with exactly one of these document types:
{", ".join(DOC_TYPES)}

Guidance:
- claim_forms: the insurer's claim form / claim application.
- cheque_or_bank_details: cheque, bank account or payment details.
- identity_document: government ID, passport, driving licence, insurance card.
- itemized_bill: any bill or invoice listing charges line by line (hospital or pharmacy).
- discharge_summary: hospital discharge summary.
- prescription: doctor's prescription (Rx).
- investigation_report: lab, pathology, radiology or other test reports.
- cash_receipt: receipt for a payment made.
- other: anything else (registration forms, consent forms, letters, questionnaires, referrals, ...).
A page may continue the document on the previous page (e.g. a letter that runs onto a second page);
give it the same type as the page it continues.
Return one label per page, using the page numbers exactly as given."""


def segregate(pages: list[dict], llm: LLM) -> tuple[dict[int, str], list[str]]:
    """Classify every page. Returns ({page: doc_type}, warnings)."""
    labels: dict[int, str] = {}
    warnings: list[str] = []
    for start in range(0, len(pages), SEG_BATCH_SIZE):
        batch = pages[start : start + SEG_BATCH_SIZE]
        user = "\n\n".join(_page_block(p, SEG_CHARS_PER_PAGE) for p in batch)
        result = llm.structured(SEG_SYSTEM, user, SegregationResult)
        for lab in result.labels:
            labels[lab.page] = lab.doc_type

    for p in pages:
        if p["page"] not in labels:
            labels[p["page"]] = "other"
            warnings.append(f"page {p['page']}: not classified by the model, defaulted to 'other'")
        if not p["text"]:
            warnings.append(f"page {p['page']}: no extractable text (scanned image?); OCR is not enabled")
    return labels, warnings


def segregator_node(state: dict, llm: LLM) -> dict:
    pages = state["pages"]
    labels, warnings = segregate(pages, llm)

    by_page = {p["page"]: p for p in pages}
    routed: dict[str, list[dict]] = {agent: [] for agent in ROUTES}
    for page_no in sorted(labels):
        for agent, wanted in ROUTES.items():
            if labels[page_no] in wanted:
                routed[agent].append(by_page[page_no])
    return {"classifications": labels, "routed": routed, "warnings": warnings}


# --------------------------------------------------------------------------- #
# 2a. ID agent
# --------------------------------------------------------------------------- #
ID_SYSTEM = """Extract identity and policy information about the PATIENT from the pages given.
Pages may include a government ID and a claim form; merge them into one record.
Dates must be ISO (YYYY-MM-DD). Put every identifier you see (government ID number, policy number,
claim reference, patient ID, MRN) in id_numbers with a descriptive type. Use null when a field is
not present; never guess."""


def id_agent_node(state: dict, llm: LLM) -> dict:
    pages = state["routed"]["id_agent"]
    if not pages:
        return {"id_result": {"status": "no_pages", "data": None, "source_pages": []}}
    user = "\n\n".join(_page_block(p, EXTRACT_CHARS_PER_PAGE) for p in pages)
    data = llm.structured(ID_SYSTEM, user, IdentityInfo)
    return {
        "id_result": {
            "status": "ok",
            "source_pages": [p["page"] for p in pages],
            "data": data.model_dump(),
        }
    }


# --------------------------------------------------------------------------- #
# 2b. Discharge summary agent
# --------------------------------------------------------------------------- #
DISCHARGE_SYSTEM = """Extract data from the hospital discharge summary pages given.
Dates must be ISO (YYYY-MM-DD). diagnosis is the primary/admission diagnosis.
List physicians exactly as named (e.g. 'Dr. Sarah Johnson, MD'). Use null / empty list when absent;
never guess."""


def discharge_agent_node(state: dict, llm: LLM) -> dict:
    pages = state["routed"]["discharge_agent"]
    if not pages:
        return {"discharge_result": {"status": "no_pages", "data": None, "source_pages": []}}
    user = "\n\n".join(_page_block(p, EXTRACT_CHARS_PER_PAGE) for p in pages)
    data = llm.structured(DISCHARGE_SYSTEM, user, DischargeInfo)
    return {
        "discharge_result": {
            "status": "ok",
            "source_pages": [p["page"] for p in pages],
            "data": data.model_dump(),
        }
    }


# --------------------------------------------------------------------------- #
# 2c. Itemized bill agent
# --------------------------------------------------------------------------- #
BILL_SYSTEM = """Extract every bill in the pages given. A bill may span several pages and one page
may be a separate bill; return one entry per bill.
For each bill list EVERY line item (including service / delivery / fee lines) with the line amount
exactly as printed. Do not include subtotal, tax, discount, total or insurance-split rows as items.
Capture subtotal, discount (positive number), tax and the total printed before any insurance split.
Dates must be ISO (YYYY-MM-DD). Do not do arithmetic yourself - copy numbers as printed."""


def finalize_bills(extraction: BillExtraction) -> dict:
    """Do the arithmetic in code, not in the LLM."""
    bills_out = []
    all_items = Decimal("0")
    grand_total = Decimal("0")
    for b in extraction.bills:
        items_sum = sum((_money(i.amount) for i in b.items), Decimal("0"))
        discount = _money(b.discount)
        tax = _money(b.tax)
        calculated_total = items_sum - discount + tax

        issues: list[str] = []
        for i in b.items:
            if i.quantity is not None and i.unit_price is not None:
                expected = _money(i.quantity) * _money(i.unit_price)
                if abs(expected - _money(i.amount)) > Decimal("0.01"):
                    issues.append(
                        f"line '{i.description}': qty x rate = {_f(expected)} but amount is {_f(_money(i.amount))}"
                    )
        if b.stated_subtotal is not None and abs(_money(b.stated_subtotal) - items_sum) > Decimal("0.01"):
            issues.append(f"stated subtotal {b.stated_subtotal} differs from sum of items {_f(items_sum)}")
        if b.stated_total is not None and abs(_money(b.stated_total) - calculated_total) > Decimal("0.01"):
            issues.append(f"stated total {b.stated_total} differs from calculated total {_f(calculated_total)}")

        out = b.model_dump()
        out.update(
            {
                "item_count": len(b.items),
                "items_total": _f(items_sum),
                "calculated_total": _f(calculated_total),
                "total_matches_stated": (
                    None
                    if b.stated_total is None
                    else abs(_money(b.stated_total) - calculated_total) <= Decimal("0.01")
                ),
                "issues": issues,
            }
        )
        bills_out.append(out)
        all_items += items_sum
        grand_total += calculated_total

    return {
        "bills": bills_out,
        "total_items_amount": _f(all_items),
        "total_amount": _f(grand_total),
    }


def bill_agent_node(state: dict, llm: LLM) -> dict:
    pages = state["routed"]["bill_agent"]
    if not pages:
        return {"bill_result": {"status": "no_pages", "data": None, "source_pages": []}}
    user = "\n\n".join(_page_block(p, EXTRACT_CHARS_PER_PAGE) for p in pages)
    extraction = llm.structured(BILL_SYSTEM, user, BillExtraction)
    return {
        "bill_result": {
            "status": "ok",
            "source_pages": [p["page"] for p in pages],
            "data": finalize_bills(extraction),
        }
    }


# --------------------------------------------------------------------------- #
# 3. Aggregator
# --------------------------------------------------------------------------- #
def aggregator_node(state: dict, llm_name: str = "") -> dict:
    by_type: dict[str, list[int]] = defaultdict(list)
    for page, doc_type in sorted(state["classifications"].items()):
        by_type[doc_type].append(page)

    result = {
        "claim_id": state["claim_id"],
        "page_count": len(state["pages"]),
        "page_classification": {t: by_type.get(t, []) for t in DOC_TYPES},
        "identity": state["id_result"],
        "discharge_summary": state["discharge_result"],
        "itemized_bills": state["bill_result"],
        "warnings": state.get("warnings", []),
        "model": llm_name,
    }
    return {"result": result}
