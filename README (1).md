# Claim Processing Pipeline

FastAPI service that takes a claim PDF, uses an LLM to classify every page into one of 9 document
types, routes **only the relevant pages** to three extraction agents, and returns one aggregated JSON.
Orchestrated with LangGraph.

```
                        ┌─► id_agent ────────────┐
START ─► segregator ────┼─► discharge_agent ─────┼─► aggregator ─► END
         (LLM, per page)└─► bill_agent ──────────┘
                         (run in parallel; aggregator waits for all three)
```

## API

`POST /api/process` — multipart form: `claim_id` (string), `file` (PDF). Returns JSON.
`GET /health` — reports `ok`, or `degraded` with the reason if the LLM isn't configured.

```bash
curl -X POST http://localhost:8000/api/process \
  -F claim_id=CLM-2024-789456 \
  -F "file=@tests/fixtures/sample_claim.pdf;type=application/pdf"
```

`sample_output.json` is the response for the included 18-page sample packet.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set LLM_PROVIDER and the matching key
set -a; source .env; set +a
uvicorn app.main:app --reload
```

| `LLM_PROVIDER` | Key variable | Default model (override with `LLM_MODEL`) |
|---|---|---|
| `gemini` (default, has a free tier) | `GOOGLE_API_KEY` | `gemini-2.5-flash` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `mock` | none | offline heuristics, for demos and tests only |

Docker: `docker build -t claims . && docker run -p 8000:8000 --env-file .env claims`

Tests (use the mock provider, no key needed): `pytest -q`

## How it works

**Shared state** (`app/graph.py`): `claim_id`, `pages` (text of every page), `classifications`
(page → type), `routed` (agent → its pages only), one result key per agent, and a `warnings` list.
Each agent writes a different key, so the parallel branches never collide.

**1. Segregator** (`app/agents.py: segregator_node`) — the only node that sees every page.
Pages are sent to the LLM in batches of 12 (first 1,500 characters of each page is enough to
recognise a document). The model must return one structured label per page from the 9 types, with a
one-line reason. It is told that a page can continue the previous page's document (e.g. a two-page
letter). Anything the model skips defaults to `other` and adds a warning. The node then builds
`routed` from a fixed table:

| Agent | Receives pages classified as |
|---|---|
| `id_agent` | `identity_document`, `claim_forms` (the policy number lives on the claim form) |
| `discharge_agent` | `discharge_summary` |
| `bill_agent` | `itemized_bill` |

Pages of the other types (bank details, prescription, lab reports, cash receipt, other) are
classified and reported but not extracted, as the assignment specifies.

**2. Extraction agents** — each reads only `state["routed"][its_name]`, never the full PDF or other
agents' pages, and asks the LLM for a typed Pydantic object (`app/schemas.py`).
- *ID agent*: patient name, DOB, gender, ID numbers, policy number, insurer, claim reference, contact.
- *Discharge agent*: diagnosis, admission/discharge dates, attending and other physicians, discharge condition, medications, follow-up.
- *Itemized bill agent*: every line item with date, qty, rate and amount, per bill. **The arithmetic is
  done in code with `Decimal`, not by the LLM**: item totals, `items − discount + tax`, a comparison
  with the printed total, and a qty × rate check on each line. Mismatches appear in an `issues` list.
  If a bill page is missing, the agent returns `status: "no_pages"` instead of failing.

**3. Aggregator** — merges the three results with the page classification map, warnings and model
name into the final JSON.

**API layer** (`app/main.py`): validates `claim_id`, checks the `%PDF` header, enforces size and page
limits, extracts text per page with `pypdf`, runs the graph in a threadpool, and maps failures to
400/413/422/502/503. The service starts even without an API key and reports it on `/health`.

## Result on the sample packet

Classification matched expectations for all 18 pages (claim form p1, bank p2, ID p3, discharge p4,
prescription p5, lab reports p6/11/12, cash receipt p7, bills p9/10, everything else `other`).
Bills: hospital bill 20 items = $6,113.00 → $6,418.65 with tax; pharmacy bill 12 items = $216.30 →
$206.35 after discount and tax; `total_amount` = **$6,625.00**. Both match the printed totals.

## Limitations

- Text-based PDFs only. A scanned page has no extractable text, is labelled `other` and a warning
  is added; OCR (or sending page images to a vision model) is the natural next step.
- The automated tests run on the `mock` provider. The Gemini/Anthropic/OpenAI paths use the same
  `with_structured_output` interface but were not exercised without keys; run one real claim before
  relying on them, and set `LLM_MODEL` if a default model name has been retired.
- Rows in a bill are extracted by the LLM; the code checks internal consistency but cannot detect a
  line the model dropped (a subtotal mismatch will flag most such cases).

## Video walkthrough outline (3–5 min)

1. The graph: show `build_graph` — segregator, parallel fan-out, fan-in aggregator.
2. Segregator: the prompt, batching, the structured label schema, the routing table.
3. Extraction agents: show that each receives only its pages (the test `test_agents_only_see_their_pages`), and the bill arithmetic in code.
4. Demo: curl the sample PDF, open the JSON, point at `page_classification` and the totals.

## Deploying (optional)

Render / Railway / Fly.io: deploy the Dockerfile, set `LLM_PROVIDER` and the key as environment
variables; the container listens on `$PORT`.
