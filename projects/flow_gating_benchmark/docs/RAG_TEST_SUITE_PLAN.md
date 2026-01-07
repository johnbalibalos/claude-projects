# RAG Test Suite Implementation Plan

## Overview

This document outlines the implementation of a RAG-based test suite for the Flow Gating Benchmark. The goal is to establish a ceiling test where LLM performance should approach (but not necessarily reach) 100% when given direct access to source OMIP papers.

## Key Design Decisions

### 1. Score Expectations

| Tier | Description | Expected F1 | Rationale |
|------|-------------|-------------|-----------|
| Oracle RAG | Exact gating section provided | ≥0.95 | Parsing/normalization edge cases |
| Full Paper RAG | Retrieved chunks from full paper | ≥0.85 | Retrieval noise, chunking artifacts |
| Cross-Validation | Leave-one-out testing | ≥0.75 | Comparable to rich context baseline |
| Negative Control | Wrong paper provided | ~0.54 | Should match minimal context baseline |

**Why not 100%?**
- Multiple valid gating strategies exist for same panel
- Fuzzy matching has inherent limitations (γδ vs gd)
- Response parsing is 94.4% successful, not 100%
- Ground truth represents one curator's interpretation

### 2. Expert Annotation System for Fuzzy Matching Gaps

#### Problem
Current `normalize_gate_name()` handles common cases but misses:
- Unicode variants: `γδ T cells` vs `gamma-delta T cells`
- Domain abbreviations: `DN` vs `Double Negative`
- Population synonyms: `Tregs` vs `CD4+CD25+FoxP3+`

#### Solution: Capture → Annotate → Learn Loop

```
┌──────────────────────────────────────────────────────────────────┐
│                    ANNOTATION PIPELINE                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: Capture Near-Misses                                    │
│  ────────────────────────────                                    │
│  During scoring, when gates don't match:                         │
│  1. Compute string similarity (Levenshtein ratio)                │
│  2. If 0.5 < similarity < 0.9, save as candidate                 │
│  3. Include context: test_case_id, parent gates, markers         │
│                                                                  │
│  PHASE 2: Expert Annotation                                      │
│  ────────────────────────────                                    │
│  Experts review pending_annotations.jsonl:                       │
│  - Mark as "equivalent" or "different"                           │
│  - Optionally specify canonical form                             │
│  - Add notes for ambiguous cases                                 │
│                                                                  │
│  PHASE 3: Integration                                            │
│  ────────────────────────────                                    │
│  Approved equivalences loaded into scorer:                       │
│  - Pre-matching check against equivalence classes                │
│  - Falls back to standard fuzzy matching                         │
│  - Logs when equivalence rules fire for debugging                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Data Structures

```yaml
# data/annotations/verified_equivalences.yaml
version: "1.0"
last_updated: "2024-01-15"

equivalence_classes:
  - canonical: "gamma-delta t cells"
    variants:
      - "γδ t cells"
      - "gd t cells"
      - "γδ"
      - "gdt"
    domain: "t_cell_subsets"
    notes: "Greek letter gamma-delta often rendered differently"

  - canonical: "double negative"
    variants:
      - "dn"
      - "cd4-cd8-"
      - "dn thymocytes"
      - "double negative t cells"
    domain: "t_cell_development"

  - canonical: "regulatory t cells"
    variants:
      - "tregs"
      - "treg"
      - "t regulatory"
      - "cd4+cd25+foxp3+"
      - "cd4+cd25hi"
    domain: "t_cell_subsets"

  - canonical: "natural killer cells"
    variants:
      - "nk cells"
      - "nk"
      - "cd3-cd56+"
      - "cd56+ nk"
    domain: "innate_lymphoid"

# Pattern-based equivalences (regex)
patterns:
  - pattern: "^(.+) cells$"
    equivalent: "\\1s"
    bidirectional: true
    # "T cells" ↔ "Ts", "B cells" ↔ "Bs"

  - pattern: "^cd(\\d+) positive$"
    equivalent: "cd\\1+"
    bidirectional: true
```

```jsonl
# data/annotations/pending_annotations.jsonl (auto-generated)
{"id": "ann_001", "predicted": "gd t cells", "ground_truth": "γδ T cells", "similarity": 0.72, "test_case": "OMIP-044", "parent_context": "CD3+ T cells", "status": "pending", "created": "2024-01-15T10:30:00Z"}
{"id": "ann_002", "predicted": "DN thymocytes", "ground_truth": "Double Negative", "similarity": 0.65, "test_case": "OMIP-026", "parent_context": "Lymphocytes", "status": "pending", "created": "2024-01-15T10:31:00Z"}
```

#### Implementation

```python
# src/evaluation/equivalences.py

from dataclasses import dataclass
from pathlib import Path
import yaml
import json
import re
from difflib import SequenceMatcher

@dataclass
class EquivalenceClass:
    """A set of gate names that are semantically equivalent."""
    canonical: str
    variants: set[str]
    domain: str | None = None
    notes: str | None = None

class EquivalenceRegistry:
    """Loads and applies expert-verified equivalences."""

    def __init__(self, equivalences_path: Path):
        self.equivalences_path = equivalences_path
        self.classes: list[EquivalenceClass] = []
        self.lookup: dict[str, str] = {}  # variant -> canonical
        self.patterns: list[tuple[re.Pattern, str]] = []
        self._load()

    def _load(self):
        """Load equivalences from YAML file."""
        if not self.equivalences_path.exists():
            return

        with open(self.equivalences_path) as f:
            data = yaml.safe_load(f)

        for eq in data.get("equivalence_classes", []):
            canonical = eq["canonical"].lower()
            variants = {v.lower() for v in eq.get("variants", [])}
            variants.add(canonical)

            ec = EquivalenceClass(
                canonical=canonical,
                variants=variants,
                domain=eq.get("domain"),
                notes=eq.get("notes")
            )
            self.classes.append(ec)

            for v in variants:
                self.lookup[v] = canonical

        for p in data.get("patterns", []):
            self.patterns.append((
                re.compile(p["pattern"], re.IGNORECASE),
                p["equivalent"]
            ))

    def get_canonical(self, gate_name: str) -> str:
        """Get canonical form of a gate name."""
        normalized = gate_name.lower().strip()

        # Direct lookup
        if normalized in self.lookup:
            return self.lookup[normalized]

        # Pattern matching
        for pattern, replacement in self.patterns:
            if pattern.match(normalized):
                canonical = pattern.sub(replacement, normalized)
                if canonical in self.lookup:
                    return self.lookup[canonical]

        return normalized

    def are_equivalent(self, gate1: str, gate2: str) -> bool:
        """Check if two gate names are equivalent."""
        return self.get_canonical(gate1) == self.get_canonical(gate2)


class AnnotationCapture:
    """Captures near-miss gate pairs for expert review."""

    SIMILARITY_THRESHOLD_LOW = 0.5
    SIMILARITY_THRESHOLD_HIGH = 0.9

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.pending: list[dict] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing pending annotations."""
        if self.output_path.exists():
            with open(self.output_path) as f:
                for line in f:
                    if line.strip():
                        self.pending.append(json.loads(line))

    def check_and_capture(
        self,
        predicted: str,
        ground_truth: str,
        test_case_id: str,
        parent_context: str | None = None,
    ) -> bool:
        """
        Check if pair is a near-miss and capture for annotation.

        Returns True if captured (should not count as match yet).
        """
        similarity = SequenceMatcher(
            None,
            predicted.lower(),
            ground_truth.lower()
        ).ratio()

        if self.SIMILARITY_THRESHOLD_LOW < similarity < self.SIMILARITY_THRESHOLD_HIGH:
            # Check if already captured
            existing = any(
                a["predicted"].lower() == predicted.lower() and
                a["ground_truth"].lower() == ground_truth.lower()
                for a in self.pending
            )

            if not existing:
                from datetime import datetime
                annotation = {
                    "id": f"ann_{len(self.pending):04d}",
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                    "similarity": round(similarity, 3),
                    "test_case": test_case_id,
                    "parent_context": parent_context,
                    "status": "pending",
                    "created": datetime.now().isoformat(),
                }
                self.pending.append(annotation)

            return True

        return False

    def save(self):
        """Save pending annotations to file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            for ann in self.pending:
                f.write(json.dumps(ann) + "\n")

    def get_stats(self) -> dict:
        """Get annotation statistics."""
        pending = sum(1 for a in self.pending if a["status"] == "pending")
        verified = sum(1 for a in self.pending if a["status"] == "verified")
        rejected = sum(1 for a in self.pending if a["status"] == "rejected")
        return {
            "total": len(self.pending),
            "pending": pending,
            "verified": verified,
            "rejected": rejected,
        }
```

### 3. PMC API Integration for OMIP Papers

OMIP papers are open access and available through PubMed Central. We'll use the NCBI E-utilities API to programmatically download and cache papers.

#### PMC API Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    PMC DOWNLOAD PIPELINE                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ Test Cases  │────►│ Extract DOI │────►│ DOI→PMCID   │        │
│  │ (JSON)      │     │             │     │ Lookup      │        │
│  └─────────────┘     └─────────────┘     └──────┬──────┘        │
│                                                  │               │
│                                                  ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ Cache Hit?  │◄────│ PMC OA API  │◄────│ Fetch Full  │        │
│  │ Return      │     │ (XML/PDF)   │     │ Text        │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### PMC API Endpoints

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `eutils/esearch` | Search for PMCID by DOI | `doi:10.1002/cyto.a.21916` |
| `eutils/efetch` | Fetch article XML | `db=pmc&id=PMC1234567` |
| `oa/oa.fcgi` | Open Access PDF/XML | `id=PMC1234567` |

#### Implementation

```python
# src/rag/pmc_client.py

import httpx
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import time
import hashlib

class PMCClient:
    """Client for downloading OMIP papers from PubMed Central."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

    def __init__(
        self,
        cache_dir: Path,
        email: str,  # Required by NCBI
        api_key: Optional[str] = None,  # Optional, increases rate limit
        rate_limit_delay: float = 0.34,  # 3 requests/sec without key
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay if not api_key else 0.1
        self._last_request_time = 0

    def _rate_limit(self):
        """Respect NCBI rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, identifier: str, format: str) -> Path:
        """Get cache file path for an identifier."""
        safe_id = identifier.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_id}.{format}"

    def doi_to_pmcid(self, doi: str) -> Optional[str]:
        """Convert DOI to PMCID using ID Converter API."""
        cache_path = self._get_cache_path(doi, "pmcid.json")

        # Check cache
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
                return data.get("pmcid")

        self._rate_limit()

        # Use ID Converter API
        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {
            "ids": doi,
            "format": "json",
            "email": self.email,
        }

        try:
            with httpx.Client() as client:
                response = client.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

            records = data.get("records", [])
            if records and "pmcid" in records[0]:
                pmcid = records[0]["pmcid"]

                # Cache result
                with open(cache_path, "w") as f:
                    json.dump({"doi": doi, "pmcid": pmcid}, f)

                return pmcid

        except Exception as e:
            print(f"Error converting DOI {doi}: {e}")

        return None

    def fetch_full_text_xml(self, pmcid: str) -> Optional[str]:
        """Fetch full text XML from PMC."""
        cache_path = self._get_cache_path(pmcid, "xml")

        # Check cache
        if cache_path.exists():
            return cache_path.read_text()

        self._rate_limit()

        # Fetch from PMC
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            with httpx.Client() as client:
                response = client.get(url, params=params, timeout=60)
                response.raise_for_status()
                xml_content = response.text

                # Cache result
                cache_path.write_text(xml_content)

                return xml_content

        except Exception as e:
            print(f"Error fetching {pmcid}: {e}")

        return None

    def fetch_pdf(self, pmcid: str) -> Optional[Path]:
        """Fetch PDF from PMC Open Access subset."""
        cache_path = self._get_cache_path(pmcid, "pdf")

        # Check cache
        if cache_path.exists():
            return cache_path

        self._rate_limit()

        # First, get the OA file list
        params = {"id": pmcid, "format": "json"}

        try:
            with httpx.Client() as client:
                response = client.get(self.OA_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

            records = data.get("records", [])
            if not records:
                return None

            # Find PDF link
            pdf_url = None
            for link in records[0].get("links", []):
                if link.get("format") == "pdf":
                    pdf_url = link.get("href")
                    break

            if not pdf_url:
                return None

            # Download PDF
            self._rate_limit()
            with httpx.Client() as client:
                response = client.get(pdf_url, timeout=120, follow_redirects=True)
                response.raise_for_status()

                cache_path.write_bytes(response.content)
                return cache_path

        except Exception as e:
            print(f"Error fetching PDF for {pmcid}: {e}")

        return None

    def extract_gating_section(self, xml_content: str) -> Optional[str]:
        """
        Extract gating-related sections from PMC XML.

        Looks for:
        - Methods section containing "gating"
        - Figure legends mentioning "gating strategy"
        - Results sections about flow cytometry
        """
        try:
            root = ET.fromstring(xml_content)

            sections = []

            # Find all <sec> elements
            for sec in root.iter("sec"):
                sec_type = sec.get("sec-type", "")
                title_elem = sec.find("title")
                title = title_elem.text.lower() if title_elem is not None and title_elem.text else ""

                # Get full text of section
                text_parts = []
                for elem in sec.iter():
                    if elem.text:
                        text_parts.append(elem.text)
                    if elem.tail:
                        text_parts.append(elem.tail)
                full_text = " ".join(text_parts).lower()

                # Check if gating-related
                gating_keywords = ["gating", "gate", "hierarchy", "population", "subset"]
                if any(kw in title or kw in full_text[:500] for kw in gating_keywords):
                    sections.append(" ".join(text_parts))

            # Also check figure captions
            for fig in root.iter("fig"):
                caption = fig.find("caption")
                if caption is not None:
                    caption_text = " ".join(caption.itertext())
                    if "gating" in caption_text.lower():
                        sections.append(caption_text)

            return "\n\n---\n\n".join(sections) if sections else None

        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return None


class OMIPPaperDownloader:
    """Download and cache all OMIP papers for the benchmark."""

    def __init__(
        self,
        test_cases_dir: Path,
        cache_dir: Path,
        email: str,
        api_key: Optional[str] = None,
    ):
        self.test_cases_dir = test_cases_dir
        self.cache_dir = cache_dir
        self.client = PMCClient(cache_dir / "pmc", email, api_key)

    def load_test_cases(self) -> list[dict]:
        """Load all test cases."""
        cases = []
        for path in sorted(self.test_cases_dir.glob("*.json")):
            with open(path) as f:
                cases.append(json.load(f))
        return cases

    def download_all(self, formats: list[str] = ["xml", "pdf"]) -> dict:
        """
        Download all OMIP papers.

        Returns dict mapping OMIP ID to download status.
        """
        cases = self.load_test_cases()
        results = {}

        for case in cases:
            omip_id = case["omip_id"]
            doi = case.get("doi")

            if not doi:
                results[omip_id] = {"status": "no_doi"}
                continue

            print(f"Processing {omip_id} (DOI: {doi})...")

            # Convert DOI to PMCID
            pmcid = self.client.doi_to_pmcid(doi)
            if not pmcid:
                results[omip_id] = {"status": "no_pmcid", "doi": doi}
                continue

            result = {"status": "success", "pmcid": pmcid, "doi": doi}

            # Download XML
            if "xml" in formats:
                xml = self.client.fetch_full_text_xml(pmcid)
                result["xml"] = xml is not None

                # Extract gating section
                if xml:
                    gating = self.client.extract_gating_section(xml)
                    result["gating_section"] = gating is not None

            # Download PDF
            if "pdf" in formats:
                pdf_path = self.client.fetch_pdf(pmcid)
                result["pdf"] = pdf_path is not None

            results[omip_id] = result

        return results

    def get_paper_content(self, omip_id: str) -> dict:
        """Get cached content for an OMIP paper."""
        # Find test case
        case_path = self.test_cases_dir / f"{omip_id.lower().replace('-', '_')}.json"
        if not case_path.exists():
            raise FileNotFoundError(f"No test case for {omip_id}")

        with open(case_path) as f:
            case = json.load(f)

        doi = case.get("doi")
        if not doi:
            raise ValueError(f"No DOI for {omip_id}")

        pmcid = self.client.doi_to_pmcid(doi)
        if not pmcid:
            raise ValueError(f"Could not resolve PMCID for {doi}")

        result = {"omip_id": omip_id, "doi": doi, "pmcid": pmcid}

        # Get XML
        xml_path = self.client._get_cache_path(pmcid, "xml")
        if xml_path.exists():
            result["xml"] = xml_path.read_text()
            result["gating_section"] = self.client.extract_gating_section(result["xml"])

        # Get PDF path
        pdf_path = self.client._get_cache_path(pmcid, "pdf")
        if pdf_path.exists():
            result["pdf_path"] = pdf_path

        return result
```

#### Usage

```python
# Download all OMIP papers
downloader = OMIPPaperDownloader(
    test_cases_dir=Path("data/ground_truth"),
    cache_dir=Path("data/papers"),
    email="researcher@example.com",
    api_key=os.environ.get("NCBI_API_KEY"),  # Optional
)

# Download everything (one-time)
results = downloader.download_all()
print(f"Downloaded {sum(1 for r in results.values() if r['status'] == 'success')}/{len(results)} papers")

# Get content for specific OMIP
content = downloader.get_paper_content("OMIP-044")
print(f"Gating section: {content.get('gating_section', 'Not found')[:500]}")
```

#### Rate Limits & Best Practices

| Scenario | Rate Limit | Notes |
|----------|------------|-------|
| Without API key | 3 req/sec | Built-in delay |
| With API key | 10 req/sec | Get key from NCBI |
| Bulk download | Use off-peak hours | Weekends/nights |

**Get NCBI API key:** https://www.ncbi.nlm.nih.gov/account/settings/

---

### 4. LlamaIndex RAG Integration

#### Why LlamaIndex?
- Battle-tested PDF parsing for scientific papers
- Multiple chunking strategies (semantic, hierarchical)
- Easy embedding model swapping
- Built-in query engine abstractions

#### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ OMIP PDFs   │────►│ LlamaIndex  │────►│ Vector      │        │
│  │ (raw)       │     │ Parser      │     │ Store       │        │
│  └─────────────┘     └─────────────┘     └──────┬──────┘        │
│                                                  │               │
│                                                  ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ Panel +     │────►│ Retriever   │────►│ LLM with    │        │
│  │ Query       │     │ (top-k)     │     │ RAG Context │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
# src/rag/omip_index.py

from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode

class OMIPIndexer:
    """Index OMIP papers for RAG retrieval."""

    def __init__(
        self,
        papers_dir: Path,
        index_dir: Path,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ):
        self.papers_dir = papers_dir
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Configure embedding
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)

        self.index: VectorStoreIndex | None = None

    def build_index(self, force_rebuild: bool = False):
        """Build or load the vector index."""
        if self.index_dir.exists() and not force_rebuild:
            # Load existing index
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_dir)
            )
            self.index = load_index_from_storage(storage_context)
            return

        # Parse PDFs
        documents = SimpleDirectoryReader(
            input_dir=str(self.papers_dir),
            filename_as_id=True,
            required_exts=[".pdf"],
        ).load_data()

        # Add OMIP ID metadata
        for doc in documents:
            # Extract OMIP ID from filename (e.g., "OMIP-044.pdf")
            filename = Path(doc.metadata.get("file_name", "")).stem
            if filename.startswith("OMIP-"):
                doc.metadata["omip_id"] = filename

        # Chunking with semantic splitting for better boundaries
        node_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model,
        )

        # Fallback to sentence splitter for sections semantic splitter misses
        sentence_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        nodes = node_parser.get_nodes_from_documents(documents)

        # Build index
        self.index = VectorStoreIndex(nodes)

        # Persist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(self.index_dir))

    def retrieve(
        self,
        query: str,
        omip_id: str | None = None,
        top_k: int = 5,
    ) -> list[TextNode]:
        """Retrieve relevant chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        retriever = self.index.as_retriever(similarity_top_k=top_k)

        # Add OMIP filter if specified
        if omip_id:
            # Filter to specific paper
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters={"omip_id": omip_id},
            )

        nodes = retriever.retrieve(query)
        return nodes


class OracleRetriever:
    """
    'Oracle' retriever that returns the exact gating section.

    Used for Tier 1 ceiling tests where we want to test
    extraction capability without retrieval noise.
    """

    def __init__(self, gating_sections: dict[str, str]):
        """
        Args:
            gating_sections: Dict mapping OMIP ID to gating section text
        """
        self.gating_sections = gating_sections

    def retrieve(self, omip_id: str) -> str:
        """Get the exact gating section for an OMIP."""
        if omip_id not in self.gating_sections:
            raise KeyError(f"No gating section for {omip_id}")
        return self.gating_sections[omip_id]
```

### 4. Test Suite Structure

```
src/rag_tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_tier1_oracle.py     # Exact section → ≥95% F1
├── test_tier2_retrieval.py  # Full paper RAG → ≥85% F1
├── test_tier3_crossval.py   # Leave-one-out → ≥75% F1
├── test_tier4_negative.py   # Wrong paper → baseline level
└── fixtures/
    └── oracle_sections.yaml # Hand-extracted gating sections
```

#### Example Test

```python
# src/rag_tests/test_tier1_oracle.py

import pytest
from pathlib import Path

from rag.omip_index import OracleRetriever
from evaluation.metrics import evaluate_prediction
from evaluation.equivalences import EquivalenceRegistry
from experiments.runner import predict_gating_strategy

# Fixture: load oracle sections
@pytest.fixture
def oracle_retriever():
    sections_path = Path(__file__).parent / "fixtures" / "oracle_sections.yaml"
    with open(sections_path) as f:
        sections = yaml.safe_load(f)
    return OracleRetriever(sections)

@pytest.fixture
def equivalence_registry():
    eq_path = Path(__file__).parent.parent.parent / "data" / "annotations" / "verified_equivalences.yaml"
    return EquivalenceRegistry(eq_path)


class TestOracleRAG:
    """Tier 1: Test extraction with perfect retrieval."""

    EXPECTED_F1_THRESHOLD = 0.95

    @pytest.mark.parametrize("omip_id", [
        "OMIP-001", "OMIP-003", "OMIP-005",  # Simple panels
        "OMIP-027", "OMIP-044",              # Medium panels
        "OMIP-069",                          # Complex (40-color)
    ])
    def test_oracle_extraction(
        self,
        omip_id: str,
        oracle_retriever,
        equivalence_registry,
        test_cases,  # Loaded from ground_truth/
    ):
        """
        Given the exact gating section, LLM should achieve ≥95% F1.

        Failures indicate:
        - Extraction bugs in response parsing
        - Normalization gaps (add to pending annotations)
        - Ground truth issues
        """
        test_case = test_cases[omip_id]

        # Get oracle context
        gating_context = oracle_retriever.retrieve(omip_id)

        # Predict with oracle context
        response = predict_gating_strategy(
            panel=test_case.panel,
            context=gating_context,
            model="claude-sonnet-4-20250514",
        )

        # Evaluate with equivalence support
        result = evaluate_prediction(
            predicted=response,
            ground_truth=test_case.gating_hierarchy,
            panel=test_case.panel,
            equivalence_registry=equivalence_registry,
        )

        # Assert threshold
        assert result.hierarchy_f1 >= self.EXPECTED_F1_THRESHOLD, (
            f"{omip_id}: F1={result.hierarchy_f1:.3f} < {self.EXPECTED_F1_THRESHOLD}\n"
            f"Missing: {result.missing_gates}\n"
            f"Extra: {result.extra_gates}"
        )

    def test_aggregate_oracle_f1(
        self,
        oracle_retriever,
        equivalence_registry,
        test_cases,
    ):
        """Mean F1 across all test cases should exceed threshold."""
        f1_scores = []
        failures = []

        for omip_id, test_case in test_cases.items():
            try:
                gating_context = oracle_retriever.retrieve(omip_id)
                response = predict_gating_strategy(
                    panel=test_case.panel,
                    context=gating_context,
                    model="claude-sonnet-4-20250514",
                )
                result = evaluate_prediction(
                    predicted=response,
                    ground_truth=test_case.gating_hierarchy,
                    panel=test_case.panel,
                    equivalence_registry=equivalence_registry,
                )
                f1_scores.append(result.hierarchy_f1)

                if result.hierarchy_f1 < self.EXPECTED_F1_THRESHOLD:
                    failures.append((omip_id, result))
            except Exception as e:
                failures.append((omip_id, str(e)))

        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        assert mean_f1 >= self.EXPECTED_F1_THRESHOLD, (
            f"Mean F1={mean_f1:.3f} < {self.EXPECTED_F1_THRESHOLD}\n"
            f"Failures: {len(failures)}/{len(test_cases)}"
        )
```

### 6. Dependencies to Add

```txt
# requirements.txt additions

# RAG pipeline
llama-index>=0.10.0
llama-index-embeddings-openai>=0.1.0
llama-index-readers-file>=0.1.0  # PDF support

# PMC API
httpx>=0.25.0  # Already in requirements

# Annotation system
PyYAML>=6.0
python-Levenshtein>=0.21.0  # Fast string similarity
```

### 7. Implementation Order

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION PHASES                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 0: PMC Paper Download (prerequisite)          ~1 day        │
│  ─────────────────────────────────────────────                      │
│  ├── Implement PMCClient with caching                               │
│  ├── Implement OMIPPaperDownloader                                  │
│  ├── Download all 30 OMIP papers (XML + PDF)                        │
│  ├── Validate gating section extraction                             │
│  └── Output: data/papers/pmc/ cache populated                       │
│                                                                     │
│  PHASE 1: Annotation Infrastructure (foundation)     ~1 day        │
│  ─────────────────────────────────────────────                      │
│  ├── Create src/evaluation/equivalences.py                          │
│  ├── Create data/annotations/ structure                             │
│  ├── Seed verified_equivalences.yaml with known mappings            │
│  │   - γδ T cells variants                                          │
│  │   - DN/Double Negative                                           │
│  │   - Tregs/Regulatory T cells                                     │
│  │   - NK cells variants                                            │
│  │   - Memory subset abbreviations (CM, EM, TEMRA)                  │
│  ├── Integrate EquivalenceRegistry into compute_hierarchy_f1()      │
│  └── Add AnnotationCapture to save near-misses                      │
│                                                                     │
│  PHASE 2: Oracle Test Suite (validate extraction)    ~1 day        │
│  ─────────────────────────────────────────────                      │
│  ├── Create src/rag_tests/ directory structure                      │
│  ├── Extract oracle gating sections from PMC XML                    │
│  │   - Use extract_gating_section() output                          │
│  │   - Manual review/cleanup for 10 test OMIPs                      │
│  ├── Create test_tier1_oracle.py                                    │
│  ├── Run and identify:                                              │
│  │   - Parsing failures → fix response_parser.py                    │
│  │   - Normalization gaps → add to pending_annotations              │
│  │   - Ground truth issues → flag for curator review                │
│  └── Target: ≥95% F1 with oracle context                            │
│                                                                     │
│  PHASE 3: LlamaIndex RAG (full pipeline)             ~2 days       │
│  ─────────────────────────────────────────────                      │
│  ├── Create src/rag/omip_index.py with LlamaIndex                   │
│  ├── Implement OMIPIndexer:                                         │
│  │   - PDF parsing with llama-index-readers-file                    │
│  │   - Semantic chunking (prioritize Methods, Figure legends)       │
│  │   - Vector store with OpenAI embeddings                          │
│  ├── Implement retrieval with OMIP-specific filtering               │
│  ├── Create test_tier2_retrieval.py                                 │
│  ├── A/B test chunking strategies:                                  │
│  │   - Semantic vs. fixed-size                                      │
│  │   - Different chunk sizes (512, 1024, 2048)                      │
│  │   - Top-k values (3, 5, 10)                                      │
│  └── Target: ≥85% F1 with retrieval                                 │
│                                                                     │
│  PHASE 4: Cross-Validation & Controls               ~1 day         │
│  ─────────────────────────────────────────────                      │
│  ├── Implement leave-one-out test:                                  │
│  │   - For each OMIP, retrieve from all OTHER papers                │
│  │   - Tests generalization, not memorization                       │
│  ├── Implement negative controls:                                   │
│  │   - Test with unrelated paper context                            │
│  │   - Test with no context (minimal baseline)                      │
│  ├── Create test_tier3_crossval.py                                  │
│  ├── Create test_tier4_negative.py                                  │
│  └── Document findings in REPORT.md                                 │
│                                                                     │
│  PHASE 5: Expert Annotation Review                   Ongoing       │
│  ─────────────────────────────────────────────                      │
│  ├── Review pending_annotations.jsonl after each run                │
│  ├── Mark equivalences as verified/rejected                         │
│  ├── Update verified_equivalences.yaml                              │
│  └── Re-run tests to validate improvements                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8. Directory Structure After Implementation

```
projects/flow_gating_benchmark/
├── src/
│   ├── evaluation/
│   │   ├── metrics.py           # Updated with equivalence support
│   │   ├── equivalences.py      # NEW: EquivalenceRegistry, AnnotationCapture
│   │   └── ...
│   ├── rag/                     # NEW: RAG components
│   │   ├── __init__.py
│   │   ├── pmc_client.py        # PMC API client
│   │   ├── omip_index.py        # LlamaIndex integration
│   │   └── oracle_retriever.py  # Oracle context provider
│   └── rag_tests/               # NEW: RAG test suite
│       ├── __init__.py
│       ├── conftest.py          # Pytest fixtures
│       ├── test_tier1_oracle.py
│       ├── test_tier2_retrieval.py
│       ├── test_tier3_crossval.py
│       ├── test_tier4_negative.py
│       └── fixtures/
│           └── oracle_sections.yaml
├── data/
│   ├── ground_truth/            # Existing test cases
│   ├── papers/                  # NEW: Cached OMIP papers
│   │   └── pmc/
│   │       ├── PMC1234567.xml
│   │       ├── PMC1234567.pdf
│   │       └── ...
│   └── annotations/             # NEW: Expert annotations
│       ├── verified_equivalences.yaml
│       └── pending_annotations.jsonl
└── docs/
    └── RAG_TEST_SUITE_PLAN.md   # This document
```

## Open Questions

1. ~~**OMIP PDF availability**~~ - RESOLVED: Use PMC API to download open access papers programmatically

2. **Embedding model choice** - `text-embedding-3-small` is cost-effective but `text-embedding-3-large` may improve retrieval. Worth A/B testing?

3. **Annotation UI** - Should we build a simple web UI for expert annotation, or is YAML/JSON editing sufficient?
   - Option A: Simple CLI tool to iterate through pending annotations
   - Option B: Streamlit app for visual review
   - Option C: Direct YAML editing (lowest overhead)

4. **CI integration** - Should failing RAG tests block merges, or just report warnings?
   - Suggestion: Tier 1 (oracle) blocks, Tiers 2-4 warn only

5. **PMC coverage** - Some older OMIPs may not have full text in PMC. Fallback strategy?
   - Could use DOI-based PDF download from Wiley as backup
   - May need institutional access for non-OA papers

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tier 1 mean F1 | ≥0.95 | Aggregate across all OMIPs |
| Tier 2 mean F1 | ≥0.85 | With retrieval pipeline |
| Annotation coverage | ≥90% | % of near-misses reviewed |
| Parse success rate | ≥98% | No regression from current 94.4% |
