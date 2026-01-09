# Paper Download

PMC client for searching and downloading papers from PubMed Central.

## Features

- NCBI E-utilities integration for paper search
- XML content fetching via OA webservice
- Supplementary file downloads
- Metadata extraction (title, DOI, OMIP ID)
- Index generation (JSON)
- Rate limiting to comply with NCBI terms

## Installation

```bash
cd libs/paper_download
pip install -e .
```

Requires `httpx`.

## Usage

### Command Line

```bash
# Download OMIP papers
python -m paper_download --email user@example.com

# Custom query
python -m paper_download --email user@example.com --query "flow cytometry[Title]"

# With supplementary files
python -m paper_download --email user@example.com --with-supplementary
```

### Python API

```python
from paper_download import PMCClient

with PMCClient(email="user@example.com") as client:
    # Search for papers
    pmc_ids = client.search_papers(
        query="OMIP[Title] AND Cytometry[Journal]",
        max_results=50
    )

    # Download XML
    for pmc_id in pmc_ids:
        result = client.fetch_xml(pmc_id, output_dir="papers/")
        print(f"{pmc_id}: {result.status}")
```

### Download with Metadata

```python
from paper_download import PMCClient
from pathlib import Path

with PMCClient(email="user@example.com") as client:
    results, metadata = client.download_papers(
        query="OMIP[Title] AND Cytometry[Journal]",
        output_dir=Path("papers/"),
        max_results=100,
        with_supplementary=True,
        progress_callback=lambda cur, total, status: print(f"[{cur}/{total}] {status}")
    )

    # Save index
    client.save_index(metadata, "papers/index.json")
```

### Extract Metadata from Existing XML

```python
from paper_download import PMCClient
from pathlib import Path

client = PMCClient(email="user@example.com")

for xml_file in Path("papers/").glob("PMC*.xml"):
    info = client.extract_metadata(xml_file)
    if info.omip_id:
        print(f"{info.omip_id}: {info.title}")
```

## API Reference

### PMCClient

| Method | Description |
|--------|-------------|
| `search_papers(query, max_results)` | Search PMC and return list of PMC IDs |
| `fetch_xml(pmc_id, output_dir)` | Download paper XML |
| `fetch_supplementary(pmc_id, output_dir)` | Download supplementary files |
| `extract_metadata(xml_path)` | Extract metadata from XML |
| `download_papers(query, output_dir, ...)` | Search, download, and extract metadata |
| `save_index(metadata, output_path)` | Save paper index to JSON |

### Data Classes

**DownloadResult**
| Field | Type | Description |
|-------|------|-------------|
| `pmc_id` | str | PMC ID |
| `status` | str | "success", "cached", "error", "invalid_xml" |
| `path` | str | Path to downloaded file |
| `error` | str | Error message if failed |

**PaperMetadata**
| Field | Type | Description |
|-------|------|-------------|
| `pmc_id` | str | PMC ID |
| `title` | str | Paper title |
| `doi` | str | DOI |
| `omip_id` | str | OMIP number (e.g., "OMIP-069") |
| `xml_path` | str | Path to XML file |

## Rate Limiting

The client includes built-in rate limiting (0.35s between requests) to comply with NCBI's terms of use. For higher throughput, register for an NCBI API key.

## Output Structure

```
papers/
├── PMC1234567.xml
├── PMC1234568.xml
├── PMC1234567_supp/     # Supplementary files
│   └── DataS1.xlsx
└── index.json           # Paper index
```

## License

MIT
