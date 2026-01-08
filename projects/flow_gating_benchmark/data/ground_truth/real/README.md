# Real Test Cases

Test cases curated from actual OMIP papers with verified panel and gating data.

**To be populated**: Download papers using the PMC client and manually curate
accurate gating hierarchies from the paper figures and methods sections.

## Curation Process
1. Download paper XML via `PMCClient.doi_to_pmcid()` and `fetch_full_text_xml()`
2. Extract gating section using `extract_gating_section()`
3. Manually verify and curate panel entries from paper tables
4. Extract gating hierarchy from paper figures
5. Validate against FlowRepository workspace files (when available)
