#!/usr/bin/env python3
"""
Parse gating descriptions into structured hierarchies using LLM.

Takes extracted OMIP data with gating descriptions and uses an LLM
to convert the text into the gating_hierarchy schema format.
"""

import json
import os
import re
import sys
from pathlib import Path


def get_client(provider: str):
    """Get the appropriate API client."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    elif provider == "google":
        from google import genai
        return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")


SYSTEM_PROMPT = """You are an expert flow cytometrist and data extraction specialist.
Your task is to extract a gating hierarchy from text descriptions of flow cytometry experiments.

A gating hierarchy is a tree structure where:
- The root is typically "All Events"
- Each node represents a cell population defined by marker expression
- Children are subpopulations gated from the parent
- Markers can be positive (+), negative (-), high (hi), low (lo), or intermediate (int)

Output format (JSON):
{
  "root": {
    "name": "All Events",
    "markers": [],
    "marker_logic": null,
    "gate_type": "root",
    "children": [...],
    "is_critical": true
  }
}

Gate node schema:
- name: Human-readable population name (e.g., "CD4+ T cells", "Memory B cells")
- markers: List of markers used for this gate (e.g., ["CD3", "CD4"])
- marker_logic: Expression logic (e.g., "CD3+ CD4+", "CD45+ CD19-")
- gate_type: One of "scatter", "viability", "lineage", "subset", "functional", "Unknown"
- children: Array of child gates (subpopulations)
- is_critical: true for essential gates in standard immunophenotyping

Common gating patterns:
1. Scatter gates (FSC/SSC): Remove debris, identify lymphocytes
2. Singlet gates (FSC-A vs FSC-H): Remove doublets
3. Viability (Live/Dead stain negative): Remove dead cells
4. CD45+ leukocyte gate
5. Lineage gates (CD3 for T cells, CD19/CD20 for B cells, CD56 for NK)
6. Subset gates (CD4/CD8 for T cell subsets, memory markers, etc.)

Important:
- Extract ALL gates mentioned in the text, maintaining parent-child relationships
- Use marker logic exactly as described (+ means positive, - means negative)
- If a gate's parent is unclear, place it under the most logical ancestor
- Include both the figure caption and full text descriptions
- Be thorough - missing gates is worse than adding uncertain ones
"""


def create_prompt(extracted_data: dict) -> str:
    """Create the prompt for the LLM."""
    omip_id = extracted_data.get("omip_id") or extracted_data.get("test_case_id")
    context = extracted_data.get("context", {})
    panel = extracted_data.get("panel", {}).get("entries", [])

    # Build panel summary
    panel_text = "Panel markers:\n"
    for entry in panel:
        marker = entry.get("marker", "")
        purpose = entry.get("purpose", "")
        panel_text += f"- {marker}"
        if purpose:
            panel_text += f" ({purpose})"
        panel_text += "\n"

    # Get gating descriptions
    figure_caption = context.get("additional_notes", "")
    full_text = context.get("full_text_gating", "")

    prompt = f"""Extract the gating hierarchy from this flow cytometry paper.

## Paper Information
ID: {omip_id}
Species: {context.get('species', 'Unknown')}
Sample type: {context.get('sample_type', 'Unknown')}

## {panel_text}

## Figure Caption (Gating Strategy Overview)
{figure_caption}

## Full Text Description
{full_text if full_text else "(No additional text available)"}

## Task
Extract the complete gating hierarchy as JSON. Include:
1. All gates mentioned (debris exclusion, singlets, viability, lineage markers, subsets)
2. Proper parent-child relationships
3. Marker logic for each gate
4. Mark critical gates (standard immunophenotyping gates like singlets, live cells, major lineages)

Return ONLY the JSON object, no explanation needed."""

    return prompt


def parse_with_llm(extracted_data: dict, model: str = "sonnet-cli") -> dict:
    """Parse gating description using Claude CLI."""
    import subprocess
    import tempfile

    prompt = create_prompt(extracted_data)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    # Write prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(full_prompt)
        prompt_file = f.name

    try:
        # Call claude CLI with the prompt via pipe
        result = subprocess.run(
            f'cat "{prompt_file}" | claude --output-format text',
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            return {"error": f"CLI error: {result.stderr}", "raw_response": ""}

        text = result.stdout

    finally:
        os.unlink(prompt_file)

    # Try to find JSON block
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            hierarchy = json.loads(json_match.group())
            return hierarchy
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}", file=sys.stderr)
            return {"error": str(e), "raw_response": text}

    return {"error": "No JSON found in response", "raw_response": text}


def count_gates(node: dict) -> int:
    """Count total gates in hierarchy."""
    if not node:
        return 0
    count = 1
    for child in node.get("children", []):
        count += count_gates(child)
    return count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse gating descriptions with LLM")
    parser.add_argument("--input-dir", type=Path, default=Path("data/extracted"),
                        help="Directory with extracted JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/parsed"),
                        help="Output directory for parsed hierarchies")
    parser.add_argument("--model", default="sonnet-cli",
                        help="Model to use (uses Claude CLI)")
    parser.add_argument("--file", type=str, help="Process single file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt without calling API")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [args.input_dir / args.file]
    else:
        files = sorted(args.input_dir.glob("*.json"))

    print(f"Processing {len(files)} files with {args.model}")

    for json_path in files:
        print(f"\n{'='*60}")
        print(f"Processing: {json_path.name}")

        with open(json_path) as f:
            data = json.load(f)

        if args.dry_run:
            prompt = create_prompt(data)
            print(f"Prompt length: {len(prompt)} chars")
            print("-" * 40)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            continue

        # Check if we have gating descriptions
        context = data.get("context", {})
        if not context.get("additional_notes") and not context.get("full_text_gating"):
            print("  Skipping: No gating description")
            continue

        # Parse with LLM
        result = parse_with_llm(data, args.model)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        # Update the hierarchy in the data
        data["gating_hierarchy"] = result
        gate_count = count_gates(result.get("root", {}))
        print(f"  Extracted {gate_count} gates")

        # Save
        output_path = args.output_dir / json_path.name
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {output_path}")

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
