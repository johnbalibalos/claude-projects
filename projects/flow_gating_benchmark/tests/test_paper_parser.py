"""Tests for paper_parser module - XML/PDF content extraction."""

import pytest
from pathlib import Path

from src.curation.paper_parser import (
    ExtractedTable,
    PaperParser,
    extract_gating_from_text,
    extract_panel_from_table,
)


class TestExtractedTable:
    """Tests for ExtractedTable dataclass."""

    def test_to_markdown(self):
        table = ExtractedTable(
            table_id="table1",
            caption="Panel reagents",
            headers=["Marker", "Fluorophore", "Clone"],
            rows=[
                ["CD3", "BUV395", "UCHT1"],
                ["CD4", "BUV496", "SK3"],
            ],
            table_type="panel",
            source_location="Table 1"
        )

        md = table.to_markdown()
        assert "| Marker | Fluorophore | Clone |" in md
        assert "| CD3 | BUV395 | UCHT1 |" in md
        assert "| CD4 | BUV496 | SK3 |" in md

    def test_get_column(self):
        table = ExtractedTable(
            table_id="table1",
            caption="Panel",
            headers=["Marker", "Fluorophore"],
            rows=[
                ["CD3", "BUV395"],
                ["CD4", "BUV496"],
            ],
            table_type="panel",
            source_location="Table 1"
        )

        markers = table.get_column("Marker")
        assert markers == ["CD3", "CD4"]

        fluors = table.get_column("fluorophore")  # Case insensitive
        assert fluors == ["BUV395", "BUV496"]

    def test_get_column_partial_match(self):
        table = ExtractedTable(
            table_id="table1",
            caption="Panel",
            headers=["Target Marker", "Dye"],
            rows=[["CD3", "PE"]],
            table_type="panel",
            source_location="Table 1"
        )

        # Should match partial "marker"
        markers = table.get_column("marker")
        assert markers == ["CD3"]


class TestExtractPanelFromTable:
    """Tests for extract_panel_from_table function."""

    def test_standard_panel_table(self):
        table = ExtractedTable(
            table_id="t1",
            caption="Antibody panel",
            headers=["Marker", "Fluorophore", "Clone", "Vendor"],
            rows=[
                ["CD3", "BUV395", "UCHT1", "BD"],
                ["CD4", "BUV496", "SK3", "BD"],
                ["CD8", "BV421", "RPA-T8", "BioLegend"],
            ],
            table_type="panel",
            source_location="Table 1"
        )

        entries = extract_panel_from_table(table)

        assert len(entries) == 3

        cd3 = next(e for e in entries if e["marker"] == "CD3")
        assert cd3["fluorophore"] == "BUV395"
        assert cd3["clone"] == "UCHT1"
        assert cd3["vendor"] == "BD"

    def test_alternate_column_names(self):
        table = ExtractedTable(
            table_id="t1",
            caption="Reagents",
            headers=["Target", "Conjugate", "Clone"],
            rows=[
                ["CD3", "PE", "OKT3"],
            ],
            table_type="panel",
            source_location="Table 1"
        )

        entries = extract_panel_from_table(table)
        assert len(entries) == 1
        assert entries[0]["marker"] == "CD3"
        assert entries[0]["fluorophore"] == "PE"

    def test_missing_columns(self):
        table = ExtractedTable(
            table_id="t1",
            caption="Panel",
            headers=["Marker", "Fluorophore"],
            rows=[["CD3", "PE"]],
            table_type="panel",
            source_location="Table 1"
        )

        entries = extract_panel_from_table(table)
        assert entries[0]["clone"] is None
        assert entries[0]["vendor"] is None


class TestExtractGatingFromText:
    """Tests for extract_gating_from_text function."""

    def test_extract_gating_sentences(self):
        text = """
        Samples were acquired on a flow cytometer. Live cells were gated
        based on viability dye exclusion. T cells were defined as CD3+CD19-
        lymphocytes. CD4+ T cells were further gated as CD3+CD4+CD8-.
        Statistical analysis was performed using GraphPad.
        """

        result = extract_gating_from_text(text)

        assert "gating_text" in result
        assert "markers_mentioned" in result
        assert result["sentence_count"] >= 2

        # Should find CD markers
        markers = [m.upper() for m in result["markers_mentioned"]]
        assert "CD3" in markers
        assert "CD4" in markers
        assert "CD19" in markers

    def test_extract_raw_text(self):
        text = "Singlets were gated on FSC-A vs FSC-H."

        result = extract_gating_from_text(text, return_raw=True)

        assert isinstance(result, str)
        assert "singlets" in result.lower()

    def test_empty_text(self):
        result = extract_gating_from_text("")
        assert result["sentence_count"] == 0


class TestTableClassification:
    """Tests for table classification logic."""

    def test_classify_panel_table(self):
        parser = PaperParser()

        # Panel table indicators
        caption = "Table 1. Antibody panel for T cell immunophenotyping"
        headers = ["Marker", "Fluorophore", "Clone"]

        table_type = parser._classify_table(caption, headers)
        assert table_type == "panel"

    def test_classify_gating_table(self):
        parser = PaperParser()

        caption = "Gating strategy and population definitions"
        headers = ["Population", "Phenotype", "Parent"]

        table_type = parser._classify_table(caption, headers)
        assert table_type == "gating"

    def test_classify_results_table(self):
        parser = PaperParser()

        caption = "Frequencies of immune cell populations"
        headers = ["Population", "Percent", "Mean", "SD"]

        table_type = parser._classify_table(caption, headers)
        assert table_type == "results"


class TestFigureClassification:
    """Tests for figure classification logic."""

    def test_classify_gating_figure(self):
        parser = PaperParser()

        caption = "Figure 1. Gating strategy for T cell subsets. Sequential gating shows..."
        fig_type = parser._classify_figure(caption)
        assert fig_type == "gating"

    def test_classify_results_figure(self):
        parser = PaperParser()

        caption = "Figure 2. Comparison of cell frequencies between groups."
        fig_type = parser._classify_figure(caption)
        assert fig_type == "results"


class TestPaperParserIntegration:
    """Integration tests for PaperParser (requires test data)."""

    @pytest.fixture
    def parser(self, tmp_path):
        """Create parser with temporary directory."""
        return PaperParser(papers_dir=tmp_path)

    def test_parser_init(self, parser, tmp_path):
        assert parser.papers_dir == tmp_path

    def test_find_xml_not_found(self, parser):
        result = parser._find_xml("OMIP-999")
        assert result is None

    def test_omip_to_pmcid_no_index(self, parser):
        # No index file exists
        result = parser._omip_to_pmcid("OMIP-069")
        assert result is None


class TestMarkerPatternExtraction:
    """Tests for marker pattern extraction from text."""

    def test_cd_markers(self):
        text = "CD3+ CD4+ CD8- CD19- CD45RA+ CCR7-"
        result = extract_gating_from_text(text)
        markers = [m.upper() for m in result["markers_mentioned"]]

        assert "CD3" in markers
        assert "CD4" in markers
        assert "CD8" in markers
        assert "CD19" in markers
        assert "CD45RA" in markers
        assert "CCR7" in markers

    def test_hla_markers(self):
        text = "HLA-DR expression was measured on monocytes."
        result = extract_gating_from_text(text)
        markers = [m.upper() for m in result["markers_mentioned"]]

        assert "HLA-DR" in markers

    def test_chemokine_receptors(self):
        text = "CCR7+ and CXCR5+ cells were identified."
        result = extract_gating_from_text(text)
        markers = [m.upper() for m in result["markers_mentioned"]]

        assert "CCR7" in markers
        assert "CXCR5" in markers
