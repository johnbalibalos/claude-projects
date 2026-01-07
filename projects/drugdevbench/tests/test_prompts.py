"""Tests for prompt construction."""

import pytest

from drugdevbench.data.schemas import FigureType, Persona, PromptCondition
from drugdevbench.prompts import (
    BASE_SCIENTIFIC_PROMPT,
    build_system_prompt,
    get_skill_for_figure_type,
    get_persona_prompt,
    get_skill_prompt,
)
from drugdevbench.prompts.personas import PERSONA_PROMPTS
from drugdevbench.prompts.skills import SKILL_PROMPTS


class TestBasePrompt:
    """Test base scientific prompt."""

    def test_base_prompt_exists(self):
        """Base prompt should be defined."""
        assert BASE_SCIENTIFIC_PROMPT is not None
        assert len(BASE_SCIENTIFIC_PROMPT) > 100

    def test_base_prompt_contains_key_sections(self):
        """Base prompt should contain key guidance sections."""
        assert "Visual Analysis" in BASE_SCIENTIFIC_PROMPT
        assert "Quantitative" in BASE_SCIENTIFIC_PROMPT
        assert "Scientific Context" in BASE_SCIENTIFIC_PROMPT


class TestPersonaPrompts:
    """Test persona prompts."""

    def test_all_personas_have_prompts(self):
        """Every persona should have a prompt defined."""
        for persona in Persona:
            assert persona in PERSONA_PROMPTS
            prompt = PERSONA_PROMPTS[persona]
            assert len(prompt) > 100

    def test_immunologist_prompt_contains_flow_cytometry(self):
        """Immunologist prompt should mention flow cytometry."""
        prompt = get_persona_prompt(Persona.IMMUNOLOGIST)
        assert "flow cytometry" in prompt.lower()

    def test_pharmacologist_prompt_contains_pk_terms(self):
        """Pharmacologist prompt should mention PK terms."""
        prompt = get_persona_prompt(Persona.PHARMACOLOGIST)
        prompt_lower = prompt.lower()
        assert "pharmacokinetic" in prompt_lower or "pk" in prompt_lower

    def test_molecular_biologist_prompt_contains_western_blot(self):
        """Molecular biologist prompt should mention Western blot."""
        prompt = get_persona_prompt(Persona.MOLECULAR_BIOLOGIST)
        assert "Western" in prompt or "blot" in prompt.lower()


class TestSkillPrompts:
    """Test skill prompts."""

    def test_all_skills_defined(self):
        """All expected skills should be defined."""
        expected_skills = [
            "western_blot",
            "dose_response",
            "pk_curve",
            "flow_biaxial",
            "flow_histogram",
            "heatmap",
            "elisa",
        ]
        for skill in expected_skills:
            assert skill in SKILL_PROMPTS
            assert len(SKILL_PROMPTS[skill]) > 100

    def test_western_blot_skill_mentions_loading_control(self):
        """Western blot skill should mention loading controls."""
        skill = get_skill_prompt("western_blot")
        assert "loading control" in skill.lower()

    def test_dose_response_skill_mentions_ic50(self):
        """Dose-response skill should mention IC50/EC50."""
        skill = get_skill_prompt("dose_response")
        skill_lower = skill.lower()
        assert "ic50" in skill_lower or "ec50" in skill_lower

    def test_pk_curve_skill_mentions_half_life(self):
        """PK curve skill should mention half-life."""
        skill = get_skill_prompt("pk_curve")
        assert "half-life" in skill.lower() or "t½" in skill


class TestSkillMapping:
    """Test figure type to skill mapping."""

    def test_western_blot_maps_to_western_blot_skill(self):
        """Western blot figure type should use western_blot skill."""
        assert get_skill_for_figure_type(FigureType.WESTERN_BLOT) == "western_blot"

    def test_dose_response_maps_to_dose_response_skill(self):
        """Dose-response figure type should use dose_response skill."""
        assert get_skill_for_figure_type(FigureType.DOSE_RESPONSE) == "dose_response"

    def test_pk_curve_maps_to_pk_curve_skill(self):
        """PK curve figure type should use pk_curve skill."""
        assert get_skill_for_figure_type(FigureType.PK_CURVE) == "pk_curve"

    def test_flow_biaxial_maps_to_flow_biaxial_skill(self):
        """Flow biaxial figure type should use flow_biaxial skill."""
        assert get_skill_for_figure_type(FigureType.FLOW_BIAXIAL) == "flow_biaxial"


class TestBuildSystemPrompt:
    """Test system prompt building for different conditions."""

    def test_vanilla_condition(self):
        """Vanilla condition should return minimal prompt."""
        prompt = build_system_prompt(
            condition=PromptCondition.VANILLA,
            figure_type=FigureType.WESTERN_BLOT,
        )
        assert "AI assistant" in prompt
        assert len(prompt) < 200  # Should be short

    def test_base_only_condition(self):
        """Base-only condition should include base prompt only."""
        prompt = build_system_prompt(
            condition=PromptCondition.BASE_ONLY,
            figure_type=FigureType.WESTERN_BLOT,
        )
        assert "Visual Analysis" in prompt
        # Should not include persona-specific or skill-specific content
        assert "loading control" not in prompt.lower()

    def test_base_plus_skill_condition(self):
        """Base+skill condition should include base and skill prompts."""
        prompt = build_system_prompt(
            condition=PromptCondition.BASE_PLUS_SKILL,
            figure_type=FigureType.WESTERN_BLOT,
        )
        assert "Visual Analysis" in prompt  # Base
        assert "loading control" in prompt.lower()  # Skill

    def test_full_stack_condition(self):
        """Full stack should include persona, base, and skill."""
        prompt = build_system_prompt(
            condition=PromptCondition.FULL_STACK,
            figure_type=FigureType.WESTERN_BLOT,
        )
        assert "Visual Analysis" in prompt  # Base
        assert "loading control" in prompt.lower()  # Skill
        # Should include molecular biologist persona
        assert "molecular" in prompt.lower() or "protein" in prompt.lower()

    def test_persona_only_condition(self):
        """Persona-only condition should include only persona prompt."""
        prompt = build_system_prompt(
            condition=PromptCondition.PERSONA_ONLY,
            figure_type=FigureType.FLOW_BIAXIAL,
        )
        # Should include immunologist persona
        assert "flow cytometry" in prompt.lower() or "immune" in prompt.lower()
        # Should not include base or skill
        assert "Visual Analysis" not in prompt

    def test_wrong_skill_condition(self):
        """Wrong skill condition should include mismatched skill."""
        # Run multiple times to ensure we don't get the right skill by chance
        got_different_skill = False
        for _ in range(10):
            prompt = build_system_prompt(
                condition=PromptCondition.WRONG_SKILL,
                figure_type=FigureType.WESTERN_BLOT,
            )
            # If it doesn't have western blot specific content, skill is different
            if "loading control" not in prompt.lower():
                got_different_skill = True
                break

        # Should eventually get a different skill
        # (though this could theoretically fail with low probability)
        assert got_different_skill or "loading control" not in prompt.lower()

    def test_prompt_builds_for_all_figure_types(self):
        """System prompt should build successfully for all figure types."""
        for figure_type in FigureType:
            for condition in PromptCondition:
                prompt = build_system_prompt(
                    condition=condition,
                    figure_type=figure_type,
                )
                assert isinstance(prompt, str)
                assert len(prompt) > 0
