"""Figure-type specific skills for interpretation."""

from drugdevbench.prompts.skills.western_blot import WESTERN_BLOT_SKILL
from drugdevbench.prompts.skills.dose_response import DOSE_RESPONSE_SKILL
from drugdevbench.prompts.skills.pk_curve import PK_CURVE_SKILL
from drugdevbench.prompts.skills.flow_biaxial import FLOW_BIAXIAL_SKILL
from drugdevbench.prompts.skills.flow_histogram import FLOW_HISTOGRAM_SKILL
from drugdevbench.prompts.skills.heatmap import HEATMAP_SKILL
from drugdevbench.prompts.skills.elisa import ELISA_SKILL


SKILL_PROMPTS: dict[str, str] = {
    "western_blot": WESTERN_BLOT_SKILL,
    "dose_response": DOSE_RESPONSE_SKILL,
    "pk_curve": PK_CURVE_SKILL,
    "flow_biaxial": FLOW_BIAXIAL_SKILL,
    "flow_histogram": FLOW_HISTOGRAM_SKILL,
    "heatmap": HEATMAP_SKILL,
    "elisa": ELISA_SKILL,
}

__all__ = [
    "SKILL_PROMPTS",
    "WESTERN_BLOT_SKILL",
    "DOSE_RESPONSE_SKILL",
    "PK_CURVE_SKILL",
    "FLOW_BIAXIAL_SKILL",
    "FLOW_HISTOGRAM_SKILL",
    "HEATMAP_SKILL",
    "ELISA_SKILL",
]
