"""Domain expert personas for figure interpretation."""

from drugdevbench.data.schemas import Persona
from drugdevbench.prompts.personas.immunologist import IMMUNOLOGIST_PROMPT
from drugdevbench.prompts.personas.pharmacologist import PHARMACOLOGIST_PROMPT
from drugdevbench.prompts.personas.bioanalytical import BIOANALYTICAL_SCIENTIST_PROMPT
from drugdevbench.prompts.personas.molecular_bio import MOLECULAR_BIOLOGIST_PROMPT
from drugdevbench.prompts.personas.computational_bio import COMPUTATIONAL_BIOLOGIST_PROMPT
from drugdevbench.prompts.personas.cell_biologist import CELL_BIOLOGIST_PROMPT


PERSONA_PROMPTS: dict[Persona, str] = {
    Persona.IMMUNOLOGIST: IMMUNOLOGIST_PROMPT,
    Persona.PHARMACOLOGIST: PHARMACOLOGIST_PROMPT,
    Persona.BIOANALYTICAL_SCIENTIST: BIOANALYTICAL_SCIENTIST_PROMPT,
    Persona.MOLECULAR_BIOLOGIST: MOLECULAR_BIOLOGIST_PROMPT,
    Persona.COMPUTATIONAL_BIOLOGIST: COMPUTATIONAL_BIOLOGIST_PROMPT,
    Persona.CELL_BIOLOGIST: CELL_BIOLOGIST_PROMPT,
}

__all__ = [
    "PERSONA_PROMPTS",
    "IMMUNOLOGIST_PROMPT",
    "PHARMACOLOGIST_PROMPT",
    "BIOANALYTICAL_SCIENTIST_PROMPT",
    "MOLECULAR_BIOLOGIST_PROMPT",
    "COMPUTATIONAL_BIOLOGIST_PROMPT",
    "CELL_BIOLOGIST_PROMPT",
]
