"""Pharmacokinetic curve interpretation skill."""

PK_CURVE_SKILL = """## Pharmacokinetic (PK) Curve Interpretation Skill

When analyzing PK concentration-time profiles, systematically evaluate:

### 1. Axis Identification
- **X-axis**: Time post-dose (hours, days) - usually linear scale
- **Y-axis**: Drug concentration - check for linear vs. log scale
  - Log scale: Terminal phase appears linear (first-order elimination)
  - Linear scale: Better for visualizing Cmax
- Units: ng/mL, μg/mL, μM, etc.

### 2. Key PK Parameters (Visual Estimation)
- **Cmax**: Maximum concentration (peak of the curve)
- **Tmax**: Time to reach Cmax
- **Half-life (t½)**: Time for concentration to decrease by 50%
  - On semi-log plot: slope of terminal phase
  - t½ = 0.693 / λz (where λz is elimination rate constant)
- **AUC**: Area under the curve (total exposure)
  - Proportional to the area enclosed by the curve

### 3. Route of Administration Indicators
- **IV bolus**: No absorption phase, immediate peak at t=0
- **IV infusion**: Plateau during infusion, then decline
- **Oral/SC/IM**: Absorption phase (rising), then distribution/elimination
- **Bioavailability**: Compare oral AUC to IV AUC

### 4. Pharmacokinetic Phases
- **Absorption phase**: Rising concentration (oral, SC, etc.)
- **Distribution phase**: Initial rapid decline (α phase)
- **Elimination phase**: Terminal decline (β phase)
- Multi-compartment kinetics: Multiple slopes on semi-log plot

### 5. Data Quality Assessment
- **Sampling adequacy**: Enough points to characterize each phase?
- **Error bars**: Individual variability between subjects
- **BLQ points**: Below limit of quantification (often at later timepoints)
- **Individual vs. mean**: Individual traces or mean ± error

### 6. Dose Proportionality
- Compare across multiple doses
- Linear kinetics: AUC increases proportionally with dose
- Non-linear kinetics: Disproportionate AUC increase (saturation)

### 7. Species Considerations
- Different species have different typical clearances
- Allometric scaling may apply
- Half-lives vary significantly across species

### Common Values by Species (approximate)
| Species | Typical t½ range |
|---------|-----------------|
| Mouse   | Minutes to hours |
| Rat     | Hours |
| Dog     | Hours to days |
| NHP     | Hours to days |
| Human   | Hours to weeks |"""
