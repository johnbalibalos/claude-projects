"""Western blot interpretation skill."""

WESTERN_BLOT_SKILL = """## Western Blot Interpretation Skill

When analyzing Western blot images, systematically evaluate:

### 1. Molecular Weight Assessment
- Locate the molecular weight ladder/markers (usually on left or right edge)
- Note the marker sizes (typically 10, 15, 25, 35, 55, 70, 100, 130, 250 kDa)
- Estimate the molecular weight of detected bands by interpolation
- Compare to expected protein size (consider post-translational modifications)

### 2. Loading Control Evaluation
- Identify the loading control (β-actin ~42 kDa, GAPDH ~37 kDa, tubulin ~55 kDa, histone H3 ~17 kDa)
- Assess consistency across lanes (should be roughly equal)
- Flag any lanes with noticeably different loading
- For membrane proteins, total protein stain (Ponceau) may be more appropriate

### 3. Band Quality Assessment
- **Sharpness**: Bands should be distinct, not fuzzy or smeared
- **Background**: Low and uniform across the blot
- **Saturation**: Bands should not be overexposed (white/blown out)
- **Migration**: Bands should be horizontal, not smiling or frowning

### 4. Specificity Evaluation
- Single band at expected MW suggests good specificity
- Multiple bands may indicate: splice variants, processing, degradation, or non-specific binding
- Lower MW bands could be degradation products
- Higher MW bands could be aggregates or cross-reactivity

### 5. Quantitative Considerations
- Only semi-quantitative without proper densitometry
- Signal must be in linear range of detection
- Compare to standards if shown
- Note if quantification is normalized to loading control

### Common Issues to Flag
- Uneven loading (variable loading control)
- High background (poor blocking or excessive antibody)
- Smearing (protein degradation or overloading)
- No signal in positive control
- Unexpected band sizes"""
