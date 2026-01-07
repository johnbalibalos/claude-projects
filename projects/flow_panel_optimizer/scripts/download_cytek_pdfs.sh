#!/bin/bash
# Download Cytek fluorochrome guide PDFs
# These are publicly available from Cytek's website

OUTPUT_DIR="${1:-../data/raw/cytek_pdfs}"
mkdir -p "$OUTPUT_DIR"

echo "Downloading Cytek fluorochrome guide PDFs to $OUTPUT_DIR"

# 4-laser VBYGR configuration
curl -L -o "$OUTPUT_DIR/cytek_4L_VBYGR.pdf" \
  "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20018_Rev._A_4L_VBYGR_Fluor_Guide.pdf"

# 3-laser VBR configuration
curl -L -o "$OUTPUT_DIR/cytek_3L_VBR.pdf" \
  "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20019_Rev._A_3L_VBR_Fluor_Guide.pdf"

# 2-laser BR configuration
curl -L -o "$OUTPUT_DIR/cytek_2L_BR.pdf" \
  "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20020_Rev._A_2L_BR_Fluor_Guide.pdf"

# 2-laser VB configuration
curl -L -o "$OUTPUT_DIR/cytek_2L_VB.pdf" \
  "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20021_Rev._A_2L_VB_FLuor_Guide.pdf"

echo "Download complete. Files saved to $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
