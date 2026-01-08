#!/bin/bash
# Search and download OMIP papers using PMC OA Web Service API
#
# Usage:
#   ./search_download_omip.sh OMIP-001          # Download single OMIP
#   ./search_download_omip.sh OMIP-001 OMIP-050 # Download range
#   ./search_download_omip.sh --check OMIP-042  # Check if available without downloading
#
# The script:
# 1. Searches PMC for the OMIP paper by title
# 2. Uses PMC OA API to check if open access files exist
# 3. Downloads XML and PDF using wget/curl

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/papers/pmc}"
EMAIL="${EMAIL:-user@example.com}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [--check] OMIP-NNN [OMIP-MMM]"
    echo ""
    echo "Options:"
    echo "  --check     Check availability without downloading"
    echo ""
    echo "Examples:"
    echo "  $0 OMIP-042              # Download OMIP-042"
    echo "  $0 OMIP-001 OMIP-100     # Download OMIP-001 through OMIP-100"
    echo "  $0 --check OMIP-042      # Check if OMIP-042 is available"
    echo ""
    echo "Environment variables:"
    echo "  OUTPUT_DIR  Output directory (default: data/papers/pmc)"
    echo "  EMAIL       Email for NCBI API (required by NCBI)"
    exit 1
}

# Search PMC for an OMIP paper and return PMC ID
search_omip() {
    local omip_num="$1"

    # Search PMC for this specific OMIP
    local search_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    local response

    response=$(curl -s "${search_url}?db=pmc&term=OMIP-${omip_num}[Title]+AND+Cytometry[Journal]&retmode=json&email=${EMAIL}")

    # Extract PMC ID from response
    local pmc_id
    pmc_id=$(echo "$response" | grep -oP '"idlist":\s*\["\K[0-9]+' | head -1)

    if [[ -z "$pmc_id" ]]; then
        # Try alternative search pattern (OMIP + number without dash)
        response=$(curl -s "${search_url}?db=pmc&term=OMIP+${omip_num}[Title]+AND+Cytometry[Journal]&retmode=json&email=${EMAIL}")
        pmc_id=$(echo "$response" | grep -oP '"idlist":\s*\["\K[0-9]+' | head -1)
    fi

    echo "$pmc_id"
}

# Check if OMIP has open access files via PMC OA API
check_oa_availability() {
    local pmc_id="$1"

    local oa_url="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    local response

    # Request XML format from OA API
    response=$(curl -s "${oa_url}?id=PMC${pmc_id}")

    # Check if we got valid records
    if echo "$response" | grep -q "<link.*format="; then
        echo "available"
        return 0
    else
        echo "unavailable"
        return 1
    fi
}

# Download files for a PMC ID using wget
download_omip_files() {
    local pmc_id="$1"
    local omip_name="$2"

    mkdir -p "$OUTPUT_DIR"

    # Get OA API response
    local oa_url="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    local oa_response
    oa_response=$(curl -s "${oa_url}?id=PMC${pmc_id}")

    # Extract download links
    # tgz link (contains XML and PDF)
    local tgz_link
    tgz_link=$(echo "$oa_response" | grep -oP 'href="[^"]*\.tar\.gz"' | sed 's/href="//;s/"$//' | head -1)

    # PDF link
    local pdf_link
    pdf_link=$(echo "$oa_response" | grep -oP 'href="[^"]*\.pdf"' | sed 's/href="//;s/"$//' | head -1)

    local downloaded=0

    # Download tgz package (contains XML and optionally PDF)
    if [[ -n "$tgz_link" ]]; then
        echo -e "  ${GREEN}Downloading package...${NC}"
        local tgz_file="${OUTPUT_DIR}/PMC${pmc_id}.tar.gz"

        if wget -q -O "$tgz_file" "$tgz_link"; then
            # Extract XML and PDF from package
            tar -xzf "$tgz_file" -C "$OUTPUT_DIR" 2>/dev/null || true
            rm -f "$tgz_file"
            downloaded=$((downloaded + 1))
        else
            echo -e "  ${RED}Failed to download package${NC}"
        fi
    fi

    # Download XML directly via efetch
    local xml_file="${OUTPUT_DIR}/PMC${pmc_id}.xml"
    if [[ ! -f "$xml_file" ]]; then
        echo -e "  ${GREEN}Downloading XML...${NC}"
        local efetch_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        if curl -s -o "$xml_file" "${efetch_url}?db=pmc&id=${pmc_id}&rettype=xml&email=${EMAIL}"; then
            downloaded=$((downloaded + 1))
        fi
    fi

    # Download PDF directly if available and not already in package
    if [[ -n "$pdf_link" && ! -f "${OUTPUT_DIR}/PMC${pmc_id}.pdf" ]]; then
        echo -e "  ${GREEN}Downloading PDF...${NC}"
        if wget -q -O "${OUTPUT_DIR}/PMC${pmc_id}.pdf" "$pdf_link"; then
            downloaded=$((downloaded + 1))
        fi
    fi

    return $((downloaded > 0 ? 0 : 1))
}

# Main logic
main() {
    local check_only=false
    local omip_start=""
    local omip_end=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)
                check_only=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            OMIP-*|omip-*)
                local num="${1#*-}"
                num="${num#0}"  # Remove leading zeros
                if [[ -z "$omip_start" ]]; then
                    omip_start="$num"
                else
                    omip_end="$num"
                fi
                shift
                ;;
            *)
                echo "Unknown argument: $1"
                usage
                ;;
        esac
    done

    if [[ -z "$omip_start" ]]; then
        usage
    fi

    omip_end="${omip_end:-$omip_start}"

    mkdir -p "$OUTPUT_DIR"

    echo "Searching for OMIP-$(printf '%03d' $omip_start) to OMIP-$(printf '%03d' $omip_end)..."
    echo ""

    local found=0
    local downloaded=0

    for ((i=omip_start; i<=omip_end; i++)); do
        local omip_name="OMIP-$(printf '%03d' $i)"
        local omip_num="$i"

        echo -n "[$omip_name] "

        # Search for PMC ID
        local pmc_id
        pmc_id=$(search_omip "$omip_num")

        if [[ -z "$pmc_id" ]]; then
            echo -e "${YELLOW}Not found in PMC${NC}"
            continue
        fi

        echo -n "PMC${pmc_id} "
        found=$((found + 1))

        if $check_only; then
            local status
            status=$(check_oa_availability "$pmc_id")
            if [[ "$status" == "available" ]]; then
                echo -e "${GREEN}✓ Open Access available${NC}"
            else
                echo -e "${YELLOW}No OA files${NC}"
            fi
            continue
        fi

        # Check if already downloaded
        if [[ -f "${OUTPUT_DIR}/PMC${pmc_id}.xml" ]]; then
            echo -e "${GREEN}✓ Already downloaded${NC}"
            downloaded=$((downloaded + 1))
            continue
        fi

        echo ""
        if download_omip_files "$pmc_id" "$omip_name"; then
            echo -e "  ${GREEN}✓ Downloaded successfully${NC}"
            downloaded=$((downloaded + 1))
        else
            echo -e "  ${RED}✗ Download failed${NC}"
        fi

        # Rate limiting (3 requests/second max for NCBI)
        sleep 0.35
    done

    echo ""
    echo "=== Summary ==="
    echo "Found: $found papers"
    if ! $check_only; then
        echo "Downloaded: $downloaded papers"
    fi
}

main "$@"
