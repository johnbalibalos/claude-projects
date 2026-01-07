"""
Antibody Availability Database for Flow Cytometry Panel Design.

This module provides information about commercially available antibody-fluorophore
conjugates from major vendors (BD Biosciences, BioLegend, Thermo Fisher).

Data compiled from:
- BD Multicolor Antibody Reagents Catalog (20th Edition)
- BioLegend Product Catalog
- Thermo Fisher eBioscience Catalog

This enables realistic panel design by constraining choices to actually
purchasable reagents.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AntibodyProduct:
    """A specific antibody-fluorophore conjugate product."""
    marker: str
    fluorophore: str
    clone: str
    vendor: str
    catalog_number: Optional[str] = None
    species_reactivity: str = "human"
    format: str = "conjugated"
    notes: Optional[str] = None


# Comprehensive database of available conjugates
# Focus on human reactivity, most common clones
ANTIBODY_DATABASE: list[AntibodyProduct] = [
    # =========================================================================
    # CD3 - Pan T-cell marker
    # =========================================================================
    AntibodyProduct("CD3", "BV421", "UCHT1", "BioLegend", "300434"),
    AntibodyProduct("CD3", "BV510", "UCHT1", "BioLegend", "300448"),
    AntibodyProduct("CD3", "BV605", "UCHT1", "BioLegend", "300460"),
    AntibodyProduct("CD3", "BV650", "UCHT1", "BioLegend", "300468"),
    AntibodyProduct("CD3", "BV711", "UCHT1", "BioLegend", "300464"),
    AntibodyProduct("CD3", "BV785", "UCHT1", "BioLegend", "300472"),
    AntibodyProduct("CD3", "FITC", "UCHT1", "BioLegend", "300406"),
    AntibodyProduct("CD3", "PE", "UCHT1", "BioLegend", "300408"),
    AntibodyProduct("CD3", "PE-Cy7", "UCHT1", "BioLegend", "300420"),
    AntibodyProduct("CD3", "APC", "UCHT1", "BioLegend", "300412"),
    AntibodyProduct("CD3", "APC-Cy7", "UCHT1", "BioLegend", "300426"),
    AntibodyProduct("CD3", "Alexa Fluor 700", "UCHT1", "BioLegend", "300424"),
    AntibodyProduct("CD3", "Pacific Blue", "UCHT1", "BioLegend", "300418"),
    AntibodyProduct("CD3", "PerCP-Cy5.5", "UCHT1", "BioLegend", "300430"),
    AntibodyProduct("CD3", "BUV395", "UCHT1", "BD", "563546"),
    AntibodyProduct("CD3", "BUV496", "UCHT1", "BD", "612940"),
    AntibodyProduct("CD3", "BUV737", "UCHT1", "BD", "612750"),
    AntibodyProduct("CD3", "BB515", "UCHT1", "BD", "564465"),

    # =========================================================================
    # CD4 - Helper T-cell marker
    # =========================================================================
    AntibodyProduct("CD4", "BV421", "RPA-T4", "BioLegend", "300532"),
    AntibodyProduct("CD4", "BV510", "RPA-T4", "BioLegend", "300546"),
    AntibodyProduct("CD4", "BV605", "RPA-T4", "BioLegend", "300556"),
    AntibodyProduct("CD4", "BV650", "RPA-T4", "BioLegend", "300536"),
    AntibodyProduct("CD4", "BV711", "RPA-T4", "BioLegend", "300558"),
    AntibodyProduct("CD4", "BV750", "SK3", "BD", "747073"),
    AntibodyProduct("CD4", "BV785", "RPA-T4", "BioLegend", "300554"),
    AntibodyProduct("CD4", "FITC", "RPA-T4", "BioLegend", "300506"),
    AntibodyProduct("CD4", "PE", "RPA-T4", "BioLegend", "300508"),
    AntibodyProduct("CD4", "PE-Cy7", "RPA-T4", "BioLegend", "300512"),
    AntibodyProduct("CD4", "PE-Dazzle 594", "RPA-T4", "BioLegend", "300548"),
    AntibodyProduct("CD4", "APC", "RPA-T4", "BioLegend", "300514"),
    AntibodyProduct("CD4", "APC-Cy7", "RPA-T4", "BioLegend", "300518"),
    AntibodyProduct("CD4", "Alexa Fluor 700", "RPA-T4", "BioLegend", "300526"),
    AntibodyProduct("CD4", "PerCP-Cy5.5", "SK3", "BD", "341654"),
    AntibodyProduct("CD4", "BUV395", "SK3", "BD", "563550"),
    AntibodyProduct("CD4", "BUV496", "SK3", "BD", "612936"),
    AntibodyProduct("CD4", "BUV563", "SK3", "BD", "612912"),

    # =========================================================================
    # CD8 - Cytotoxic T-cell marker
    # =========================================================================
    AntibodyProduct("CD8", "BV421", "SK1", "BioLegend", "344748"),
    AntibodyProduct("CD8", "BV510", "SK1", "BioLegend", "344732"),
    AntibodyProduct("CD8", "BV605", "SK1", "BioLegend", "344742"),
    AntibodyProduct("CD8", "BV650", "SK1", "BioLegend", "344730"),
    AntibodyProduct("CD8", "BV711", "SK1", "BioLegend", "344734"),
    AntibodyProduct("CD8", "BV785", "SK1", "BioLegend", "344740"),
    AntibodyProduct("CD8", "FITC", "SK1", "BioLegend", "344704"),
    AntibodyProduct("CD8", "PE", "SK1", "BioLegend", "344706"),
    AntibodyProduct("CD8", "PE-Cy7", "SK1", "BioLegend", "344712"),
    AntibodyProduct("CD8", "APC", "SK1", "BioLegend", "344722"),
    AntibodyProduct("CD8", "APC-Cy7", "SK1", "BioLegend", "344714"),
    AntibodyProduct("CD8", "APC-Fire750", "SK1", "BioLegend", "344746"),
    AntibodyProduct("CD8", "Alexa Fluor 700", "SK1", "BioLegend", "344724"),
    AntibodyProduct("CD8", "PerCP-Cy5.5", "SK1", "BD", "341051"),
    AntibodyProduct("CD8", "BUV395", "SK1", "BD", "563795"),
    AntibodyProduct("CD8", "BUV496", "RPA-T8", "BD", "612942"),
    AntibodyProduct("CD8", "BUV805", "SK1", "BD", "612889"),

    # =========================================================================
    # CD19 - B-cell marker
    # =========================================================================
    AntibodyProduct("CD19", "BV421", "HIB19", "BioLegend", "302234"),
    AntibodyProduct("CD19", "BV510", "HIB19", "BioLegend", "302242"),
    AntibodyProduct("CD19", "BV605", "HIB19", "BioLegend", "302244"),
    AntibodyProduct("CD19", "BV650", "HIB19", "BioLegend", "302238"),
    AntibodyProduct("CD19", "BV711", "HIB19", "BioLegend", "302246"),
    AntibodyProduct("CD19", "BV785", "HIB19", "BioLegend", "302240"),
    AntibodyProduct("CD19", "FITC", "HIB19", "BioLegend", "302206"),
    AntibodyProduct("CD19", "PE", "HIB19", "BioLegend", "302208"),
    AntibodyProduct("CD19", "PE-Cy7", "HIB19", "BioLegend", "302216"),
    AntibodyProduct("CD19", "APC", "HIB19", "BioLegend", "302212"),
    AntibodyProduct("CD19", "APC-Cy7", "HIB19", "BioLegend", "302218"),
    AntibodyProduct("CD19", "BUV395", "SJ25C1", "BD", "563549"),
    AntibodyProduct("CD19", "BUV661", "SJ25C1", "BD", "612966"),
    AntibodyProduct("CD19", "BV480", "HIB19", "BD", "566104"),

    # =========================================================================
    # CD45RA - Naive T-cell marker
    # =========================================================================
    AntibodyProduct("CD45RA", "BV421", "HI100", "BioLegend", "304130"),
    AntibodyProduct("CD45RA", "BV510", "HI100", "BD", "563031"),
    AntibodyProduct("CD45RA", "BV570", "HI100", "BioLegend", "304132"),
    AntibodyProduct("CD45RA", "BV605", "HI100", "BioLegend", "304134"),
    AntibodyProduct("CD45RA", "BV650", "HI100", "BioLegend", "304136"),
    AntibodyProduct("CD45RA", "BV711", "HI100", "BioLegend", "304138"),
    AntibodyProduct("CD45RA", "BV785", "HI100", "BioLegend", "304140"),
    AntibodyProduct("CD45RA", "FITC", "HI100", "BioLegend", "304106"),
    AntibodyProduct("CD45RA", "PE", "HI100", "BioLegend", "304108"),
    AntibodyProduct("CD45RA", "PE-Cy7", "HI100", "BioLegend", "304126"),
    AntibodyProduct("CD45RA", "APC", "HI100", "BioLegend", "304112"),
    AntibodyProduct("CD45RA", "APC-Cy7", "HI100", "BioLegend", "304128"),
    AntibodyProduct("CD45RA", "PerCP-Cy5.5", "HI100", "BioLegend", "304122"),
    AntibodyProduct("CD45RA", "Alexa Fluor 700", "HI100", "BioLegend", "304120"),

    # =========================================================================
    # CD45RO - Memory T-cell marker
    # =========================================================================
    AntibodyProduct("CD45RO", "BV421", "UCHL1", "BioLegend", "304224"),
    AntibodyProduct("CD45RO", "BV510", "UCHL1", "BD", "563215"),
    AntibodyProduct("CD45RO", "BV605", "UCHL1", "BioLegend", "304238"),
    AntibodyProduct("CD45RO", "BV650", "UCHL1", "BD", "563752"),
    AntibodyProduct("CD45RO", "BV711", "UCHL1", "BioLegend", "304236"),
    AntibodyProduct("CD45RO", "FITC", "UCHL1", "BioLegend", "304204"),
    AntibodyProduct("CD45RO", "PE", "UCHL1", "BioLegend", "304206"),
    AntibodyProduct("CD45RO", "PE-Cy7", "UCHL1", "BioLegend", "304230"),
    AntibodyProduct("CD45RO", "APC", "UCHL1", "BioLegend", "304210"),
    AntibodyProduct("CD45RO", "PerCP-Cy5.5", "UCHL1", "BioLegend", "304222"),

    # =========================================================================
    # CD25 - IL-2R alpha / Treg marker
    # =========================================================================
    AntibodyProduct("CD25", "BV421", "M-A251", "BD", "562442"),
    AntibodyProduct("CD25", "BV510", "M-A251", "BD", "740198"),
    AntibodyProduct("CD25", "BV605", "M-A251", "BD", "562660"),
    AntibodyProduct("CD25", "BV650", "M-A251", "BD", "563756"),
    AntibodyProduct("CD25", "BV711", "M-A251", "BD", "563159"),
    AntibodyProduct("CD25", "BV785", "M-A251", "BD", "741035"),
    AntibodyProduct("CD25", "FITC", "M-A251", "BD", "555431"),
    AntibodyProduct("CD25", "PE", "M-A251", "BD", "555432"),
    AntibodyProduct("CD25", "PE-Cy7", "M-A251", "BD", "557741"),
    AntibodyProduct("CD25", "PE-Dazzle 594", "M-A251", "BioLegend", "356138"),
    AntibodyProduct("CD25", "APC", "M-A251", "BD", "555434"),
    AntibodyProduct("CD25", "APC-Cy7", "M-A251", "BD", "561782"),
    AntibodyProduct("CD25", "BB515", "M-A251", "BD", "564467"),
    AntibodyProduct("CD25", "BUV395", "2A3", "BD", "564034"),

    # =========================================================================
    # CD127 - IL-7R / Treg exclusion marker
    # =========================================================================
    AntibodyProduct("CD127", "BV421", "A019D5", "BioLegend", "351310"),
    AntibodyProduct("CD127", "BV510", "A019D5", "BioLegend", "351332"),
    AntibodyProduct("CD127", "BV605", "A019D5", "BioLegend", "351334"),
    AntibodyProduct("CD127", "BV711", "A019D5", "BioLegend", "351328"),
    AntibodyProduct("CD127", "BV785", "A019D5", "BioLegend", "351330"),
    AntibodyProduct("CD127", "FITC", "A019D5", "BioLegend", "351312"),
    AntibodyProduct("CD127", "PE", "HIL-7R-M21", "BD", "557938"),
    AntibodyProduct("CD127", "PE-Cy7", "A019D5", "BioLegend", "351320"),
    AntibodyProduct("CD127", "PE-Dazzle 594", "A019D5", "BioLegend", "351336"),
    AntibodyProduct("CD127", "APC", "A019D5", "BioLegend", "351316"),
    AntibodyProduct("CD127", "PerCP-Cy5.5", "A019D5", "BioLegend", "351324"),
    AntibodyProduct("CD127", "Alexa Fluor 700", "A019D5", "BioLegend", "351344"),

    # =========================================================================
    # CCR7 - Central memory marker
    # =========================================================================
    AntibodyProduct("CCR7", "BV421", "G043H7", "BioLegend", "353208"),
    AntibodyProduct("CCR7", "BV510", "G043H7", "BioLegend", "353232"),
    AntibodyProduct("CCR7", "BV605", "G043H7", "BioLegend", "353224"),
    AntibodyProduct("CCR7", "BV650", "G043H7", "BioLegend", "353234"),
    AntibodyProduct("CCR7", "BV711", "G043H7", "BioLegend", "353228"),
    AntibodyProduct("CCR7", "BV785", "G043H7", "BioLegend", "353230"),
    AntibodyProduct("CCR7", "FITC", "G043H7", "BioLegend", "353216"),
    AntibodyProduct("CCR7", "PE", "G043H7", "BioLegend", "353204"),
    AntibodyProduct("CCR7", "PE-Cy5", "G043H7", "BioLegend", "353210"),
    AntibodyProduct("CCR7", "PE-Cy7", "G043H7", "BioLegend", "353226"),
    AntibodyProduct("CCR7", "PE-Dazzle 594", "G043H7", "BioLegend", "353236"),
    AntibodyProduct("CCR7", "APC", "G043H7", "BioLegend", "353214"),
    AntibodyProduct("CCR7", "APC-Cy7", "G043H7", "BioLegend", "353212"),
    AntibodyProduct("CCR7", "PerCP-Cy5.5", "G043H7", "BioLegend", "353220"),
    AntibodyProduct("CCR7", "Alexa Fluor 700", "G043H7", "BioLegend", "353238"),

    # =========================================================================
    # CD27 - Memory/costimulation marker
    # =========================================================================
    AntibodyProduct("CD27", "BV421", "O323", "BioLegend", "302824"),
    AntibodyProduct("CD27", "BV510", "O323", "BioLegend", "302836"),
    AntibodyProduct("CD27", "BV605", "O323", "BioLegend", "302830"),
    AntibodyProduct("CD27", "BV650", "O323", "BioLegend", "302828"),
    AntibodyProduct("CD27", "BV711", "O323", "BioLegend", "302834"),
    AntibodyProduct("CD27", "BV785", "O323", "BioLegend", "302832"),
    AntibodyProduct("CD27", "FITC", "O323", "BioLegend", "302806"),
    AntibodyProduct("CD27", "PE", "O323", "BioLegend", "302808"),
    AntibodyProduct("CD27", "PE-Cy7", "O323", "BioLegend", "302838"),
    AntibodyProduct("CD27", "APC", "O323", "BioLegend", "302810"),
    AntibodyProduct("CD27", "APC-Cy7", "O323", "BioLegend", "302816"),
    AntibodyProduct("CD27", "PerCP-Cy5.5", "O323", "BioLegend", "302820"),
    AntibodyProduct("CD27", "Alexa Fluor 700", "O323", "BioLegend", "302814"),

    # =========================================================================
    # CD56 - NK cell marker
    # =========================================================================
    AntibodyProduct("CD56", "BV421", "HCD56", "BioLegend", "318328"),
    AntibodyProduct("CD56", "BV510", "HCD56", "BioLegend", "318340"),
    AntibodyProduct("CD56", "BV605", "HCD56", "BioLegend", "318334"),
    AntibodyProduct("CD56", "BV650", "HCD56", "BioLegend", "318344"),
    AntibodyProduct("CD56", "BV711", "HCD56", "BioLegend", "318336"),
    AntibodyProduct("CD56", "BV785", "HCD56", "BioLegend", "318346"),
    AntibodyProduct("CD56", "FITC", "HCD56", "BioLegend", "318304"),
    AntibodyProduct("CD56", "PE", "HCD56", "BioLegend", "318306"),
    AntibodyProduct("CD56", "PE-Cy7", "HCD56", "BioLegend", "318318"),
    AntibodyProduct("CD56", "APC", "HCD56", "BioLegend", "318310"),
    AntibodyProduct("CD56", "APC-Cy7", "HCD56", "BioLegend", "318332"),
    AntibodyProduct("CD56", "PerCP-Cy5.5", "HCD56", "BioLegend", "318322"),
    AntibodyProduct("CD56", "BUV395", "NCAM16.2", "BD", "563554"),
    AntibodyProduct("CD56", "BUV737", "NCAM16.2", "BD", "612767"),

    # =========================================================================
    # CD14 - Monocyte marker
    # =========================================================================
    AntibodyProduct("CD14", "BV421", "M5E2", "BioLegend", "301830"),
    AntibodyProduct("CD14", "BV510", "M5E2", "BioLegend", "301842"),
    AntibodyProduct("CD14", "BV605", "M5E2", "BioLegend", "301834"),
    AntibodyProduct("CD14", "BV650", "M5E2", "BD", "563419"),
    AntibodyProduct("CD14", "BV711", "M5E2", "BioLegend", "301838"),
    AntibodyProduct("CD14", "BV785", "M5E2", "BioLegend", "301840"),
    AntibodyProduct("CD14", "FITC", "M5E2", "BioLegend", "301804"),
    AntibodyProduct("CD14", "PE", "M5E2", "BioLegend", "301806"),
    AntibodyProduct("CD14", "PE-Cy7", "M5E2", "BioLegend", "301814"),
    AntibodyProduct("CD14", "APC", "M5E2", "BioLegend", "301808"),
    AntibodyProduct("CD14", "APC-Cy7", "M5E2", "BioLegend", "301820"),
    AntibodyProduct("CD14", "PerCP-Cy5.5", "M5E2", "BioLegend", "301824"),
    AntibodyProduct("CD14", "BUV395", "M5E2", "BD", "563561"),
    AntibodyProduct("CD14", "BUV615", "M5E2", "BD", "751476"),
    AntibodyProduct("CD14", "BUV737", "M5E2", "BD", "612763"),

    # =========================================================================
    # CD16 - Fc receptor / NK / monocyte subset
    # =========================================================================
    AntibodyProduct("CD16", "BV421", "3G8", "BioLegend", "302038"),
    AntibodyProduct("CD16", "BV510", "3G8", "BioLegend", "302048"),
    AntibodyProduct("CD16", "BV605", "3G8", "BioLegend", "302040"),
    AntibodyProduct("CD16", "BV650", "3G8", "BioLegend", "302042"),
    AntibodyProduct("CD16", "BV711", "3G8", "BioLegend", "302044"),
    AntibodyProduct("CD16", "BV785", "3G8", "BioLegend", "302046"),
    AntibodyProduct("CD16", "FITC", "3G8", "BioLegend", "302006"),
    AntibodyProduct("CD16", "PE", "3G8", "BioLegend", "302008"),
    AntibodyProduct("CD16", "PE-Cy7", "3G8", "BioLegend", "302016"),
    AntibodyProduct("CD16", "APC", "3G8", "BioLegend", "302012"),
    AntibodyProduct("CD16", "APC-Cy7", "3G8", "BioLegend", "302018"),
    AntibodyProduct("CD16", "PerCP-Cy5.5", "3G8", "BioLegend", "302028"),
    AntibodyProduct("CD16", "Alexa Fluor 700", "3G8", "BioLegend", "302026"),

    # =========================================================================
    # HLA-DR - MHC Class II / APC marker
    # =========================================================================
    AntibodyProduct("HLA-DR", "BV421", "L243", "BioLegend", "307636"),
    AntibodyProduct("HLA-DR", "BV510", "L243", "BioLegend", "307646"),
    AntibodyProduct("HLA-DR", "BV605", "L243", "BioLegend", "307640"),
    AntibodyProduct("HLA-DR", "BV650", "L243", "BioLegend", "307650"),
    AntibodyProduct("HLA-DR", "BV711", "L243", "BioLegend", "307644"),
    AntibodyProduct("HLA-DR", "BV785", "L243", "BioLegend", "307642"),
    AntibodyProduct("HLA-DR", "FITC", "L243", "BioLegend", "307604"),
    AntibodyProduct("HLA-DR", "PE", "L243", "BioLegend", "307606"),
    AntibodyProduct("HLA-DR", "PE-Cy7", "L243", "BioLegend", "307616"),
    AntibodyProduct("HLA-DR", "APC", "L243", "BioLegend", "307610"),
    AntibodyProduct("HLA-DR", "APC-Cy7", "L243", "BioLegend", "307618"),
    AntibodyProduct("HLA-DR", "PerCP-Cy5.5", "L243", "BioLegend", "307630"),
    AntibodyProduct("HLA-DR", "Alexa Fluor 700", "L243", "BioLegend", "307626"),

    # =========================================================================
    # CXCR3 - Th1 chemokine receptor
    # =========================================================================
    AntibodyProduct("CXCR3", "BV421", "G025H7", "BioLegend", "353716"),
    AntibodyProduct("CXCR3", "BV510", "G025H7", "BioLegend", "353726"),
    AntibodyProduct("CXCR3", "BV605", "G025H7", "BioLegend", "353728"),
    AntibodyProduct("CXCR3", "BV711", "G025H7", "BioLegend", "353730"),
    AntibodyProduct("CXCR3", "FITC", "G025H7", "BioLegend", "353704"),
    AntibodyProduct("CXCR3", "PE", "G025H7", "BioLegend", "353706"),
    AntibodyProduct("CXCR3", "PE-Cy7", "G025H7", "BioLegend", "353714"),
    AntibodyProduct("CXCR3", "APC", "G025H7", "BioLegend", "353708"),
    AntibodyProduct("CXCR3", "APC-Cy7", "G025H7", "BioLegend", "353712"),
    AntibodyProduct("CXCR3", "PerCP-Cy5.5", "G025H7", "BioLegend", "353720"),

    # =========================================================================
    # CD161 - NKT / MAIT marker
    # =========================================================================
    AntibodyProduct("CD161", "BV421", "HP-3G10", "BioLegend", "339914"),
    AntibodyProduct("CD161", "BV510", "HP-3G10", "BioLegend", "339924"),
    AntibodyProduct("CD161", "BV605", "HP-3G10", "BioLegend", "339918"),
    AntibodyProduct("CD161", "BV650", "HP-3G10", "BioLegend", "339930"),
    AntibodyProduct("CD161", "BV711", "HP-3G10", "BioLegend", "339928"),
    AntibodyProduct("CD161", "BV785", "HP-3G10", "BioLegend", "339932"),
    AntibodyProduct("CD161", "FITC", "HP-3G10", "BioLegend", "339906"),
    AntibodyProduct("CD161", "PE", "HP-3G10", "BioLegend", "339904"),
    AntibodyProduct("CD161", "PE-Cy7", "HP-3G10", "BioLegend", "339910"),
    AntibodyProduct("CD161", "APC", "HP-3G10", "BioLegend", "339912"),
    AntibodyProduct("CD161", "PerCP-Cy5.5", "HP-3G10", "BioLegend", "339920"),

    # =========================================================================
    # CD38 - Activation / plasmablast marker
    # =========================================================================
    AntibodyProduct("CD38", "BV421", "HIT2", "BioLegend", "303526"),
    AntibodyProduct("CD38", "BV510", "HIT2", "BioLegend", "303532"),
    AntibodyProduct("CD38", "BV605", "HIT2", "BioLegend", "303534"),
    AntibodyProduct("CD38", "BV650", "HIT2", "BioLegend", "303536"),
    AntibodyProduct("CD38", "BV711", "HIT2", "BioLegend", "303530"),
    AntibodyProduct("CD38", "BV785", "HIT2", "BioLegend", "303538"),
    AntibodyProduct("CD38", "FITC", "HIT2", "BioLegend", "303504"),
    AntibodyProduct("CD38", "PE", "HIT2", "BioLegend", "303506"),
    AntibodyProduct("CD38", "PE-Cy7", "HIT2", "BioLegend", "303516"),
    AntibodyProduct("CD38", "APC", "HIT2", "BioLegend", "303510"),
    AntibodyProduct("CD38", "APC-Cy7", "HIT2", "BioLegend", "303518"),
    AntibodyProduct("CD38", "APC-Fire750", "HB-7", "BioLegend", "356638"),
    AntibodyProduct("CD38", "PerCP-Cy5.5", "HIT2", "BioLegend", "303522"),
    AntibodyProduct("CD38", "BB515", "HIT2", "BD", "564499"),

    # =========================================================================
    # IgD - Naive B-cell marker
    # =========================================================================
    AntibodyProduct("IgD", "BV421", "IA6-2", "BioLegend", "348226"),
    AntibodyProduct("IgD", "BV510", "IA6-2", "BioLegend", "348220"),
    AntibodyProduct("IgD", "BV605", "IA6-2", "BioLegend", "348232"),
    AntibodyProduct("IgD", "BV711", "IA6-2", "BioLegend", "348240"),
    AntibodyProduct("IgD", "FITC", "IA6-2", "BioLegend", "348206"),
    AntibodyProduct("IgD", "PE", "IA6-2", "BioLegend", "348204"),
    AntibodyProduct("IgD", "PE-Cy7", "IA6-2", "BioLegend", "348210"),
    AntibodyProduct("IgD", "APC", "IA6-2", "BioLegend", "348222"),
    AntibodyProduct("IgD", "PerCP-Cy5.5", "IA6-2", "BioLegend", "348208"),

    # =========================================================================
    # CD20 - B-cell marker
    # =========================================================================
    AntibodyProduct("CD20", "BV421", "2H7", "BioLegend", "302330"),
    AntibodyProduct("CD20", "BV510", "2H7", "BioLegend", "302340"),
    AntibodyProduct("CD20", "BV605", "2H7", "BioLegend", "302334"),
    AntibodyProduct("CD20", "BV650", "2H7", "BioLegend", "302336"),
    AntibodyProduct("CD20", "BV711", "2H7", "BioLegend", "302342"),
    AntibodyProduct("CD20", "BV785", "2H7", "BioLegend", "302344"),
    AntibodyProduct("CD20", "FITC", "2H7", "BioLegend", "302304"),
    AntibodyProduct("CD20", "PE", "2H7", "BioLegend", "302306"),
    AntibodyProduct("CD20", "PE-Cy7", "2H7", "BioLegend", "302312"),
    AntibodyProduct("CD20", "APC", "2H7", "BioLegend", "302310"),
    AntibodyProduct("CD20", "APC-Cy7", "2H7", "BioLegend", "302314"),
    AntibodyProduct("CD20", "APC-R700", "2H7", "BD", "564418"),
    AntibodyProduct("CD20", "PerCP-Cy5.5", "2H7", "BioLegend", "302326"),
]


def check_availability(marker: str, fluorophore: str) -> dict:
    """
    Check if a marker-fluorophore combination is commercially available.

    Args:
        marker: Target antigen (e.g., "CD4")
        fluorophore: Fluorochrome (e.g., "BV421")

    Returns:
        dict with availability info and product details
    """
    matches = [
        p for p in ANTIBODY_DATABASE
        if p.marker.lower() == marker.lower()
        and p.fluorophore.lower() == fluorophore.lower()
    ]

    if matches:
        p = matches[0]  # Take first match
        return {
            "available": True,
            "marker": p.marker,
            "fluorophore": p.fluorophore,
            "clone": p.clone,
            "vendor": p.vendor,
            "catalog_number": p.catalog_number,
            "alternatives": [
                {"vendor": m.vendor, "clone": m.clone, "catalog": m.catalog_number}
                for m in matches[1:]
            ] if len(matches) > 1 else []
        }
    else:
        # Find what IS available for this marker
        marker_options = get_available_fluorophores(marker)
        # Find what IS available for this fluorophore
        fluor_options = get_available_markers(fluorophore)

        return {
            "available": False,
            "marker": marker,
            "fluorophore": fluorophore,
            "message": f"No {marker}-{fluorophore} conjugate found in catalog",
            "available_fluorophores_for_marker": marker_options[:10],
            "available_markers_for_fluorophore": fluor_options[:10],
        }


def get_available_fluorophores(marker: str) -> list[str]:
    """Get all available fluorophore conjugates for a marker."""
    return list(set(
        p.fluorophore for p in ANTIBODY_DATABASE
        if p.marker.lower() == marker.lower()
    ))


def get_available_markers(fluorophore: str) -> list[str]:
    """Get all markers available in a specific fluorophore."""
    return list(set(
        p.marker for p in ANTIBODY_DATABASE
        if p.fluorophore.lower() == fluorophore.lower()
    ))


def validate_panel_availability(assignments: list[dict]) -> dict:
    """
    Validate that all panel assignments are commercially available.

    Args:
        assignments: List of {"marker": X, "fluorophore": Y} dicts

    Returns:
        dict with validation results
    """
    available = []
    unavailable = []

    for a in assignments:
        result = check_availability(a["marker"], a["fluorophore"])
        if result["available"]:
            available.append({
                "marker": a["marker"],
                "fluorophore": a["fluorophore"],
                "vendor": result["vendor"],
                "clone": result["clone"],
                "catalog": result["catalog_number"],
            })
        else:
            unavailable.append({
                "marker": a["marker"],
                "fluorophore": a["fluorophore"],
                "alternatives": result.get("available_fluorophores_for_marker", []),
            })

    return {
        "total": len(assignments),
        "available": len(available),
        "unavailable": len(unavailable),
        "availability_rate": len(available) / len(assignments) if assignments else 0,
        "available_products": available,
        "unavailable_assignments": unavailable,
        "ready_to_order": len(unavailable) == 0,
    }


def suggest_available_fluorophores(
    marker: str,
    existing_panel: list[str],
    expression_level: str = "medium"
) -> list[dict]:
    """
    Suggest available fluorophores for a marker that are compatible with panel.

    Args:
        marker: Target antigen
        existing_panel: Currently selected fluorophores
        expression_level: "high", "medium", or "low"

    Returns:
        List of available fluorophores ranked by compatibility
    """
    from flow_panel_optimizer.mcp.server import check_compatibility, get_fluorophore

    available = get_available_fluorophores(marker)

    if not available:
        return []

    suggestions = []
    for fluor in available:
        # Check spectral compatibility
        compat = check_compatibility(fluor, existing_panel)
        if not compat.get("compatible", True):
            continue

        fluor_info = get_fluorophore(fluor)
        brightness = fluor_info.relative_brightness if fluor_info else 50

        # Score based on compatibility and brightness match
        score = (1 - compat.get("max_similarity", 0)) * 0.7

        # Adjust for expression level
        if expression_level == "low" and brightness >= 70:
            score += 0.3
        elif expression_level == "high" and brightness <= 60:
            score += 0.2
        elif expression_level == "medium" and 40 <= brightness <= 80:
            score += 0.2

        # Get product info
        products = [p for p in ANTIBODY_DATABASE
                   if p.marker.lower() == marker.lower()
                   and p.fluorophore.lower() == fluor.lower()]

        suggestions.append({
            "fluorophore": fluor,
            "score": round(score, 3),
            "max_similarity": compat.get("max_similarity", 0),
            "recommendation": compat.get("recommendation", "UNKNOWN"),
            "brightness": brightness,
            "vendor": products[0].vendor if products else "Unknown",
            "clone": products[0].clone if products else "Unknown",
            "catalog": products[0].catalog_number if products else None,
        })

    # Sort by score
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return suggestions


if __name__ == "__main__":
    import json

    print("Testing Antibody Availability Database\n")

    # Test availability check
    print("1. Check CD4-BV421:")
    result = check_availability("CD4", "BV421")
    print(json.dumps(result, indent=2))

    print("\n2. Check CD4-BV999 (doesn't exist):")
    result = check_availability("CD4", "BV999")
    print(json.dumps(result, indent=2))

    print("\n3. Validate a panel:")
    panel = [
        {"marker": "CD3", "fluorophore": "BV421"},
        {"marker": "CD4", "fluorophore": "PE"},
        {"marker": "CD8", "fluorophore": "APC"},
        {"marker": "FakeMarker", "fluorophore": "FITC"},
    ]
    result = validate_panel_availability(panel)
    print(json.dumps(result, indent=2))

    print(f"\nDatabase contains {len(ANTIBODY_DATABASE)} products")
    print(f"Unique markers: {len(set(p.marker for p in ANTIBODY_DATABASE))}")
