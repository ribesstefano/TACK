import re
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _base_row(row: pd.Series) -> Dict[str, Any]:
    """Convert base row to dict."""
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _normalize_unit(unit: str) -> str:
    if not unit:
        return ""
    unit = unit.replace("μ", "u").replace("µ", "u")
    return unit.strip()


def _parse_value_with_operator(value_str: str) -> tuple[float | None, str]:
    """
    Extract numeric value and operator from string like ">85%", "~100", "≥10".
    
    Returns:
        (value, operator) where operator is one of: '>', '<', '~', '≥', '≤', ''
    """
    if not value_str:
        return None, ""
    
    value_str = value_str.strip()
    
    # Match operator + value
    match = re.match(r'^([>≥<≤~]+)\s*([\d.]+)', value_str)
    if match:
        operator = match.group(1)
        try:
            value = float(match.group(2))
            return value, operator
        except ValueError:
            return None, ""
    
    # No operator, just value
    match = re.match(r'^([\d.]+)', value_str)
    if match:
        try:
            value = float(match.group(1))
            return value, ""
        except ValueError:
            return None, ""
    
    return None, ""


def _make_numeric_row(
    row: pd.Series,
    value: float,
    value_type: str,
    unit: str,
    category: str = "numeric",
    operator: str = "",
    cell_line: str | None = None,
    poi_name: str | None = None,
    value_error: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Helper to build a row dict with numeric value."""
    out = _base_row(row)
    out["Value"] = float(value)
    out["Value_Mean"] = float(value)
    out["Value_Type"] = value_type
    out["Value_Unit"] = _normalize_unit(unit)
    out["Value_Category"] = category
    if operator:
        out["Value_Operator"] = operator
        out["Value_Symbol"] = operator
    if value_error is not None:
        out["Value_Error"] = float(value_error)
    if cell_line is not None:
        out["Cell_Line"] = cell_line
    if poi_name is not None:
        out["POI_Name"] = poi_name
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Override table
# ---------------------------------------------------------------------------

def get_manual_curation_overrides() -> Dict[str, Dict[str, Any]]:
    """
    Manual curation overrides for specific entries that need special handling.

    Pattern is matched against the free-text comment; handler returns
    a list of additional row dicts extracted from that comment.
    """
    overrides: Dict[str, Dict[str, Any]] = {
        # Pre-built handlers
        "pDC50_Brd4_pattern": {
            "pattern": r"pDC50\s+for\s+Brd4\s+short/Brd4\s+long/Brd3/Brd2",
            "handler": "parse_multi_protein_degradation",
        },
        "Dmax_BBL358_T47D": {
            "pattern": r"Dmax\s+in\s+BBL358/T47D",
            "handler": "parse_dual_cell_line_dmax",
        },
        "Dmax_at_concentration": {
            "pattern": r"DCmax\s+was\s+measured\s+(?:in|at)\s+(\d+)\s*([nμu]M)",
            "handler": "parse_concentration_dependent_dmax",
        },

        # Very simple cleanups
        "NaN_comment": {
            "pattern": r"^\s*nan\s*$",
            "handler": "parse_nan_comment",
        },

        # DC50 ranges like:
        #  - DC50 is between 0.1uM-1uM
        #  - DC50 is 1-3nM
        #  - DC50 is 10-30nM
        "DC50_range_generic": {
            "pattern": r"DC50\s+is\s+(?:between\s+)?[\d.]+\s*",
            "handler": "parse_dc50_range",
        },

        # Bound on DC50 for CDK4:
        #  - DC50 for CDK4 > 100 nM
        #  - DC50 for CDK4 > 50 nM
        #  - DC50 for CDK4 > 500 nM
        "DC50_CDK4_bound": {
            "pattern": r"DC50\s+for\s+CDK4\s*>\s*[\d.]+\s*nM",
            "handler": "parse_cdk4_dc50_bound",
        },

        # Simple two-values DC50 without explicit cell line:
        #  - DC50 is 1.76 nM and 4.5 nM
        "DC50_two_values_no_cell": {
            "pattern": r"DC50\s+is\s+[\d.]+\s*nM\s+and\s+[\d.]+\s*nM",
            "handler": "parse_dc50_two_values_no_cell",
        },

        # Ligand dual IC50:
        #  - Ligand IC50 is 0.26 nM (Kd), 4.0 nM (kinase activity)
        "Ligand_dual_IC50_modes": {
            "pattern": r"Ligand\s+IC50\s+is\s+[\d.]+\s*nM\s*\([^)]*\),\s*[\d.]+\s*nM\s*\([^)]*\)",
            "handler": "parse_ligand_dual_ic50_modes",
        },

        # Ligand IC50 ranges:
        #  - IC50 of the ligand is between 2nM and 7nM
        #  - IC50 of the ligand is between 1nM-34 nM
        "Ligand_IC50_range_generic": {
            "pattern": r"IC50\s+of\s+the\s+ligand\s+is\s+between\s+",
            "handler": "parse_ligand_ic50_range",
        },

        # Ligand EC50 ranges:
        #  - EC50 of ligand is between 0.044uM-1.06 uM
        "Ligand_EC50_range_generic": {
            "pattern": r"EC50\s+of\s+ligand\s+is\s+between\s+",
            "handler": "parse_ligand_ec50_range",
        },

        # BTK ligand: IC50 + EC50
        "BTK_ligand_IC50_EC50": {
            "pattern": r"IC50\s+of\s+ligand\s+is\s+4\s*nM\s*\(BTK\s+inhibition\)",
            "handler": "parse_btk_ligand_ic50_ec50",
        },

        # BTK ligand variants (WT / C481S):
        "BTK_ligand_IC50_variants": {
            "pattern": r"IC50\s+of\s+ligand\s+is\s+51\.0\s*nM\s*\(WT\s+BTK\),\s*30\.7\s*\(C481S\)",
            "handler": "parse_btk_ligand_ic50_variants",
        },

        # Cytotoxicity IC50 (CC50)
        "Cytotoxicity_CC50": {
            "pattern": r"Cytotoxicity\s+IC50\s*\(CC50\)",
            "handler": "parse_cytotoxicity_ic50",
        },

        # Generic AKT panel (IC50 and EC50 ranges and per-isoform EC50)
        "AKT_panel_ligand_PROTAC": {
            "pattern": r"IC50\s+of\s+ligand\s+is\s+AKT1",
            "handler": "parse_akt_panel_metrics",
        },

        # HDAC1/2/3 Dmax panel:
        "HDAC123_Dmax_panel": {
            "pattern": r"For\s+HDAC1/HDAC2/HDAC3,\s+the\s+Dmax",
            "handler": "parse_hdac_panel_dmax",
        },

        # BCLXL selectivity (no numeric extraction, but kept as no-op handler)
        "BCLXL_selectivity": {
            "pattern": r"Selective\s+BCLXL\s+degradation",
            "handler": "parse_bclxl_selectivity",
        },

        # PMID 24068666 block: DCmax range + DC50 range
        "PMID24068666_DCmax_DC50_ranges": {
            "pattern": r"Ligand\s+name\s+is\s+from\s+PMID\s+24068666",
            "handler": "parse_pmid24068666_dcmax_dc50_ranges",
        },

        # CDK2 / CDK9 DC50 panel:
        "CDK2_CDK9_DC50_panel": {
            "pattern": r"DC50\s+is\s+CDK2:\s*[\d.]+\s*nM,\s*CDK9:\s*[\d.]+\s*nM",
            "handler": "parse_cdk_panel_dc50",
        },

        # SMARCA2/4 simple DC50 + DCmax:
        "SMARCA2_SMARCA4_DC50_DCmax_simple": {
            "pattern": r"SMARCA2\s+300nM,\s*SMARCA4\s+250nM",
            "handler": "parse_smarca_dc50_dcmax_simple",
        },

        # SMARCA2/4/PBRM1 multi-cell DC50:
        "SMARCA_PBRM1_DC50_two_cells": {
            "pattern": r"DC50\s+is\s*\(SMARCA2\s+[\d.]+nM,\s*SMARCA4\s+[\d.]+nM,\s*PBRM1",
            "handler": "parse_smarca_pbrm1_dc50_two_cells",
        },

        # Multi-cell Dmax: Karpas 422, ULA, etc.
        "Multi_cell_Dmax_lymphoma": {
            "pattern": r"Dmax\s+is\s+76%\s+Karpas\s+422",
            "handler": "parse_multi_cell_dmax_only",
        },

        # PDEdelta / deltasonamide block:
        "PDEdelta_deltasonamide_metrics": {
            "pattern": r"PDEdelta\s+degradation",
            "handler": "parse_pdedelta_metrics",
        },

        # HeLa vs DLBCL DC50 / DMAX:
        "HeLa_vs_DLBCL_DC50_Dmax": {
            "pattern": r"DC50\s+value\s+is\s+for\s+HeLa\s+cells",
            "handler": "parse_hela_vs_dlbcl_dc50_dmax",
        },

        # BTK multi-cell DC50:
        "BTK_multi_cell_DC50": {
            "pattern": r"DC50\s+is\s+6\.3\s*nM\s*\(HBL1\)",
            "handler": "parse_btk_multi_cell_dc50",
        },

        # SU-DHL-1 / NCI-H2228 DC50 with error:
        "SU_DHL1_NCI_H2228_DC50_block": {
            "pattern": r"DC50\s+is\s+SU-DHL-1:\s*[\d.]+\s*±\s*[\d.]+\s*nM,\s*NCI-H2228:\s*[\d.]+\s*±\s*[\d.]+\s*nM",
            "handler": "parse_two_cell_dc50_with_error",
        },

        # KYSE520 / MV4;11 DC50 / EC50 / Dmax:
        "KYSE520_MV411_DC50_EC50_Dmax": {
            "pattern": r"DC50\s+for\s+KYSE520\s+cell:\s*[\d.]+nM;DC50\s+for\s+MV4;11\s+cell",
            "handler": "parse_kyse520_mv411_block",
        },

        # LNCaP / VCaP / 22Rv1 DC50 & EC50:
        "LNCaP_VCaP_22Rv1_EC50_DC50": {
            "pattern": r"EC50\s+of\s+ligand\s+is\s+LNCaP",
            "handler": "parse_lncap_vcap_22rv1_ec50_dc50",
        },

        # PROTAC2 multi-cell EC50 (slash format):
        "PROTAC2_multicell_EC50_slash": {
            "pattern": r"EC50\s+of\s+PROTAC2:",
            "handler": "parse_multicell_ec50_slash_format",
        },

        # PC9/HCC827/H1975 EC50 (comma format):
        "PC9_HCC827_H1975_EC50": {
            "pattern": r"for\s+PC9/HCC827/H1975\s+cells",
            "handler": "parse_multicell_ec50_comma_format",
        },

        # SU-DHL-1/H3122/A549 EC50 (slash format):
        "SU_DHL1_H3122_A549_EC50": {
            "pattern": r"for\s+SU-DHL-1/H3122/A549\s+cells",
            "handler": "parse_multicell_ec50_slash_format",
        },

        # Generic "in cell X, in cell Y" EC50 for PROTAC:
        "PROTAC_EC50_in_cells_simple": {
            "pattern": r"EC50\s+of\s+PROTAC\s+is\s+[\d.]+[nuμ]M?\s+in\s+",
            "handler": "parse_ec50_in_cells_simple",
        },

        # Multi-cell EC50 colon format:
        #   EC50 of PROTAC is MCF-7: 2.70 ± 0.19 ...
        "PROTAC_multicell_EC50_colon_format": {
            "pattern": r"EC50\s+of\s+PROTAC\s+is\s+MCF-7:",
            "handler": "parse_multicell_ec50_colon_format",
        },

        # EC50 of ligand and PROTAC in SF / H2228:
        "SF_H2228_EC50_ligand_PROTAC": {
            "pattern": r"EC50\s+of\s+ligand\s+is\s+2\.7\s*±",
            "handler": "parse_sf_h2228_ec50_ligand_protac",
        },

        # EGFR Ba/F3 mutant vs WT:
        "EGFR_BaF3_EC50_mutant_vs_WT": {
            "pattern": r"EGFR\s+Ba/F3\s+cells",
            "handler": "parse_egfr_baf3_ec50_mutant_wt",
        },

        # PROTAC EC50 range + ligand EC50 range:
        "EC50_PROTAC_and_ligand_range": {
            "pattern": r"EC50\s+of\s+PROTAC\s+is\s+[\d.]+[nuμ]M-\s*[\d.]+",
            "handler": "parse_protac_ligand_ec50_range",
        },

        # Two-cell PROTAC EC50 (simple format):
        "MDA_MB231_MDA_MB435_EC50_PROTAC": {
            "pattern": r"EC50\s+of\s+PROTAC\s+is\s+[\d.]+\s*uM\s+for\s+MDA-MB-231",
            "handler": "parse_two_cell_ec50_simple",
        },

        # NCI-H2030: EC50/DC50/Dmax ranges:
        "NCI_H2030_EC50_DC50_Dmax_range": {
            "pattern": r"EC50/DC50/Dmax\s+reported\s+above\s+were\s+obtained\s+using\s+NCI-H2030",
            "handler": "parse_nci_h2030_ec50_dc50_dmax_range",
        },

        # HeLa vs HEK293:
        "HeLa_vs_HEK293_DC50_Dmax": {
            "pattern": r"DC50\s+for\s+HEK293\s+cells",
            "handler": "parse_hela_vs_hek293_dc50_dmax",
        },

        # EGFR Exon 20 Ins + Exon 19 del / L858R:
        "EGFR_variant_DC50_Dmax": {
            "pattern": r"DC50\s+is\s+for\s+WT\s+EGFR\.|DC50\s+is\s+for\s+EGFR\s+\(Exon\s+19\s+del\)",
            "handler": "parse_egfr_variant_dc50_dmax",
        },

        # MCF7/T47D DC50 + DCmax block:
        "MCF7_T47D_DC50_block": {
            "pattern": r"DC50\s+is\s+0\.17\s*nM\s*MCF-7,\s*0\.43",
            "handler": "parse_mcf7_t47d_dc50_block",
        },

        # Generic EC50 of ligand / PROTAC, when only cell line is mentioned:
        "Generic_RS4_11_EC50": {
            "pattern": r"EC50\s+of\s+ligand\s+and\s+PROTAC\s+was\s+measured\s+in\s+RS4;11",
            "handler": "parse_rs4_11_ec50_ligand_protac",
        },

        # ZFP91 DC50 in text:
        "ZFP91_DC50_from_XD2_149": {
            "pattern": r"ZFP91\s*\(DC50:\s*[\d.]+uM",
            "handler": "parse_zfp91_dc50",
        },

        # Comments that mostly carry narrative/flags are mapped to no-op handlers:
        "Covalent_PROTAC_flag": {
            "pattern": r"Covalent\s+Protac",
            "handler": "parse_covalent_protac_flag",
        },
        "ERRa_in_vivo_efficacy_flag": {
            "pattern": r"PROTAC_ERRα\s+is\s+efficacious\s+in\s+mice",
            "handler": "parse_erralpha_in_vivo_flag",
        },
        "Anti_HCV_cyclophilinA_flag": {
            "pattern": r"anti-HCV\s+activity",
            "handler": "parse_anti_hcv_host_protein_flag",
        },
        "BCR_ABL_degradation_flag": {
            "pattern": r"BCR-ABL\s+degre|BCR-ABL\s+degrad",
            "handler": "parse_bcr_abl_degradation_flag",
        },
        "Target_engagement_in_vitro_flag": {
            "pattern": r"Engag[e]?ment\s+was\s+tested\s+in-vitro",
            "handler": "parse_engagement_in_vitro_flag",
        },
        "PDB_ligand_Am80": {
            "pattern": r"The\s+PDB\s+ligand\s+ID\s+is\s+for\s+Am80",
            "handler": "parse_pdb_ligand_am80",
        },
    }

    return overrides


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def parse_nan_comment(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """Treat 'nan' comment as no additional info."""
    return []


def parse_dc50_range(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is between 0.1uM-1uM
    DC50 is 1-3nM
    DC50 is 10-30nM
    DC50 is 1487 - 1994.5 nM
    """
    results: List[Dict[str, Any]] = []

    # with "between"
    m = re.search(
        r"DC50\s+is\s+between\s+([\d.]+)\s*([nμu]M)?\s*[-–~]\s*([\d.]+)\s*([nμu]M)?",
        comment,
        flags=re.IGNORECASE,
    )
    if not m:
        # without "between" but explicit unit(s)
        m = re.search(
            r"DC50\s+is\s+([\d.]+)\s*([nμu]M|nM)\s*[-–~]\s*([\d.]+)\s*([nμu]M|nM)",
            comment,
            flags=re.IGNORECASE,
        )
    if not m:
        # numbers with unit only once at the end, e.g. "DC50 is 1487 - 1994.5 nM"
        m = re.search(
            r"DC50\s+is\s+([\d.]+)\s*[-–~]\s*([\d.]+)\s*(nM|[nμu]M)",
            comment,
            flags=re.IGNORECASE,
        )
        if m:
            low, high, unit = m.groups()
            low_v = float(low)
            high_v = float(high)
            mean_v = 0.5 * (low_v + high_v)
            results.append(
                _make_numeric_row(
                    row,
                    value=mean_v,
                    value_type="DC50",
                    unit=unit,
                    category="range",
                    extra={"Value_Min": low_v, "Value_Max": high_v},
                )
            )
        return results

    v1, u1, v2, u2 = m.groups()
    low_v = float(v1)
    high_v = float(v2)
    unit = u1 or u2 or "nM"
    mean_v = 0.5 * (low_v + high_v)
    results.append(
        _make_numeric_row(
            row,
            value=mean_v,
            value_type="DC50",
            unit=unit,
            category="range",
            extra={"Value_Min": low_v, "Value_Max": high_v},
        )
    )
    return results


def parse_cdk4_dc50_bound(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 for CDK4 > 100 nM
    DC50 for CDK4 > 50 nM
    DC50 for CDK4 > 500 nM
    """
    m = re.search(r"DC50\s+for\s+CDK4\s*([>≥]+)\s*([\d.]+)\s*nM", comment, re.IGNORECASE)
    if not m:
        return []
    
    operator, value = m.groups()
    return [
        _make_numeric_row(
            row,
            value=float(value),
            value_type="DC50",
            unit="nM",
            category="inequality",
            operator=operator,
            poi_name="CDK4",
            extra={"Value_Range_Min": float(value)},
        )
    ]


def parse_dc50_two_values_no_cell(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is 1.76 nM and 4.5 nM
    """
    m = re.search(r"DC50\s+is\s+([\d.]+)\s*nM\s+and\s+([\d.]+)\s*nM", comment, re.IGNORECASE)
    if not m:
        return []
    
    return [
        _make_numeric_row(row, value=float(v), value_type="DC50", unit="nM", extra={"Replicate": i})
        for i, v in enumerate(m.groups(), start=1)
    ]


def parse_ligand_dual_ic50_modes(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Ligand IC50 is 0.26 nM (Kd), 4.0 nM (kinase activity)
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"Ligand\s+IC50\s+is\s+([\d.]+)\s*nM\s*\(([^)]+)\),\s*([\d.]+)\s*nM\s*\(([^)]+)\)",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    v1, a1, v2, a2 = m.groups()
    for v, assay in ((v1, a1), (v2, a2)):
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="LIGAND_IC50",
                unit="nM",
                category="numeric",
                extra={"Assay_Description": assay.strip()},
            )
        )
    return results


def parse_ligand_ic50_range(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    IC50 of the ligand is between 2nM and 7nM
    IC50 of the ligand is between 1nM-34 nM
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"IC50\s+of\s+the\s+ligand\s+is\s+between\s+([\d.]+)\s*nM\s*(?:and|-)\s*([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    low, high = m.groups()
    low_v, high_v = float(low), float(high)
    mean_v = 0.5 * (low_v + high_v)
    results.append(
        _make_numeric_row(
            row,
            value=mean_v,
            value_type="LIGAND_IC50",
            unit="nM",
            category="range",
            extra={"Value_Min": low_v, "Value_Max": high_v},
        )
    )
    return results


def parse_ligand_ec50_range(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of ligand is between 0.044uM-1.06 uM
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"EC50\s+of\s+ligand\s+is\s+between\s+([\d.]+)\s*uM\s*[-–~]\s*([\d.]+)\s*uM",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    low, high = m.groups()
    low_v, high_v = float(low), float(high)
    mean_v = 0.5 * (low_v + high_v)
    results.append(
        _make_numeric_row(
            row,
            value=mean_v,
            value_type="LIGAND_EC50",
            unit="uM",
            category="range",
            extra={"Value_Min": low_v, "Value_Max": high_v},
        )
    )
    return results


def parse_btk_ligand_ic50_ec50(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    IC50 of ligand is 4 nM (BTK inhibition)/ 0.3 nM (binding).
    EC50 of ligand is 20-30 nM (B-cell signaling).
    """
    results: List[Dict[str, Any]] = []

    ic50_m = re.search(
        r"IC50\s+of\s+ligand\s+is\s+([\d.]+)\s*nM\s*\(([^)]+)\)/\s*([\d.]+)\s*nM\s*\(([^)]+)\)",
        comment,
        re.IGNORECASE,
    )
    if ic50_m:
        v1, a1, v2, a2 = ic50_m.groups()
        for v, assay in ((v1, a1), (v2, a2)):
            results.append(
                _make_numeric_row(
                    row,
                    value=float(v),
                    value_type="LIGAND_IC50",
                    unit="nM",
                    category="numeric",
                    extra={"Assay_Description": assay.strip()},
                )
            )

    ec50_m = re.search(
        r"EC50\s+of\s+ligand\s+is\s+([\d.]+)\s*[-–~]\s*([\d.]+)\s*nM\s*\(([^)]+)\)",
        comment,
        re.IGNORECASE,
    )
    if ec50_m:
        low, high, assay = ec50_m.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="LIGAND_EC50",
                unit="nM",
                category="range",
                extra={
                    "Value_Min": low_v,
                    "Value_Max": high_v,
                    "Assay_Description": assay.strip(),
                },
            )
        )
    return results


def parse_btk_ligand_ic50_variants(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    IC50 of ligand is 51.0 nM (WT BTK), 30.7 (C481S)
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"IC50\s+of\s+ligand\s+is\s+([\d.]+)\s*nM\s*\((WT\s+BTK)\),\s*([\d.]+)\s*\((C481S)\)",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    v1, var1, v2, var2 = m.groups()
    for v, var in ((v1, var1), (v2, var2)):
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="LIGAND_IC50",
                unit="nM",
                category="numeric",
                extra={"Variant": var},
            )
        )
    return results


def parse_cytotoxicity_ic50(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Cytotoxicity IC50 (CC50): 37.43uM
    """
    m = re.search(r"Cytotoxicity\s+IC50\s*\(CC50\)\s*:\s*([\d.]+)\s*([nμu]M)", comment, re.IGNORECASE)
    if not m:
        return []
    
    value, unit = m.groups()
    return [_make_numeric_row(row, value=float(value), value_type="CYTOTOXICITY_IC50", unit=unit)]


def parse_akt_panel_metrics(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    IC50 of ligand is AKT1 5 nM, AKT2 18 nM, AKT3 8 nM.
    EC50 of PROTAC is AKT1 2 nM, AKT2 6.8 nM, AKT3 3.5 nM.
    Also ranges for EC50 of ligand/PROTAC (cell dependent).
    """
    results: List[Dict[str, Any]] = []

    # IC50 ligand panel
    ic50_m = re.search(
        r"IC50\s+of\s+ligand\s+is\s+AKT1\s+([\d.]+)\s*nM,\s*AKT2\s+([\d.]+)\s*nM,\s*AKT3\s+([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if ic50_m:
        for iso, v in zip(("AKT1", "AKT2", "AKT3"), ic50_m.groups()):
            results.append(
                _make_numeric_row(
                    row,
                    value=float(v),
                    value_type="LIGAND_IC50",
                    unit="nM",
                    category="numeric",
                    poi_name=iso,
                )
            )

    # EC50 PROTAC per isoform
    ec50_protac_m = re.search(
        r"EC50\s+of\s+PROTAC\s+is\s+AKT1\s+([\d.]+)\s*nM,\s*AKT2\s+([\d.]+)\s*nM,\s*AKT3\s+([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if ec50_protac_m:
        for iso, v in zip(("AKT1", "AKT2", "AKT3"), ec50_protac_m.groups()):
            results.append(
                _make_numeric_row(
                    row,
                    value=float(v),
                    value_type="PROTAC_EC50",
                    unit="nM",
                    category="numeric",
                    poi_name=iso,
                )
            )

    # EC50 ranges (cell dependent)
    ec50_lig_range = re.search(
        r"EC50\s+of\s+ligand\s+is\s+([\d.]+)\s*-\s*([\d.]+)\s*uM", comment, re.IGNORECASE
    )
    if ec50_lig_range:
        low, high = ec50_lig_range.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="LIGAND_EC50",
                unit="uM",
                category="range",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )

    ec50_prot_range = re.search(
        r"EC50\s+of\s+PROTAC\s+is\s+([\d.]+)\s*-\s*([\d.]+)\s*uM",
        comment,
        re.IGNORECASE,
    )
    if ec50_prot_range:
        low, high = ec50_prot_range.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="PROTAC_EC50",
                unit="uM",
                category="range",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )

    return results


def parse_hdac_panel_dmax(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    For HDAC1/HDAC2/HDAC3, the Dmax of this PROTAC is: >85%, >76%, >63%, respectively.
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"HDAC1/HDAC2/HDAC3.*Dmax.*:\s*([>~\d.%]+),\s*([>~\d.%]+),\s*([>~\d.%]+)",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    
    targets = ["HDAC1", "HDAC2", "HDAC3"]
    for target, raw_value in zip(targets, m.groups()):
        value, operator = _parse_value_with_operator(raw_value.replace('%', ''))
        if value is not None:
            results.append(
                _make_numeric_row(
                    row,
                    value=value,
                    value_type="Dmax",
                    unit="%",
                    category="numeric",
                    operator=operator,
                    poi_name=target,
                )
            )
    return results


def parse_bclxl_selectivity(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """Selective BCLXL degradation: only in MOLT4 cells, not platelets - Narrative only."""
    return []


def parse_pmid24068666_dcmax_dc50_ranges(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Ligand name is from PMID 24068666. DCmax is 18-33%. DC50 is 1487 - 1994.5 nM.
    (and similar variants)
    """
    results: List[Dict[str, Any]] = []

    dcmax_m = re.search(
        r"DCmax\s+is\s+([\d.]+)\s*[-–~]\s*([\d.]+)\s*%",
        comment,
        re.IGNORECASE,
    )
    if dcmax_m:
        low, high = dcmax_m.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="Dmax",
                unit="%",
                category="range",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )

    dc50_m = re.search(
        r"DC50\s+is\s+([\d.]+)\s*[-–~]\s*([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if dc50_m:
        low, high = dc50_m.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="DC50",
                unit="nM",
                category="range",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )

    return results


def parse_cdk_panel_dc50(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is CDK2: 62 nM, CDK9: 33 nM.
    """
    m = re.search(
        r"DC50\s+is\s+CDK2:\s*([\d.]+)\s*nM,\s*CDK9:\s*([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return []
    
    return [
        _make_numeric_row(row, value=float(v), value_type="DC50", unit="nM", poi_name=target)
        for target, v in zip(["CDK2", "CDK9"], m.groups())
    ]


def parse_smarca_dc50_dcmax_simple(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is SMARCA2 300nM, SMARCA4 250nM.
    DCmax is SMARCA2 65%, SMARCA4 70%.
    """
    results: List[Dict[str, Any]] = []

    dc50_m = re.search(
        r"DC50\s+is\s+SMARCA2\s+([\d.]+)nM,\s*SMARCA4\s+([\d.]+)nM",
        comment,
        re.IGNORECASE,
    )
    if dc50_m:
        v2, v4 = dc50_m.groups()
        results.append(
            _make_numeric_row(
                row,
                value=float(v2),
                value_type="DC50",
                unit="nM",
                poi_name="SMARCA2",
            )
        )
        results.append(
            _make_numeric_row(
                row,
                value=float(v4),
                value_type="DC50",
                unit="nM",
                poi_name="SMARCA4",
            )
        )

    dcmax_m = re.search(
        r"DCmax\s+is\s*SMARCA2\s*([\d.]+)%\s*,\s*SMARCA4\s*([\d.]+)%",
        comment,
        re.IGNORECASE,
    )
    if dcmax_m:
        d2, d4 = dcmax_m.groups()
        results.append(
            _make_numeric_row(
                row,
                value=float(d2),
                value_type="Dmax",
                unit="%",
                poi_name="SMARCA2",
            )
        )
        results.append(
            _make_numeric_row(
                row,
                value=float(d4),
                value_type="Dmax",
                unit="%",
                poi_name="SMARCA4",
            )
        )
    return results


def parse_smarca_pbrm1_dc50_two_cells(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is (SMARCA2 6nM, SMARCA4 11nM, PBRM1 32nM in MV-4-11; SMARCA2 3.3nM, PBRM1 15.6nM in NCI-H1568)
    """
    results: List[Dict[str, Any]] = []

    # MV-4-11 part
    m1 = re.search(
        r"SMARCA2\s+([\d.]+)nM,\s*SMARCA4\s+([\d.]+)nM,\s*PBRM1\s+([\d.]+)nM\s+in\s+MV-4-11",
        comment,
        re.IGNORECASE,
    )
    if m1:
        v2, v4, vp = m1.groups()
        for target, v in (("SMARCA2", v2), ("SMARCA4", v4), ("PBRM1", vp)):
            results.append(
                _make_numeric_row(
                    row,
                    value=float(v),
                    value_type="DC50",
                    unit="nM",
                    poi_name=target,
                    cell_line="MV-4-11",
                )
            )

    # NCI-H1568 part
    m2 = re.search(
        r"SMARCA2\s+([\d.]+)nM,\s*PBRM1\s+([\d.]+)nM\s+in\s+NCI-H1568",
        comment,
        re.IGNORECASE,
    )
    if m2:
        v2, vp = m2.groups()
        for target, v in (("SMARCA2", v2), ("PBRM1", vp)):
            results.append(
                _make_numeric_row(
                    row,
                    value=float(v),
                    value_type="DC50",
                    unit="nM",
                    poi_name=target,
                    cell_line="NCI-H1568",
                )
            )
    return results


def parse_multi_cell_dmax_only(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Dmax is 76% Karpas 422, 59% ULA, 84% SUDHL4, 82% OCI-Ly1, 82% Ramos.
    """
    results: List[Dict[str, Any]] = []
    # crude but effective: find "<num>% NAME" pairs
    for value, cell in re.findall(r"([\d.]+)%\s+([A-Za-z0-9\-]+)", comment):
        results.append(
            _make_numeric_row(
                row,
                value=float(value),
                value_type="Dmax",
                unit="%",
                cell_line=cell,
            )
        )
    return results


def parse_pdedelta_metrics(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Dmax was measured in Panc-Tu-1 cell line.
    DC50 is 83.4% (24 h, Panc-Tu-1), 85% (24 h, 1 uM, Jurkat).
    IC50 Deltasonamide 1 is 203 pM, and of the Bn derivative is 8 nM.
    """
    results: List[Dict[str, Any]] = []

    # DC50 as percentages (unusual, but follow text)
    m = re.search(
        r"DC50\s+is\s+([\d.]+)%.*Panc-Tu-1\),\s*([\d.]+)%.*Jurkat",
        comment,
        re.IGNORECASE,
    )
    if m:
        p1, p2 = m.groups()
        results.append(
            _make_numeric_row(
                row,
                value=float(p1),
                value_type="DC50",
                unit="%",
                cell_line="Panc-Tu-1",
            )
        )
        results.append(
            _make_numeric_row(
                row,
                value=float(p2),
                value_type="DC50",
                unit="%",
                cell_line="Jurkat",
            )
        )

    # IC50 Deltasonamide 1: 203 pM (convert to nM)
    ic50_m = re.search(
        r"IC50\s+Deltasonamide\s+1\s+is\s+([\d.]+)\s*pM",
        comment,
        re.IGNORECASE,
    )
    if ic50_m:
        v_pm = float(ic50_m.group(1))
        v_nM = v_pm / 1000.0
        results.append(
            _make_numeric_row(
                row,
                value=v_nM,
                value_type="LIGAND_IC50",
                unit="nM",
                category="numeric",
                poi_name="PDEdelta",
                extra={"Ligand_Name": "Deltasonamide 1"},
            )
        )

    bn_m = re.search(
        r"Bn\s+derivative\s+is\s+([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if bn_m:
        v = float(bn_m.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="LIGAND_IC50",
                unit="nM",
                category="numeric",
                poi_name="PDEdelta",
                extra={"Ligand_Name": "Bn derivative"},
            )
        )
    return results


def parse_hela_vs_dlbcl_dc50_dmax(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 value is for HeLa cells. DC50 is 0.61 uM for DLBCL cells.
    DMAX value is for HeLa cells. DMAX is 96 % for DLBCL cells.
    """
    results: List[Dict[str, Any]] = []

    dc50_m = re.search(
        r"DC50\s+is\s+([\d.]+)\s*uM\s+for\s+DLBCL\s+cells",
        comment,
        re.IGNORECASE,
    )
    if dc50_m:
        v = float(dc50_m.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="DC50",
                unit="uM",
                cell_line="DLBCL",
            )
        )

    dmax_m = re.search(
        r"DMAX\s+is\s+([\d.]+)\s*%\s+for\s+DLBCL\s+cells",
        comment,
        re.IGNORECASE,
    )
    if dmax_m:
        v = float(dmax_m.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="Dmax",
                unit="%",
                cell_line="DLBCL",
            )
        )
    return results


def parse_btk_multi_cell_dc50(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is 6.3 nM (HBL1), 8.5 nM (Ramos), 9.2 nM (Mino), 11.4 nM (IgE MM)
    """
    results: List[Dict[str, Any]] = []
    for v, cell in re.findall(
        r"([\d.]+)\s*nM\s*\(([A-Za-z0-9\- ]+)\)", comment, re.IGNORECASE
    ):
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="DC50",
                unit="nM",
                cell_line=cell.strip(),
            )
        )
    return results


def parse_two_cell_dc50_with_error(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is SU-DHL-1: 3 ± 1 nM, NCI-H2228: 34 ± 9 nM.
    """
    results: List[Dict[str, Any]] = []
    for cell, v, err in re.findall(
        r"(SU-DHL-1|NCI-H2228):\s*([\d.]+)\s*±\s*([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    ):
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="DC50",
                unit="nM",
                cell_line=cell,
                value_error=float(err),
            )
        )
    return results


def parse_kyse520_mv411_block(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 for KYSE520 cell: 6.0nM;DC50 for MV4;11 cell: 2.6nM;
    EC50 for KYSE520 cell: 0.66uM; EC50 for MV4;11 cell: 9.9nM;
    Dmax for KYSE520 cell: >95%; Dmax for MV4;11 cell: >90%
    """
    results: List[Dict[str, Any]] = []

    for cell_pat, cell_name in [
        (r"KYSE520", "KYSE520"),
        (r"MV4;11", "MV4;11"),
    ]:
        dc50_m = re.search(
            rf"DC50\s+for\s+{cell_pat}\s+cell:\s*([\d.]+)nM", comment, re.IGNORECASE
        )
        if dc50_m:
            v = float(dc50_m.group(1))
            results.append(
                _make_numeric_row(
                    row,
                    value=v,
                    value_type="DC50",
                    unit="nM",
                    cell_line=cell_name,
                )
            )

        ec50_m = re.search(
            rf"EC50\s+for\s+{cell_pat}\s+cell:\s*([\d.]+)\s*([nμu]M)",
            comment,
            re.IGNORECASE,
        )
        if ec50_m:
            v, unit = ec50_m.groups()
            results.append(
                _make_numeric_row(
                    row,
                    value=float(v),
                    value_type="PROTAC_EC50",
                    unit=unit,
                    cell_line=cell_name,
                )
            )

        dmax_m = re.search(
            rf"Dmax\s+for\s+{cell_pat}\s+cell:\s*([>~\d.]+)%",
            comment,
            re.IGNORECASE,
        )
        if dmax_m:
            raw = dmax_m.group(1)
            num = re.search(r"([\d.]+)", raw)
            if num:
                v = float(num.group(1))
                results.append(
                    _make_numeric_row(
                        row,
                        value=v,
                        value_type="Dmax",
                        unit="%",
                        cell_line=cell_name,
                    )
                )

    return results


def parse_lncap_vcap_22rv1_ec50_dc50(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of ligand is LNCaP 25 nM, VCaP 87.4 nM, 22RV1 > 10 uM
    DC50 is 0.86nM in LNCaP, 0.76 in VCaP and 10.4 nM at 1uM in 22Rv1.
    """
    results: List[Dict[str, Any]] = []

    # EC50 ligand per cell
    m = re.search(
        r"EC50\s+of\s+ligand\s+is\s+LNCaP\s+([\d.]+)\s*nM,\s*VCaP\s+([\d.]+)\s*nM,\s*22RV1\s*>\s*([\d.]+)\s*uM",
        comment,
        re.IGNORECASE,
    )
    if m:
        v_ln, v_vc, v_22 = m.groups()
        results.append(
            _make_numeric_row(
                row,
                value=float(v_ln),
                value_type="LIGAND_EC50",
                unit="nM",
                cell_line="LNCaP",
            )
        )
        results.append(
            _make_numeric_row(
                row,
                value=float(v_vc),
                value_type="LIGAND_EC50",
                unit="nM",
                cell_line="VCaP",
            )
        )
        results.append(
            _make_numeric_row(
                row,
                value=float(v_22),
                value_type="LIGAND_EC50",
                unit="uM",
                category="inequality",
                cell_line="22Rv1",
                extra={"Value_Lower_Bound": float(v_22)},
            )
        )

    # DC50 PROTAC per cell
    dc_m = re.search(
        r"DC50\s+is\s+([\d.]+)nM\s+in\s+LNCaP,\s*([\d.]+)\s+in\s+VCaP\s+and\s+([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if dc_m:
        v_ln, v_vc, v_22 = dc_m.groups()
        for cell, v in (("LNCaP", v_ln), ("VCaP", v_vc), ("22Rv1", v_22)):
            results.append(
                _make_numeric_row(
                    row,
                    value=float(v),
                    value_type="DC50",
                    unit="nM",
                    cell_line=cell,
                )
            )

    return results


def parse_multicell_ec50_slash_format(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of PROTAC2: 17.36±1.43uM/31.19±2.95uM/... for MDA-MB-231, A549, A549/DDP, HUVEC cells, respectively.
    EC50: 0.058uM/0.18uM/>10uM for SU-DHL-1/H3122/A549 cells, respectively.
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"EC50(?:\s+of\s+\w+)?\s*:\s*([0-9.uM±/>\s]+)\s+for\s+([^()]+?)\s+cells",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    vals_str, cells_str = m.groups()
    vals_ch = [s.strip() for s in vals_str.split("/") if s.strip()]
    cells = [s.strip() for s in re.split(r"[,/]", cells_str) if s.strip()]
    if len(vals_ch) != len(cells):
        return results

    for val_chunk, cell in zip(vals_ch, cells):
        # allow ">10uM" or "0.058uM" or "0.058±0.01uM"
        mv = re.match(
            r"([>~]?)([\d.]+)\s*(?:±\s*([\d.]+))?\s*([nμu]M)",
            val_chunk,
            re.IGNORECASE,
        )
        if not mv:
            continue
        sign, v, err, unit = mv.groups()
        v_f = float(v)
        unit = _normalize_unit(unit)
        if sign == ">":
            cat = "inequality"
            extra = {"Value_Lower_Bound": v_f}
        else:
            cat = "numeric"
            extra = {}
        results.append(
            _make_numeric_row(
                row,
                value=v_f,
                value_type="PROTAC_EC50",
                unit=unit,
                category=cat,
                cell_line=cell,
                value_error=float(err) if err else None,
                extra=extra,
            )
        )
    return results


def parse_multicell_ec50_comma_format(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50: 0.413±0.087uM, 1.344±0.112uM, 0.657±0.008uM for PC9/HCC827/H1975 cells, respectively.
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"EC50:\s*([0-9.uM±,\s]+)\s+for\s+([^()]+?)\s+cells",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    vals_str, cells_str = m.groups()
    vals_ch = [s.strip() for s in vals_str.split(",") if s.strip()]
    cells = [s.strip() for s in cells_str.split("/") if s.strip()]
    if len(vals_ch) != len(cells):
        return results

    for val_chunk, cell in zip(vals_ch, cells):
        mv = re.match(
            r"([\d.]+)\s*(?:±\s*([\d.]+))?\s*([nμu]M)",
            val_chunk,
            re.IGNORECASE,
        )
        if not mv:
            continue
        v, err, unit = mv.groups()
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="PROTAC_EC50",
                unit=unit,
                cell_line=cell,
                value_error=float(err) if err else None,
            )
        )
    return results


def parse_ec50_in_cells_simple(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of PROTAC is 28nM in MV-4-11, 68nM in NCI-H1568.
    EC50 of PROTAC is 2.6 uM for MDA-MB-231 and 1.99 uM for MDA-MB-435 (handled separately).
    """
    results: List[Dict[str, Any]] = []
    # pattern: "EC50 of PROTAC is 28nM in MV-4-11, 68nM in NCI-H1568."
    for v, unit, cell in re.findall(
        r"([\d.]+)\s*([nμu]M)\s+in\s+([A-Za-z0-9\-\;]+)",
        comment,
        re.IGNORECASE,
    ):
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="PROTAC_EC50",
                unit=unit,
                cell_line=cell,
            )
        )
    return results


def parse_multicell_ec50_colon_format(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Dmax is measured in 10uM. EC50 of PROTAC is MCF-7: 2.70 ± 0.19 ... LO2: 41.11 ± 3.70 B16: 22.68 ± 2.03 uM.
    EC50 of ligand is MCF-7: 4.17 ± 0.31 ... B16: 14.49 ± 1.28 uM.
    """
    results: List[Dict[str, Any]] = []
    # capture "X: v ± err" pairs followed by unit at the end
    unit_m = re.search(r"([nμu]M)\.", comment, re.IGNORECASE)
    unit = unit_m.group(1) if unit_m else "uM"

    for cell, v, err in re.findall(
        r"([A-Za-z0-9\-]+)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
        comment,
        re.IGNORECASE,
    ):
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="PROTAC_EC50",
                unit=unit,
                cell_line=cell,
                value_error=float(err),
            )
        )
    return results


def parse_sf_h2228_ec50_ligand_protac(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of ligand is 2.7 ± 0.4nM (SF cells), 58.2 ± 20.7 nM (H2228 cells).
    EC50 of PROTAC is 1.7 ± 1 nM (SF cells), 46nM ± 16 nM (H2228 cells).
    """
    results: List[Dict[str, Any]] = []
    for kind, v, err, cell in re.findall(
        r"EC50\s+of\s+(ligand|PROTAC)\s+is\s+([\d.]+)\s*±\s*([\d.]+)\s*nM\s*\(([^)]+)\)",
        comment,
        re.IGNORECASE,
    ):
        value_type = "LIGAND_EC50" if kind.lower() == "ligand" else "PROTAC_EC50"
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type=value_type,
                unit="nM",
                cell_line=cell.strip(),
                value_error=float(err),
            )
        )
    # deal with "46nM ± 16 nM (H2228 cells)" variant
    m = re.search(
        r"EC50\s+of\s+PROTAC\s+is\s+([\d.]+)\s*nM\s*±\s*([\d.]+)\s*nM\s*\(H2228\s+cells\)",
        comment,
        re.IGNORECASE,
    )
    if m:
        v, err = m.groups()
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="PROTAC_EC50",
                unit="nM",
                cell_line="H2228",
                value_error=float(err),
            )
        )
    return results


def parse_egfr_baf3_ec50_mutant_wt(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50: 0.096uM for L858R/T790M mutant EGFR Ba/F3 cells; >10uM for wildtype EGFR Ba/F3 cells
    """
    results: List[Dict[str, Any]] = []
    m1 = re.search(
        r"EC50:\s*([\d.]+)\s*uM\s+for\s+L858R/T790M\s+mutant\s+EGFR\s+Ba/F3\s+cells",
        comment,
        re.IGNORECASE,
    )
    if m1:
        v = float(m1.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="PROTAC_EC50",
                unit="uM",
                poi_name="EGFR L858R/T790M",
                cell_line="Ba/F3",
            )
        )
    m2 = re.search(
        r">\s*([\d.]+)\s*uM\s+for\s+wildtype\s+EGFR\s+Ba/F3\s+cells",
        comment,
        re.IGNORECASE,
    )
    if m2:
        v = float(m2.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="PROTAC_EC50",
                unit="uM",
                category="inequality",
                poi_name="EGFR WT",
                cell_line="Ba/F3",
                extra={"Value_Lower_Bound": v},
            )
        )
    return results


def parse_protac_ligand_ec50_range(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of PROTAC is 0.0083nM-0.062nM. EC50 of ligand is 55.8-207 nM.
    """
    results: List[Dict[str, Any]] = []
    m_p = re.search(
        r"EC50\s+of\s+PROTAC\s+is\s+([\d.]+)\s*nM\s*-\s*([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if m_p:
        low, high = m_p.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="PROTAC_EC50",
                unit="nM",
                category="range",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )

    m_l = re.search(
        r"EC50\s+of\s+ligand\s+is\s+([\d.]+)\s*-\s*([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if m_l:
        low, high = m_l.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="LIGAND_EC50",
                unit="nM",
                category="range",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )
    return results


def parse_two_cell_ec50_simple(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of PROTAC is 2.6 uM for MDA-MB-231 and 1.99 uM for MDA-MB-435
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"EC50\s+of\s+PROTAC\s+is\s+([\d.]+)\s*uM\s+for\s+([A-Za-z0-9\-]+)\s+and\s+([\d.]+)\s*uM\s+for\s+([A-Za-z0-9\-]+)",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    v1, c1, v2, c2 = m.groups()
    for v, cell in ((v1, c1), (v2, c2)):
        results.append(
            _make_numeric_row(
                row,
                value=float(v),
                value_type="PROTAC_EC50",
                unit="uM",
                cell_line=cell,
            )
        )
    return results


def parse_nci_h2030_ec50_dc50_dmax_range(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50/DC50/Dmax reported above were obtained using NCI-H2030 cells.
    DC50: 0.25~0.76uM; Dmax: ~75%-90%
    """
    results: List[Dict[str, Any]] = []

    dc_m = re.search(
        r"DC50:\s*([\d.]+)\s*~\s*([\d.]+)\s*uM",
        comment,
        re.IGNORECASE,
    )
    if dc_m:
        low, high = dc_m.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="DC50",
                unit="uM",
                category="range",
                cell_line="NCI-H2030",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )

    dmax_m = re.search(
        r"Dmax:\s*~?([\d.]+)\s*%\s*-\s*([\d.]+)\s*%",
        comment,
        re.IGNORECASE,
    )
    if dmax_m:
        low, high = dmax_m.groups()
        low_v, high_v = float(low), float(high)
        mean_v = 0.5 * (low_v + high_v)
        results.append(
            _make_numeric_row(
                row,
                value=mean_v,
                value_type="Dmax",
                unit="%",
                category="range",
                cell_line="NCI-H2030",
                extra={"Value_Min": low_v, "Value_Max": high_v},
            )
        )
    return results


def parse_hela_vs_hek293_dc50_dmax(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 for HEK293 cells: 230nM; Dmax for HEK293 cells: 98%.
    """
    results: List[Dict[str, Any]] = []
    dc_m = re.search(
        r"DC50\s+for\s+HEK293\s+cells:\s*([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if dc_m:
        v = float(dc_m.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="DC50",
                unit="nM",
                cell_line="HEK293",
            )
        )
    dmax_m = re.search(
        r"Dmax\s+for\s+HEK293\s+cells:\s*([\d.]+)\s*%",
        comment,
        re.IGNORECASE,
    )
    if dmax_m:
        v = float(dmax_m.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="Dmax",
                unit="%",
                cell_line="HEK293",
            )
        )
    return results


def parse_egfr_variant_dc50_dmax(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is for WT EGFR. DC50 for EGFR Exon 20 Ins is 736.2 nM. DMAX is for WT EGFR. DMAX for EGFR Exon 20 Ins is 68.8 %.
    DC50 is for EGFR (Exon 19 del). DC50 for EGFR (L858R) 22.3 nM. ...
    """
    results: List[Dict[str, Any]] = []

    # Exon 20 Ins
    m1 = re.search(
        r"DC50\s+for\s+EGFR\s+Exon\s+20\s+Ins\s+is\s+([\d.]+)\s*nM",
        comment,
        re.IGNORECASE,
    )
    if m1:
        v = float(m1.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="DC50",
                unit="nM",
                poi_name="EGFR Exon20Ins",
            )
        )
    d1 = re.search(
        r"DMAX\s+for\s+EGFR\s+Exon\s+20\s+Ins\s+is\s+([\d.]+)\s*%",
        comment,
        re.IGNORECASE,
    )
    if d1:
        v = float(d1.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="Dmax",
                unit="%",
                poi_name="EGFR Exon20Ins",
            )
        )

    # Exon 19 del / L858R pair
    m2 = re.search(
        r"DC50\s+for\s+EGFR\s+\(L858R\)\s*([\d.]+)\s*nM", comment, re.IGNORECASE
    )
    if m2:
        v = float(m2.group(1))
        results.append(
            _make_numeric_row(
                row,
                value=v,
                value_type="DC50",
                unit="nM",
                poi_name="EGFR L858R",
            )
        )
    return results


def parse_mcf7_t47d_dc50_block(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    DC50 is 0.17 nM MCF-7, 0.43 T47D. ...
    """
    results: List[Dict[str, Any]] = []
    m = re.search(
        r"DC50\s+is\s+([\d.]+)\s*nM\s*MCF-7,\s*([\d.]+)\s*T47D",
        comment,
        re.IGNORECASE,
    )
    if not m:
        return results
    v1, v2 = m.groups()
    results.append(
        _make_numeric_row(
            row,
            value=float(v1),
            value_type="DC50",
            unit="nM",
            cell_line="MCF-7",
        )
    )
    results.append(
        _make_numeric_row(
            row,
            value=float(v2),
            value_type="DC50",
            unit="nM",
            cell_line="T47D",
        )
    )
    return results


def parse_rs4_11_ec50_ligand_protac(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    EC50 of ligand and PROTAC was measured in RS4;11 cell line.
    (no numeric data; narrative only)
    """
    return []


def parse_zfp91_dc50(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    ... pomalidomide itself can induce the degradation of CRBN neo-substrates like ZFP91 (DC50: 0.42uM, ...)
    """
    results: List[Dict[str, Any]] = []
    m = re.search(r"ZFP91\s*\(DC50:\s*([\d.]+)\s*uM", comment, re.IGNORECASE)
    if not m:
        return results
    v = float(m.group(1))
    results.append(
        _make_numeric_row(
            row,
            value=v,
            value_type="DC50",
            unit="uM",
            poi_name="ZFP91",
        )
    )
    return results


# ---------------------------------------------------------------------------
# Flag / no-op handlers (narrative only)
# ---------------------------------------------------------------------------

def parse_covalent_protac_flag(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    return []


def parse_erralpha_in_vivo_flag(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    return []


def parse_anti_hcv_host_protein_flag(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    return []


def parse_bcr_abl_degradation_flag(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    return []


def parse_engagement_in_vitro_flag(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    return []


def parse_pdb_ligand_am80(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    return []