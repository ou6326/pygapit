"""
Genotype and phenotype file I/O.
Translates GAPIT.HapMap.R, GAPIT.Numericalization.R, GAPIT.QC.R

Supports:
  - HapMap format (.hmp.txt): SNPs in rows, individuals in columns
  - Numeric format: individuals in rows (GD), separate map file (GM)
  - Phenotype files: taxa in col1, traits in remaining columns
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── IUPAC single-bit and double-bit genotype codes ────────────────────────
# 0 = homozygous reference, 1 = heterozygous, 2 = homozygous alternate
# Based on GAPIT.Numericalization.R lookup table

HETEROZYGOUS_1BIT = frozenset("RYSWKM")
HETEROZYGOUS_2BIT = frozenset(
    {"AT", "AG", "AC", "TA", "GA", "CA", "GT", "TG", "GC", "CG", "CT", "TC"}
)
MISSING_1BIT = frozenset({"N", "X", "-", "+", "/", "NA", "NAN"})
MISSING_2BIT = frozenset({"NN", "XX", "--", "++", "//", "00", "N", "NA", "NAN"})


@dataclass
class GenotypeData:
    """Container for processed genotype data."""

    GD: np.ndarray  # (n_individuals, n_snps) float, 0/1/2 coded
    GM: pd.DataFrame  # columns: SNP, Chromosome, Position
    taxa: np.ndarray  # (n,) individual IDs


@dataclass
class PhenotypeData:
    """Container for phenotype data."""

    Y: pd.DataFrame  # col0 = Taxa, col1+ = trait values
    taxa: np.ndarray  # (n,) individual IDs
    trait_names: list[str]


def _numericalize_snp(
    alleles: np.ndarray,
    major_allele_zero: bool = False,
) -> np.ndarray:
    """
    Convert a SNP's character allele calls to 0/1/2.
    Translates GAPIT.Numericalization.R

    Parameters
    ----------
    alleles : array of genotype strings for one SNP across all individuals
    major_allele_zero : if True, major allele = 0 (GAPIT's Major.allele.zero flag)

    Returns 0/1/2 coded array with NaN for missing.
    """
    values = np.char.upper(np.asarray(alleles, dtype=str))
    nonmissing_lengths = [
        len(value)
        for value in values
        if value not in MISSING_1BIT and value not in MISSING_2BIT
    ]
    bit = max(nonmissing_lengths, default=2)
    missing_codes = MISSING_1BIT if bit == 1 else MISSING_2BIT

    normalized = values.copy()
    if bit == 1:
        # GAPIT replaces K by Z so the heterozygote sorts after homozygotes.
        normalized[normalized == "K"] = "Z"
    normalized[np.isin(normalized, tuple(missing_codes))] = "N"

    levels = sorted(set(normalized) - {"N"})
    if bit == 2:
        heterozygotes = [level for level in levels if level in HETEROZYGOUS_2BIT]
        if len(heterozygotes) > 1:
            normalized[normalized == heterozygotes[1]] = heterozygotes[0]
            levels = sorted(set(normalized) - {"N"})

    if len(levels) <= 1 or len(levels) > 3:
        return np.zeros(len(normalized), dtype=float)

    counts = {level: int(np.count_nonzero(normalized == level)) for level in levels}
    if major_allele_zero:
        heterozygote_codes = HETEROZYGOUS_1BIT if bit == 1 else HETEROZYGOUS_2BIT
        if bit == 1 and len(levels) == 3:
            heterozygote_codes = heterozygote_codes | {"Z"}
        heterozygotes = [level for level in levels if level in heterozygote_codes]
        homozygotes = [level for level in levels if level not in heterozygote_codes]

        def major_allele_order(level: str) -> tuple[int, str]:
            return -counts[level], level

        homozygotes.sort(key=major_allele_order)
        if len(levels) == 3 and len(heterozygotes) == 1:
            levels = [homozygotes[0], heterozygotes[0], homozygotes[1]]
        elif not heterozygotes:
            levels = homozygotes

    result = np.full(len(normalized), np.nan, dtype=float)
    observed = normalized != "N"
    if len(levels) == 2:
        heterozygotes = [
            level
            for level in levels
            if level in (HETEROZYGOUS_1BIT if bit == 1 else HETEROZYGOUS_2BIT)
        ]
        if heterozygotes:
            result[observed] = 0.0
            result[normalized == heterozygotes[0]] = 1.0
        else:
            result[normalized == levels[0]] = 0.0
            result[normalized == levels[1]] = 2.0
    elif bit == 1:
        result[normalized == levels[0]] = 0.0
        result[normalized == levels[1]] = 2.0
        result[normalized == levels[2]] = 1.0
    else:
        heterozygotes = [level for level in levels if level in HETEROZYGOUS_2BIT]
        if len(heterozygotes) != 1:
            raise ValueError("Two-bit SNPs with three states require one heterozygote")
        homozygotes = [level for level in levels if level != heterozygotes[0]]
        result[normalized == homozygotes[0]] = 0.0
        result[normalized == heterozygotes[0]] = 1.0
        result[normalized == homozygotes[1]] = 2.0

    return result


def read_hapmap(
    filepath: str | Path | pd.DataFrame,
    major_allele_zero: bool = False,
    impute_method: str = "middle",
) -> GenotypeData:
    """
    Read HapMap-format genotype file and convert to numeric.
    Translates GAPIT.HapMap.R

    HapMap format:
      - SNPs in rows, individuals in columns
      - First 11 columns: rs, alleles, chrom, pos, strand, assembly,
        center, protLSID, assayLSID, panelLSID, QCcode
      - Remaining columns: genotype calls per individual

    Parameters
    ----------
    filepath : path to .hmp.txt file
    major_allele_zero : if True, major allele coded as 0 (minor = 2)
    impute_method : 'middle' (1), 'major' (2), 'minor' (0), or 'none'

    Returns
    -------
    GenotypeData with GD (n×m), GM (m×3), taxa (n,)
    """
    if isinstance(filepath, pd.DataFrame):
        header = pd.DataFrame([filepath.columns], columns=filepath.columns)
        raw = pd.concat([header, filepath], ignore_index=True)
    else:
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"HapMap file not found: {fp}")
        # Read with no header (row 0 is header)
        raw = pd.read_csv(fp, sep="\t", header=None, low_memory=False)
    raw.shape[1]
    n_meta = 11  # first 11 columns are SNP metadata

    # Extract taxa names from first row, skip first 11 columns
    taxa = np.asarray(raw.iloc[0, n_meta:].values, dtype=str)
    # Extract SNP info: rs (col 0), chrom (col 2), pos (col 3)
    snp_info = raw.iloc[1:, [0, 2, 3]].copy()
    snp_info.columns = ["SNP", "Chromosome", "Position"]
    snp_info["Position"] = pd.to_numeric(snp_info["Position"], errors="coerce")
    snp_info = snp_info.reset_index(drop=True)

    # Genotype block: rows = SNPs, cols = individuals
    geno_block = raw.iloc[1:, n_meta:].values  # (n_snps, n_individuals)

    n_snps, n_indiv = geno_block.shape

    # Convert each SNP row to numeric
    GD_T = np.full((n_snps, n_indiv), np.nan)
    for i in range(n_snps):
        GD_T[i, :] = _numericalize_snp(geno_block[i, :], major_allele_zero)

    # Transpose: GD should be (n_individuals, n_snps)
    GD = GD_T.T

    # Impute missing values
    GD = impute_missing(GD, method=impute_method)

    return GenotypeData(GD=GD, GM=snp_info, taxa=taxa)


def read_numeric(
    gd_path: str | Path,
    gm_path: str | Path | pd.DataFrame,
    impute_method: str = "middle",
) -> GenotypeData:
    """
    Read numeric genotype format (GD + GM files).
    Translates GAPIT numeric format reading.

    GD format: rows = individuals, cols = SNPs (0/1/2 coded)
               first row = header (SNP names), first col = taxa IDs
    GM format: 3 columns: SNP, Chromosome, Position

    Parameters
    ----------
    gd_path : path to numeric genotype file
    gm_path : path to SNP map file
    impute_method : 'middle', 'major', 'minor'

    Returns
    -------
    GenotypeData
    """
    gd_path = Path(gd_path)

    # Read GD — GAPIT format: col 0 = taxa names, col 1+ = SNP genotypes
    gd_df = pd.read_csv(gd_path, sep="\t", low_memory=False)
    taxa = np.asarray(gd_df.iloc[:, 0].astype(str).values, dtype=str)
    GD = gd_df.iloc[:, 1:].values.astype(float)

    # Read GM
    gm_df = (
        gm_path.copy()
        if isinstance(gm_path, pd.DataFrame)
        else pd.read_csv(gm_path, sep="\t", header=0)
    )
    if gm_df.shape[1] >= 3:
        gm_df = gm_df.iloc[:, :3]
        gm_df.columns = ["SNP", "Chromosome", "Position"]

    # Validate column alignment
    if GD.shape[1] != len(gm_df):
        raise ValueError(
            f"GD has {GD.shape[1]} SNPs but GM has {len(gm_df)} rows. "
            "Ensure GD columns and GM rows are in the same order."
        )

    # Impute missing
    GD = impute_missing(GD, method=impute_method)

    return GenotypeData(GD=GD, GM=gm_df, taxa=taxa)


def impute_missing(GD: np.ndarray, method: str = "middle") -> np.ndarray:
    """
    Impute missing genotype values.
    Translates GAPIT's SNP.impute options.

    Methods:
      'middle' : impute with 1 (heterozygous / mean dosage)
      'major'  : impute with 2 (homozygous major)
      'minor'  : impute with 0 (homozygous minor)
      'mean'   : impute with column mean (population allele frequency)
      'none'   : leave as NaN
    """
    GD = GD.copy()
    missing = np.isnan(GD)

    if method == "middle":
        GD[missing] = 1.0
    elif method == "major":
        GD[missing] = 2.0
    elif method == "minor":
        GD[missing] = 0.0
    elif method == "mean":
        col_means = np.nanmean(GD, axis=0)
        for j in range(GD.shape[1]):
            mask = np.isnan(GD[:, j])
            if mask.any():
                GD[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 1.0
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown impute method: {method}")

    return GD


def read_phenotype(filepath: str | Path) -> PhenotypeData:
    """
    Read phenotype file.
    Format: tab-delimited, first col = Taxa, remaining = trait values.
    Missing = NA or NaN.

    Returns
    -------
    PhenotypeData with Y (DataFrame), taxa, trait_names
    """
    fp = Path(filepath)
    df = pd.read_csv(fp, sep="\t", na_values=["NA", "NaN", "nan", "N/A"])
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)

    taxa = np.asarray(df.iloc[:, 0].values, dtype=str)
    trait_names = df.columns[1:].tolist()

    return PhenotypeData(Y=df, taxa=taxa, trait_names=trait_names)


def align_taxa(
    pheno: PhenotypeData,
    geno: GenotypeData,
    cv_df: pd.DataFrame | None = None,
    ki_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Align all input datasets to common taxa.
    Translates GAPIT.IC.R and GAPIT.QC.R taxa-matching logic.

    GAPIT rule: only the intersection of taxa across all provided
    datasets is kept. Taxa names are case-sensitive.

    Returns
    -------
    dict with 'Y', 'GD', 'GM', 'taxa', 'KI' (optional), 'CV' (optional)
    """
    common_taxa = set(pheno.taxa) & set(geno.taxa)

    if cv_df is not None:
        cv_taxa = set(cv_df.iloc[:, 0].astype(str).values)
        common_taxa &= cv_taxa

    if ki_df is not None:
        ki_taxa = set(ki_df.iloc[:, 0].astype(str).values)
        common_taxa &= ki_taxa

    if len(common_taxa) == 0:
        raise ValueError(
            "No common taxa found across input files. "
            "Check that taxa names match exactly (case-sensitive) "
            "across phenotype, genotype, and any kinship/covariate files."
        )

    common_taxa = sorted(common_taxa)
    n_common = len(common_taxa)

    if n_common < len(pheno.taxa):
        warnings.warn(
            f"Kept {n_common}/{len(pheno.taxa)} taxa after alignment. "
            f"Dropped {len(pheno.taxa) - n_common} due to missing data."
        )

    taxa_arr = np.array(common_taxa)

    # Align phenotype
    pheno_idx = [np.where(pheno.taxa == t)[0][0] for t in common_taxa]
    Y_aligned = pheno.Y.iloc[pheno_idx].reset_index(drop=True)

    # Align genotype
    geno_idx = [np.where(geno.taxa == t)[0][0] for t in common_taxa]
    GD_aligned = geno.GD[geno_idx, :]

    result = {
        "taxa": taxa_arr,
        "Y": Y_aligned,
        "GD": GD_aligned,
        "GM": geno.GM,
    }

    # Align kinship if provided
    if ki_df is not None:
        ki_taxa_col = ki_df.iloc[:, 0].astype(str).values
        ki_idx = [np.where(ki_taxa_col == t)[0][0] for t in common_taxa]
        ki_vals = ki_df.iloc[:, 1:].values
        KI_aligned = ki_vals[np.ix_(ki_idx, ki_idx)].astype(float)
        result["KI"] = KI_aligned

    # Align covariates if provided
    if cv_df is not None:
        cv_taxa_col = cv_df.iloc[:, 0].astype(str).values
        cv_idx = [np.where(cv_taxa_col == t)[0][0] for t in common_taxa]
        CV_aligned = cv_df.iloc[cv_idx, 1:].values.astype(float)
        result["CV"] = CV_aligned

    return result


def maf_filter(
    GD: np.ndarray, threshold: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter SNPs by minor allele frequency.
    Translates GAPIT.QC.R MAF filtering logic.

    Parameters
    ----------
    GD : (n, m) genotype matrix
    threshold : minimum MAF (default 0.05)

    Returns
    -------
    (filtered_GD, kept_indices)
    """
    n = GD.shape[0]
    freq = np.nansum(GD, axis=0) / (2.0 * n)
    maf = np.minimum(freq, 1.0 - freq)
    keep = maf >= threshold
    return GD[:, keep], np.where(keep)[0]
