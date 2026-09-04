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

from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    Matrix,
    StrVector,
    Vector,
    as_float_matrix,
    as_str_vector,
    require_length,
)

# ── IUPAC single-bit and double-bit genotype codes ────────────────────────
# 0 = homozygous reference, 1 = heterozygous, 2 = homozygous alternate
# Based on GAPIT.Numericalization.R lookup table

HETEROZYGOUS_1BIT = frozenset("RYSWKM")
HETEROZYGOUS_2BIT = frozenset({
    "AT",
    "AG",
    "AC",
    "TA",
    "GA",
    "CA",
    "GT",
    "TG",
    "GC",
    "CG",
    "CT",
    "TC",
})
MISSING_1BIT = frozenset({"N", "X", "-", "+", "/", "NA", "NAN"})
MISSING_2BIT = frozenset({"NN", "XX", "--", "++", "//", "00", "N", "NA", "NAN"})


@dataclass
class GenotypeData:
    """Container for processed genotype data."""

    GD: FloatMatrix  # (n_individuals, n_snps), 0/1/2 coded
    GM: pd.DataFrame  # columns: SNP, Chromosome, Position
    taxa: StrVector  # individual IDs

    def __post_init__(self) -> None:
        """Enforce the array contract even for direct construction."""
        self.GD = as_float_matrix(self.GD, name="genotype matrix")
        self.taxa = as_str_vector(self.taxa, name="genotype taxa")
        require_length(self.taxa, self.GD.shape[0], name="genotype taxa")
        if len(self.GM) != self.GD.shape[1]:
            raise ValueError(
                "marker map must have one row per genotype column; "
                f"expected {self.GD.shape[1]}, got {len(self.GM)}"
            )

    @classmethod
    def from_numeric_frame(
        cls,
        genotype: pd.DataFrame,
        marker_map: pd.DataFrame,
        impute_method: str = "middle",
    ) -> GenotypeData:
        """Build validated numeric genotype data from labeled tables."""
        return _numeric_from_frames(
            genotype, marker_map, impute_method, "Numeric genotype"
        )

    @classmethod
    def from_array(
        cls,
        genotype: Matrix,
        marker_map: pd.DataFrame,
        taxa: Vector,
        impute_method: str = "middle",
    ) -> GenotypeData:
        """Build validated numeric genotype data using explicit row taxa."""
        try:
            values = np.asarray(genotype, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric genotype array values must be numeric") from exc
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError(
                "Numeric genotype array must be two-dimensional with at least one SNP"
            )
        taxa_array = _taxa_array(taxa, "genotype")
        if values.shape[0] != len(taxa_array):
            raise ValueError(
                "Numeric genotype array must have one row per supplied taxon; "
                f"expected {len(taxa_array)}, got {values.shape[0]}"
            )
        if np.isinf(values).any():
            raise ValueError("Numeric genotype array must not contain infinity")
        normalized_map = _marker_map_from_frame(marker_map, "Marker map")
        if values.shape[1] != len(normalized_map):
            raise ValueError(
                f"Numeric genotype array has {values.shape[1]} SNPs but marker map "
                f"has {len(normalized_map)} rows"
            )
        return cls(
            GD=impute_missing(values, method=impute_method),
            GM=normalized_map,
            taxa=taxa_array,
        )


@dataclass
class PhenotypeData:
    """Container for phenotype data."""

    Y: pd.DataFrame  # col0 = Taxa, col1+ = trait values
    taxa: StrVector  # individual IDs
    trait_names: list[str]

    @classmethod
    def from_frame(cls, phenotype: pd.DataFrame) -> PhenotypeData:
        """Build validated phenotype data from a labeled table."""
        return _phenotype_from_frame(phenotype, "Phenotype data")


@dataclass(frozen=True, slots=True)
class AlignedData:
    """Typed result of aligning phenotype, genotype, and optional inputs."""

    taxa: StrVector
    phenotypes: pd.DataFrame
    genotypes: FloatMatrix
    markers: pd.DataFrame
    kinship: FloatMatrix | None = None
    covariates: FloatMatrix | None = None

    def as_legacy_dict(self) -> dict[str, Any]:
        """Return the historical mapping produced by :func:`align_taxa`."""
        result: dict[str, Any] = {
            "taxa": self.taxa,
            "Y": self.phenotypes,
            "GD": self.genotypes,
            "GM": self.markers,
        }
        if self.kinship is not None:
            result["KI"] = self.kinship
        if self.covariates is not None:
            result["CV"] = self.covariates
        return result


def _unique_taxa_index(values: Vector, source: str) -> dict[str, int]:
    """Build a taxa-to-row mapping and reject ambiguous duplicate IDs."""
    taxa = np.asarray(values, dtype=str)
    if taxa.ndim != 1:
        raise ValueError(f"{source} taxa must be one-dimensional")

    index: dict[str, int] = {}
    duplicates: list[str] = []
    for row, taxon in enumerate(taxa):
        name = str(taxon)
        if name in index:
            duplicates.append(name)
        else:
            index[name] = row
    if duplicates:
        duplicate_names = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate taxa in {source}: {duplicate_names}")
    return index


def _taxa_array(values: pd.Series[Any] | Vector, source: str) -> StrVector:
    """Return validated, non-empty string taxa identifiers."""
    series = pd.Series(values, copy=False)
    if series.isna().any():
        raise ValueError(f"{source} taxa must not contain missing values")
    taxa = np.asarray(series.astype(str), dtype=str)
    if np.any(np.char.strip(taxa) == ""):
        raise ValueError(f"{source} taxa must not contain empty values")
    _unique_taxa_index(taxa, source)
    return taxa


def _phenotype_from_frame(df: pd.DataFrame, source: str) -> PhenotypeData:
    """Validate and normalize a phenotype DataFrame."""
    if df.shape[1] < 2:
        raise ValueError(f"{source} must contain a taxa column and at least one trait")
    if df.empty:
        raise ValueError(f"{source} must contain at least one phenotype row")

    result = df.copy()
    taxa = _taxa_array(result.iloc[:, 0], "phenotype")
    result.isetitem(0, np.asarray(taxa, dtype=object))
    trait_names = result.columns[1:].tolist()
    if len(set(trait_names)) != len(trait_names):
        raise ValueError(f"{source} contains duplicate trait names")
    for column in result.columns[1:]:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Phenotype trait {column!r} must contain numeric values"
            ) from exc
    if np.isinf(result.iloc[:, 1:].to_numpy(dtype=float)).any():
        raise ValueError(f"{source} trait values must not contain infinity")
    return PhenotypeData(Y=result, taxa=taxa, trait_names=trait_names)


def _marker_map_from_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate and normalize a GAPIT marker map."""
    if df.shape[1] < 3:
        raise ValueError(f"{source} must contain SNP, Chromosome, and Position columns")
    if df.empty:
        raise ValueError(f"{source} must contain at least one marker")

    result = df.iloc[:, :3].copy()
    result.columns = ["SNP", "Chromosome", "Position"]
    if result[["SNP", "Chromosome", "Position"]].isna().any(axis=None):
        raise ValueError(f"{source} must not contain missing marker annotations")
    result["SNP"] = result["SNP"].astype(str)
    chromosome_labels = result["Chromosome"].astype(str)
    if (result["SNP"].str.strip() == "").any() or (
        chromosome_labels.str.strip() == ""
    ).any():
        raise ValueError(f"{source} must not contain empty marker annotations")
    if result["SNP"].duplicated().any():
        raise ValueError(f"{source} contains duplicate SNP identifiers")
    try:
        result["Position"] = pd.to_numeric(result["Position"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} positions must be numeric") from exc
    if np.isinf(result["Position"].to_numpy(dtype=float)).any():
        raise ValueError(f"{source} positions must not contain infinity")
    return result.reset_index(drop=True)


def _numeric_from_frames(
    gd_df: pd.DataFrame,
    gm_df: pd.DataFrame,
    impute_method: str,
    source: str,
) -> GenotypeData:
    """Validate and normalize numeric genotype and marker-map frames."""
    if gd_df.shape[1] < 2:
        raise ValueError(f"{source} must contain a taxa column and at least one SNP")
    if gd_df.empty:
        raise ValueError(f"{source} must contain at least one genotype row")

    taxa = _taxa_array(gd_df.iloc[:, 0], "genotype")
    marker_map = _marker_map_from_frame(gm_df, "Marker map")
    genotype_markers = np.asarray(gd_df.columns[1:], dtype=str).tolist()
    map_markers = marker_map["SNP"].tolist()
    if genotype_markers != map_markers:
        raise ValueError(
            "Numeric genotype SNP columns must match marker-map rows in order"
        )
    try:
        values = gd_df.iloc[:, 1:].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} SNP values must be numeric") from exc
    if np.isinf(values).any():
        raise ValueError(f"{source} SNP values must not contain infinity")
    return GenotypeData(
        GD=_impute_missing_inplace(values, method=impute_method),
        GM=marker_map,
        taxa=taxa,
    )


def _numericalize_snp(
    alleles: Vector,
    major_allele_zero: bool = False,
    *,
    already_uppercase: bool = False,
) -> FloatVector:
    """
    Convert a SNP's character allele calls to 0/1/2.
    Translates GAPIT.Numericalization.R

    Parameters
    ----------
    alleles : array of genotype strings for one SNP across all individuals
    major_allele_zero : if True, major allele = 0 (GAPIT's Major.allele.zero flag)

    Returns 0/1/2 coded array with NaN for missing.
    """
    values = np.asarray(alleles, dtype=str)
    if not already_uppercase:
        values = np.char.upper(values)
    unique_values = np.unique(values)
    nonmissing_lengths = [
        len(value)
        for value in unique_values
        if value not in MISSING_1BIT and value not in MISSING_2BIT
    ]
    bit = max(nonmissing_lengths, default=2)
    missing_codes = MISSING_1BIT if bit == 1 else MISSING_2BIT

    normalized = values.copy()
    if bit == 1:
        # GAPIT replaces K by Z so the heterozygote sorts after homozygotes.
        normalized[normalized == "K"] = "Z"
    for value in unique_values:
        if value in missing_codes:
            normalized[normalized == value] = "N"

    levels = np.unique(normalized[normalized != "N"]).tolist()
    if bit == 2:
        heterozygotes = [level for level in levels if level in HETEROZYGOUS_2BIT]
        if len(heterozygotes) > 1:
            normalized[normalized == heterozygotes[1]] = heterozygotes[0]
            levels = np.unique(normalized[normalized != "N"]).tolist()

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
        raw = filepath
        raw_taxa: StrVector = np.asarray(raw.columns[11:], dtype=str)
    else:
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"HapMap file not found: {fp}")
        header = pd.read_csv(
            fp,
            sep="\t",
            header=None,
            nrows=1,
            dtype=str,
            keep_default_na=False,
        )
        raw_taxa = np.asarray(header.iloc[0, 11:], dtype=str)
        raw = pd.read_csv(fp, sep="\t", low_memory=False)
    n_meta = 11  # first 11 columns are SNP metadata
    if raw.shape[1] <= n_meta:
        raise ValueError("HapMap data must contain 11 metadata columns and taxa")
    if raw.empty:
        raise ValueError("HapMap data must contain at least one marker row")

    # Extract taxa names from the header, skipping the metadata columns.
    taxa = _taxa_array(raw_taxa, "HapMap")
    # Extract SNP info: rs (col 0), chrom (col 2), pos (col 3)
    snp_info = _marker_map_from_frame(raw.iloc[:, [0, 2, 3]], "HapMap marker data")

    # Genotype block: rows = SNPs, cols = individuals
    geno_block = np.char.upper(
        np.asarray(raw.iloc[:, n_meta:].values, dtype=str)
    )  # (n_snps, n_individuals)

    n_snps, n_indiv = geno_block.shape

    # Convert each SNP row to numeric
    GD_T = np.full((n_snps, n_indiv), np.nan, dtype=np.float64)
    for i in range(n_snps):
        GD_T[i, :] = _numericalize_snp(
            geno_block[i, :],
            major_allele_zero,
            already_uppercase=True,
        )

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
    if not gd_path.exists():
        raise FileNotFoundError(f"Numeric genotype file not found: {gd_path}")

    # Read GD — GAPIT format: col 0 = taxa names, col 1+ = SNP genotypes
    gd_df = pd.read_csv(gd_path, sep="\t", low_memory=False)

    # Read GM
    if isinstance(gm_path, pd.DataFrame):
        gm_df = gm_path
    else:
        marker_path = Path(gm_path)
        if not marker_path.exists():
            raise FileNotFoundError(f"Marker map file not found: {marker_path}")
        gm_df = pd.read_csv(marker_path, sep="\t", header=0)

    return _numeric_from_frames(gd_df, gm_df, impute_method, "Numeric genotype")


def impute_missing(GD: FloatMatrix, method: str = "middle") -> FloatMatrix:
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
    values = as_float_matrix(GD, name="genotype matrix").copy()
    return _impute_missing_inplace(values, method)


def _impute_missing_inplace(GD: FloatMatrix, method: str) -> FloatMatrix:
    """Impute a validated, exclusively owned floating-point matrix in place."""
    if method == "none":
        return GD
    if method not in {"middle", "major", "minor", "mean"}:
        raise ValueError(f"Unknown impute method: {method}")

    missing = np.isnan(GD)

    if method == "middle":
        GD[missing] = 1.0
    elif method == "major":
        GD[missing] = 2.0
    elif method == "minor":
        GD[missing] = 0.0
    elif method == "mean":
        observed_count = np.sum(~missing, axis=0)
        col_means: FloatVector = np.ones(GD.shape[1], dtype=np.float64)
        np.divide(
            np.nansum(GD, axis=0),
            observed_count,
            out=col_means,
            where=observed_count > 0,
        )
        np.copyto(GD, col_means[np.newaxis, :], where=missing)

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
    if not fp.exists():
        raise FileNotFoundError(f"Phenotype file not found: {fp}")
    df = pd.read_csv(fp, sep="\t", na_values=["NA", "NaN", "nan", "N/A"])
    return _phenotype_from_frame(df, "Phenotype data")


def align_inputs(
    pheno: PhenotypeData,
    geno: GenotypeData,
    cv_df: pd.DataFrame | None = None,
    ki_df: pd.DataFrame | None = None,
) -> AlignedData:
    """
    Align all input datasets to common taxa in phenotype order.
    Translates GAPIT.IC.R and GAPIT.QC.R taxa-matching logic.

    GAPIT rule: only the intersection of taxa across all provided
    datasets is kept. Taxa names are case-sensitive.

    Returns
    -------
    :class:`AlignedData`
    """
    if len(pheno.Y) != len(pheno.taxa):
        raise ValueError("Phenotype rows and phenotype taxa must have equal length")
    if geno.GD.ndim != 2:
        raise ValueError("Genotype matrix must be two-dimensional")
    if geno.GD.shape[0] != len(geno.taxa):
        raise ValueError("Genotype rows and genotype taxa must have equal length")
    if geno.GD.shape[1] != len(geno.GM):
        raise ValueError("Genotype columns and marker-map rows must have equal length")

    phenotype_index = _unique_taxa_index(pheno.taxa, "phenotype")
    genotype_index = _unique_taxa_index(geno.taxa, "genotype")
    common = set(phenotype_index) & set(genotype_index)

    covariate_index: dict[str, int] | None = None
    covariate_values: FloatMatrix | None = None

    if cv_df is not None:
        if cv_df.shape[1] < 2:
            raise ValueError(
                "Covariate data must contain taxa and at least one covariate"
            )
        covariate_taxa = np.asarray(cv_df.iloc[:, 0].astype(str), dtype=str)
        covariate_index = _unique_taxa_index(covariate_taxa, "covariates")
        covariate_values = cv_df.iloc[:, 1:].to_numpy(dtype=float)
        common &= set(covariate_index)

    kinship_index: dict[str, int] | None = None
    kinship_values: FloatMatrix | None = None

    if ki_df is not None:
        kinship_taxa = np.asarray(ki_df.iloc[:, 0].astype(str), dtype=str)
        kinship_index = _unique_taxa_index(kinship_taxa, "kinship")
        kinship_values = ki_df.iloc[:, 1:].to_numpy(dtype=float)
        expected_shape = (len(kinship_taxa), len(kinship_taxa))
        if kinship_values.shape != expected_shape:
            raise ValueError(
                "Kinship data must contain one taxa column followed by a square "
                f"matrix; expected {expected_shape}, got {kinship_values.shape}"
            )
        common &= set(kinship_index)

    if not common:
        raise ValueError(
            "No common taxa found across input files. "
            "Check that taxa names match exactly (case-sensitive) "
            "across phenotype, genotype, and any kinship/covariate files."
        )

    common_taxa = [str(taxon) for taxon in pheno.taxa if str(taxon) in common]
    n_common = len(common_taxa)

    if n_common < len(pheno.taxa):
        warnings.warn(
            f"Kept {n_common}/{len(pheno.taxa)} taxa after alignment. "
            f"Dropped {len(pheno.taxa) - n_common} due to missing data."
        )

    taxa_arr = np.asarray(common_taxa, dtype=str)

    pheno_idx = [phenotype_index[taxon] for taxon in common_taxa]
    Y_aligned = pheno.Y.iloc[pheno_idx].reset_index(drop=True)
    geno_idx = [genotype_index[taxon] for taxon in common_taxa]
    GD_aligned = geno.GD[geno_idx, :]

    KI_aligned = None
    if kinship_index is not None and kinship_values is not None:
        ki_idx = [kinship_index[taxon] for taxon in common_taxa]
        KI_aligned = kinship_values[np.ix_(ki_idx, ki_idx)]

    CV_aligned = None
    if covariate_index is not None and covariate_values is not None:
        cv_idx = [covariate_index[taxon] for taxon in common_taxa]
        CV_aligned = covariate_values[cv_idx, :]

    return AlignedData(
        taxa=taxa_arr,
        phenotypes=Y_aligned,
        genotypes=GD_aligned,
        markers=geno.GM,
        kinship=KI_aligned,
        covariates=CV_aligned,
    )


def align_taxa(
    pheno: PhenotypeData,
    geno: GenotypeData,
    cv_df: pd.DataFrame | None = None,
    ki_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the legacy mapping form of :func:`align_inputs`."""
    return align_inputs(pheno, geno, cv_df=cv_df, ki_df=ki_df).as_legacy_dict()


def maf_filter(
    GD: FloatMatrix, threshold: float = 0.05
) -> tuple[FloatMatrix, IntVector]:
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
