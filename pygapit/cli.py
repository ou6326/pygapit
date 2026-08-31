"""
Command-line interface for pyGAPIT.

Usage
-----
    pygapit --Y pheno.txt --GD geno.txt --GM map.txt --model BLINK
    pygapit --Y pheno.txt --GD geno.txt --GM map.txt --model MLM FarmCPU BLINK
    pygapit --Y pheno.txt --G hapmap.hmp.txt --model BLINK --PCA_total 5
    pygapit --Y pheno.txt --GD geno.txt --GM map.txt --model gBLUP
"""

import argparse
from pathlib import Path

from pygapit.gapit import GAPITResult


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pygapit",
        description="pyGAPIT: Genome Association and Prediction Integrated Tool (Python)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BLINK GWAS (default, highest power):
  pygapit --Y traits.txt --GD geno.txt --GM map.txt --model BLINK

  # Multiple models:
  pygapit --Y traits.txt --GD geno.txt --GM map.txt --model GLM MLM BLINK FarmCPU

  # Genomic prediction:
  pygapit --Y traits.txt --GD geno.txt --GM map.txt --model gBLUP

  # HapMap format:
  pygapit --Y traits.txt --G genotype.hmp.txt --model BLINK

  # Custom settings:
  pygapit --Y traits.txt --GD geno.txt --GM map.txt \\
          --model BLINK --PCA_total 5 --maf_threshold 0.01 --output_dir results/
        """,
    )

    # Input files
    io_group = parser.add_argument_group("Input data")
    io_group.add_argument(
        "--Y", required=True, help="Phenotype file (tab-delimited, col1=Taxa)"
    )
    io_group.add_argument(
        "--GD", help="Numeric genotype file (col1=taxa, col2+=SNPs 0/1/2)"
    )
    io_group.add_argument(
        "--GM", help="SNP map file (3 cols: SNP, Chromosome, Position)"
    )
    io_group.add_argument(
        "--G", help="HapMap genotype file (alternative to --GD + --GM)"
    )
    io_group.add_argument(
        "--KI", help="Kinship matrix file (optional; computed from GD if absent)"
    )
    io_group.add_argument("--CV", help="Covariate file (optional)")

    # Model selection
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        nargs="+",
        default=["BLINK"],
        choices=[
            "GLM",
            "MLM",
            "CMLM",
            "MLMM",
            "FarmCPU",
            "BLINK",
            "gBLUP",
            "cBLUP",
        ],
        help="GWAS/GS model(s) to run (default: BLINK)",
    )
    model_group.add_argument(
        "--trait", help="Trait name or column index to analyze (default: all traits)"
    )

    # PCA / QC
    qc_group = parser.add_argument_group("Quality control & PCA")
    qc_group.add_argument(
        "--PCA_total",
        type=int,
        default=3,
        help="Number of PCs for population structure control (default: 3)",
    )
    qc_group.add_argument(
        "--maf_threshold",
        type=float,
        default=0.05,
        help="Minimum minor allele frequency (default: 0.05)",
    )
    qc_group.add_argument(
        "--SNP_impute",
        default="middle",
        choices=["middle", "major", "minor", "mean", "none"],
        help="Missing genotype imputation method (default: middle)",
    )

    # GWAS thresholds
    thresh_group = parser.add_argument_group("Significance thresholds")
    thresh_group.add_argument(
        "--cutOff",
        type=float,
        default=None,
        help="P-value threshold (default: Bonferroni 0.05/m)",
    )
    thresh_group.add_argument(
        "--LD",
        type=float,
        default=0.7,
        help="LD threshold for BLINK pruning (default: 0.7)",
    )
    thresh_group.add_argument(
        "--maxLoop",
        type=int,
        default=10,
        help="Max iterations for BLINK/FarmCPU (default: 10)",
    )

    # CMLM parameters
    cmlm_group = parser.add_argument_group("CMLM parameters")
    cmlm_group.add_argument(
        "--group_from", type=int, default=1, help="Min groups for CMLM (default: 1)"
    )
    cmlm_group.add_argument(
        "--group_to",
        type=int,
        default=None,
        help="Max groups for CMLM (default: n individuals)",
    )

    # FarmCPU parameters
    farm_group = parser.add_argument_group("FarmCPU parameters")
    farm_group.add_argument(
        "--bin_size",
        type=int,
        default=5_000_000,
        help="Bin size in bp for FarmCPU (default: 5000000)",
    )

    # Simulation
    sim_group = parser.add_argument_group("Phenotype simulation")
    sim_group.add_argument(
        "--h2",
        type=float,
        default=None,
        help="Heritability for phenotype simulation (e.g. 0.7)",
    )
    sim_group.add_argument(
        "--NQTN", type=int, default=None, help="Number of QTNs for simulation (e.g. 20)"
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output_dir",
        default=".",
        help="Output directory for results (default: current dir)",
    )
    out_group.add_argument(
        "--no_file_output",
        action="store_true",
        help="Suppress file output (only return object)",
    )
    out_group.add_argument(
        "--buspred", action="store_true", help="Run genomic prediction after GWAS"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.G is None and (args.GD is None or args.GM is None):
        parser.error(
            "Provide either --G (HapMap) or both --GD and --GM (numeric format)."
        )

    if args.h2 is not None and args.NQTN is None:
        parser.error("--NQTN is required when --h2 is provided for simulation.")

    print("=" * 60)
    print("  pyGAPIT — Genome Association & Prediction Tool (Python)")
    print("=" * 60)

    import warnings

    warnings.filterwarnings("ignore")

    import pandas as pd

    from pygapit import GAPIT

    # Load data
    print(f"\n[CLI] Loading phenotype: {args.Y}")
    Y = pd.read_csv(args.Y, sep="\t")

    GD = GM = G = None
    if args.G:
        print(f"[CLI] Loading HapMap: {args.G}")
        G = pd.read_csv(args.G, sep="\t", header=None)
    else:
        print(f"[CLI] Loading genotype: {args.GD}")
        print(f"[CLI] Loading map:      {args.GM}")
        GD = pd.read_csv(args.GD, sep="\t")
        GM = pd.read_csv(args.GM, sep="\t")

    KI = pd.read_csv(args.KI, sep="\t", header=None) if args.KI else None
    CV = pd.read_csv(args.CV, sep="\t") if args.CV else None

    print(f"[CLI] Model(s): {args.model}")
    print(f"[CLI] Output directory: {Path(args.output_dir).resolve()}")

    # Run GAPIT
    result = GAPIT(
        Y=Y,
        G=G,
        GD=GD,
        GM=GM,
        KI=KI,
        CV=CV,
        model=args.model,
        trait=args.trait,
        PCA_total=args.PCA_total,
        maf_threshold=args.maf_threshold,
        SNP_impute=args.SNP_impute,
        cutOff=args.cutOff,
        LD=args.LD,
        maxLoop=args.maxLoop,
        group_from=args.group_from,
        group_to=args.group_to,
        bin_size=args.bin_size,
        h2=args.h2,
        NQTN=args.NQTN,
        file_output=not args.no_file_output,
        output_dir=args.output_dir,
        buspred=args.buspred,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    if isinstance(result, dict):
        for key, r in result.items():
            _print_summary(key, r)
    else:
        _print_summary(f"{result.trait} / {result.model}", result)

    print("\n[CLI] Done.")


def _print_summary(label: str, result: GAPITResult) -> None:
    print(f"\n  {label}")
    print(f"    h^2     = {result.h2:.4f}")
    print(f"    lambda (GC) = {result.lambda_gc:.4f}")
    if result.GWAS is not None:
        print(f"    SNPs    = {len(result.GWAS):,}")
    if result.significant is not None and len(result.significant) > 0:
        print(f"    Sig SNPs = {len(result.significant)} (Bonferroni)")
        top = result.significant.nsmallest(3, "P.value")
        for _, row in top.iterrows():
            print(
                f"      {row['SNP']}  chr{row['Chr']}:{int(float(str(row['Pos']))):,}  "
                f"p={row['P.value']:.2e}  effect={row['effect']:.4f}"
            )
    else:
        print("    Sig SNPs = 0 (no Bonferroni-significant hits)")
    if result.QTNs is not None and len(result.QTNs) > 0:
        print(f"    QTNs selected = {len(result.QTNs)}")
    if result.runtime_seconds > 0:
        print(f"    Runtime = {result.runtime_seconds:.1f}s")


if __name__ == "__main__":
    main()
