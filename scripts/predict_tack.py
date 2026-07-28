import argparse

from tackai.ensemble_predictor import EnsemblePredictor, SampleInput


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SLURM submission script.

    Returns:
        Populated ``argparse.Namespace``.
    """
    ap = argparse.ArgumentParser(
        description="Submit a SLURM job array for AiZynthFinder component scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    slurm = ap.add_argument_group("SLURM")
    slurm.add_argument("--account",   required=True,
                       help="SLURM account (e.g. berzelius-2026-62).")
    slurm.add_argument("--partition", default="berzelius-cpu",
                       help="SLURM partition.")
    slurm.add_argument("--time",      default="08:00:00",
                       help="Wall-clock time limit per task.")
    slurm.add_argument("--cpus",      type=int, default=16,
                       help="CPUs per task (--cpus-per-task). CPU nodes have 128 cores (2×AMD EPYC 9534); "
                            "16 fits 8 concurrent tasks per node and saturates TF intra-op parallelism.")
    slurm.add_argument("--mem",       default="32G",
                       help="Memory per task.")
    slurm.add_argument("--job-name",  default="comp_scores",
                       help="SLURM job name.")
    slurm.add_argument("--log-dir",   type=Path, default=REPO_ROOT / "logs" / "component_scores",
                       help="Directory for SLURM stdout/stderr logs and the task list/script.")
    slurm.add_argument("--mail",      default=None,
                       help="Email address for END/FAIL notifications.")

    scorer = ap.add_argument_group("scorer (forwarded to get_component_routes_and_scores.py)")
    scorer.add_argument("--input", required=True, type=Path, help="Input CSV.")
    scorer.add_argument("--output", required=True, type=Path, help="Output CSV.")
    scorer.add_argument("--chunk_size", type=int, default=0,
                        help="Split slices with more unique SMILES than this into chunks (0 = no split).")
    scorer.add_argument("--smiles_col", default="cap_smiles",
                        help="SMILES column in the capped CSV.")
    scorer.add_argument("--error_col",  default="error",
                        help="Error column in the capped CSV.")

    ap.add_argument("--dry-run", action="store_true",
                    help="Write the sbatch script and print it without submitting.")
    return ap.parse_args()


def main() -> None:
    """Build the task list and submit (or dry-run) the SLURM job array."""
    args = parse_args()

if __name__ == "__main__":
    main()