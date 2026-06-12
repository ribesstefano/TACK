"""
Submit a SLURM job array to parse TPDdb HTML files in parallel.

Reads TPD IDs from a delimited table, splits them into N chunks, writes
a staging ID file, generates an sbatch array script, and optionally submits it.

Usage examples:
  # Basic submission (reads PROTAC_main_table.txt, 50 parallel jobs)
  python tack_dataset/tpddb_parsing_slurm.py

  # Dry-run: print the generated sbatch script without submitting
  python tack_dataset/tpddb_parsing_slurm.py --dry-run

  # Custom cluster settings
  python tack_dataset/tpddb_parsing_slurm.py \\
      --jobs 100 --account myaccount --partition main --time 4:00:00

  # Parse Molecular Glues instead of PROTACs
  python tack_dataset/tpddb_parsing_slurm.py \\
      --input data/original/MG_main_table.csv --mol-type MG --sep ','
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tpd_ids(input_path: Path, id_col: int = 0, sep: str = '\t') -> list:
    """Return a deduplicated, ordered list of TPD IDs from a delimited file."""
    ids = []
    seen = set()
    with open(input_path, 'r', encoding='utf-8') as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split(sep)
            if not parts:
                continue
            tpd_id = parts[id_col].strip()
            if tpd_id.startswith('TPD-') and tpd_id not in seen:
                ids.append(tpd_id)
                seen.add(tpd_id)
    return ids


def build_sbatch_script(
    *,
    ids_file: Path,
    total_ids: int,
    n_jobs: int,
    workdir: Path,
    venv: str,
    parsing_script: str,
    html_dir: str,
    save_dir: str,
    log_dir: str,
    mol_type: str,
    skip_existing: bool,
    verbose: int,
    # SLURM directives
    job_name: str,
    account: str,
    partition: str,
    time_limit: str,
    mail_user: str,
    mail_type: str,
    cpus_per_task: int,
    mem: str,
    extra_directives: list,
) -> str:
    """Return the text of the SLURM sbatch array script."""
    ids_per_job = math.ceil(total_ids / n_jobs)
    actual_jobs = math.ceil(total_ids / ids_per_job)  # may be < n_jobs for small sets

    # --- SLURM directives ---
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{actual_jobs - 1}",
        f"#SBATCH --output={log_dir}/tpddb_parsing_%A_%a.out",
        f"#SBATCH --error={log_dir}/tpddb_parsing_%A_%a.err",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH -n 1",
    ]
    if account:
        directives.append(f"#SBATCH --account={account}")
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if cpus_per_task and cpus_per_task > 1:
        directives.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    if mem:
        directives.append(f"#SBATCH --mem={mem}")
    if mail_user:
        directives.append(f"#SBATCH --mail-user={mail_user}")
    if mail_type:
        directives.append(f"#SBATCH --mail-type={mail_type}")
    for extra in extra_directives:
        directives.append(f"#SBATCH {extra}")

    # --- Python call flags ---
    extra_flags = []
    if verbose > 0:
        extra_flags.append('-' + 'v' * verbose)
    if skip_existing:
        extra_flags.append('--skip-existing')
    if mol_type:
        extra_flags.append(f'--mol-type {mol_type}')
    extra_flags_str = ' \\\n    '.join(extra_flags)
    if extra_flags_str:
        extra_flags_str = ' \\\n    ' + extra_flags_str

    activate = f'source "{venv}/bin/activate"' if venv else '# (no virtual environment specified)'

    script = f"""\
#!/bin/bash
{chr(10).join(directives)}

# --- Environment setup ---
cd "{workdir}"
{activate}

# --- Compute the ID range for this array task ---
IDS_PER_JOB={ids_per_job}
TASK_ID=$SLURM_ARRAY_TASK_ID
TOTAL_IDS={total_ids}

START=$(( TASK_ID * IDS_PER_JOB + 1 ))
END=$(( (TASK_ID + 1) * IDS_PER_JOB ))
[ "$END" -gt "$TOTAL_IDS" ] && END=$TOTAL_IDS

# Read the chunk of IDs from the staging file (awk uses 1-based line numbers)
IDS=$(awk "NR>=$START && NR<=$END" "{ids_file}" | tr '\\n' ' ')

if [ -z "${{IDS// /}}" ]; then
    echo "[task $TASK_ID] No IDs in range $START-$END — exiting."
    exit 0
fi

echo "[task $TASK_ID] Processing IDs $START-$END of $TOTAL_IDS"

python "{parsing_script}" \\
    --html-dir "{html_dir}" \\
    --save-dir "{save_dir}" \\
    --log-dir "{log_dir}"{extra_flags_str} \\
    --ids $IDS
"""
    return script


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    parser = argparse.ArgumentParser(
        description='Submit a SLURM job array for parallel TPDdb HTML parsing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input / parsing options ---
    io_group = parser.add_argument_group('Input / parsing options')
    io_group.add_argument(
        '--input', type=Path,
        default=repo_root / 'data' / 'original' / 'PROTAC_main_table.txt',
        help='Delimited file containing TPD IDs (default: data/original/PROTAC_main_table.txt)',
    )
    io_group.add_argument(
        '--sep', default='\t',
        help='Column separator of --input (default: tab)',
    )
    io_group.add_argument(
        '--id-col', type=int, default=0,
        help='Zero-based column index of TPD IDs (default: 0)',
    )
    io_group.add_argument(
        '--html-dir', type=Path, default=repo_root / 'data' / 'html',
        help='Directory containing downloaded HTML files (default: data/html)',
    )
    io_group.add_argument(
        '--save-dir', type=Path, default=repo_root / 'data' / 'parsed',
        help='Directory to write parsed CSVs (default: data/parsed)',
    )
    io_group.add_argument(
        '--log-dir', type=Path, default=repo_root / 'logs',
        help='Directory for SLURM stdout/stderr and Python logs (default: logs)',
    )
    io_group.add_argument(
        '--mol-type',
        choices=['AUTOTAC', 'LYTAC', 'ATTEC', 'AUTAC', 'PROTAC', 'MG'],
        default='PROTAC',
        help='Molecule type passed to tpddb_parsing.py (default: PROTAC)',
    )
    io_group.add_argument(
        '--skip-existing', action='store_true',
        help='Pass --skip-existing to tpddb_parsing.py',
    )
    io_group.add_argument(
        '--verbose', '-v', action='count', default=0,
        help='Verbosity passed to tpddb_parsing.py (-v INFO, -vv DEBUG)',
    )
    io_group.add_argument(
        '--parsing-script', type=Path,
        default=here / 'tpddb_parsing.py',
        help='Path to tpddb_parsing.py (default: sibling of this file)',
    )

    # --- SLURM options ---
    slurm_group = parser.add_argument_group('SLURM options')
    slurm_group.add_argument(
        '--jobs', type=int, default=50,
        help='Number of parallel SLURM array tasks (default: 50)',
    )
    slurm_group.add_argument(
        '--job-name', default='TPDdb-Parsing',
        help='SLURM job name (default: TPDdb-Parsing)',
    )
    slurm_group.add_argument(
        '--account', default='',
        help='SLURM account/project (e.g. naiss2025-5-462)',
    )
    slurm_group.add_argument(
        '--partition', default='',
        help='SLURM partition (e.g. tetralith, standard)',
    )
    slurm_group.add_argument(
        '--time', dest='time_limit', default='2:00:00',
        help='Wall-clock time limit per task (default: 2:00:00)',
    )
    slurm_group.add_argument(
        '--cpus-per-task', type=int, default=1,
        help='CPUs per array task (default: 1)',
    )
    slurm_group.add_argument(
        '--mem', default='',
        help='Memory per task, e.g. 4G (default: cluster default)',
    )
    slurm_group.add_argument(
        '--mail-user', default='',
        help='Email address for SLURM notifications',
    )
    slurm_group.add_argument(
        '--mail-type', default='END,FAIL',
        help='SLURM mail event types (default: END,FAIL)',
    )
    slurm_group.add_argument(
        '--extra-directive', dest='extra_directives',
        action='append', default=[],
        metavar='DIRECTIVE',
        help='Additional raw SLURM directives, e.g. \'--constraint=fat\' (repeatable)',
    )

    # --- Execution options ---
    exec_group = parser.add_argument_group('Execution options')
    exec_group.add_argument(
        '--workdir', type=Path, default=repo_root,
        help='Working directory for SLURM tasks (default: repo root)',
    )
    exec_group.add_argument(
        '--venv', default='',
        help='Path to virtual environment to activate (e.g. .venv)',
    )
    exec_group.add_argument(
        '--staging-file', type=Path, default=None,
        help='Where to write the staging ID list (default: <log-dir>/tpd_ids_staging.txt)',
    )
    exec_group.add_argument(
        '--script-out', type=Path, default=None,
        help='Where to write the generated sbatch script (default: <log-dir>/tpddb_parsing_array.sh)',
    )
    exec_group.add_argument(
        '--dry-run', action='store_true',
        help='Print the sbatch script without submitting',
    )

    args = parser.parse_args()

    # --- Resolve paths ---
    log_dir = args.log_dir.resolve()
    staging_file = (args.staging_file or log_dir / 'tpd_ids_staging.txt').resolve()
    script_out = (args.script_out or log_dir / 'tpddb_parsing_array.sh').resolve()

    # --- Read IDs ---
    if not args.input.exists():
        sys.exit(f"Error: input file not found: {args.input}")
    ids = read_tpd_ids(args.input, id_col=args.id_col, sep=args.sep)
    if not ids:
        sys.exit("Error: no TPD IDs found in input file.")
    print(f"Found {len(ids):,} TPD IDs in {args.input}")

    # --- Write staging file ---
    log_dir.mkdir(parents=True, exist_ok=True)
    staging_file.write_text('\n'.join(ids) + '\n', encoding='utf-8')
    print(f"Staging ID file written: {staging_file}")

    # --- Build sbatch script ---
    script_text = build_sbatch_script(
        ids_file=staging_file,
        total_ids=len(ids),
        n_jobs=args.jobs,
        workdir=args.workdir.resolve(),
        venv=args.venv,
        parsing_script=str(args.parsing_script.resolve()),
        html_dir=str(args.html_dir.resolve()),
        save_dir=str(args.save_dir.resolve()),
        log_dir=str(log_dir),
        mol_type=args.mol_type,
        skip_existing=args.skip_existing,
        verbose=args.verbose,
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        time_limit=args.time_limit,
        mail_user=args.mail_user,
        mail_type=args.mail_type,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        extra_directives=args.extra_directives,
    )

    # --- Write / print the script ---
    if args.dry_run:
        print("\n--- Generated sbatch script (dry run) ---\n")
        print(script_text)
        return

    script_out.write_text(script_text, encoding='utf-8')
    script_out.chmod(0o755)
    print(f"sbatch script written: {script_out}")

    # --- Submit ---
    result = subprocess.run(
        ['sbatch', str(script_out)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(result.stdout.strip())
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
