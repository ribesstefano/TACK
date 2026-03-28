import os
import random
from typing import Optional, List
import logging
import argparse
import time
import requests
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from tack_dataset.logging_utils import setup_logging


# TPDdb URL template
TPDDB_URL_TEMPLATE = "https://tpddb.idrblab.net/data/tpd/details/{tpd_id}"


def fetch_html_from_web(
        tpd_id: str,
        save_to_file: Optional[Path] = None,
        timeout: float = 30,
) -> Optional[str]:
    """ Fetch HTML content from TPDdb website for a given TPD ID.
    
    Args:
        tpd_id: TPD identifier (e.g., 'TPD-RLQHR3')
        save_to_file: Optional path to save the HTML file
        timeout: Request timeout in seconds
    
    Returns:
        HTML content as string, or None if fetch failed
    """
    url = TPDDB_URL_TEMPLATE.format(tpd_id=tpd_id)
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug(f"Fetching HTML from: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        html_content = response.text
        
        # Optionally save to file
        if save_to_file:
            os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
            with open(save_to_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.debug(f"Saved HTML to: {save_to_file}")
        
        return html_content
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching HTML for {tpd_id}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch HTML for {tpd_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching HTML for {tpd_id}: {e}")
        return None


def batch_scrape(
    protac_txt_filepath,
    limit: Optional[int] = None,
    skip_existing: bool = False,
    specific_ids: Optional[List[str]] = None,
    html_base_dir: Optional[Path] = None,
):
    """
    Download all TPD IDs from PROTAC_main_table.txt.
    
    Args:
        limit: Optional limit on number of entries to process
        skip_existing: If True, skip TPD IDs already in output CSV
        save_html: If True, save fetched HTML files to data/html/
        specific_ids: Optional list of specific TPD IDs to process
        html_base_dir: Optional base directory to save HTML files (default: data/html/)
    """    
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("BATCH EXTRACTION STARTED")
    logger.info("="*80)
    logger.info(f"Mode: Fetch HTML from TPDdb website")
    # logger.info(f"Total TPD IDs in CSV: {len(protac_df)}")
    if specific_ids:
        logger.info(f"Processing specific IDs: {', '.join(specific_ids)}")
    if limit:
        logger.info(f"Processing limit: {limit} entries")
    if skip_existing:
        logger.info("Skip existing: Enabled")
    logger.info("="*80)
    
    # Get list of TPD IDs to process
    if specific_ids:
        tpd_ids = specific_ids
    else:
        # Read TPD IDs from the PROTAC_main_table.txt file
        tpd_ids = []
        with open(protac_txt_filepath, "r") as f:
            lines = f.readlines()[1:]

            for line in lines:
                tpd_id = line.strip().split("\t")[0].strip()
                if not tpd_id.startswith("TPD-"):
                    continue
                tpd_ids.append(tpd_id)

        if limit:
            tpd_ids = tpd_ids[:limit]
            logger.info(f"Limiting to first {limit} entries of the dataset.")
        else:
            logger.info(f"Processing all {len(tpd_ids)} entries from the dataset.")

    # Load HTML directory from .env
    if html_base_dir is None:
        html_base_dir = Path(os.getenv('HTML_SAVE_DIR', 'data/html'))
        # Create directory if it doesn't exist
        html_base_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"HTML base directory set to: {html_base_dir}")

    # Statistics
    total = len(tpd_ids)
    success = 0
    skipped = 0
    fetch_failed = 0
    failed = 0
    
    logger.info(f"\nProcessing {total} TPD IDs...")
    print(f"\nProcessing {total} TPD IDs...")
    print("="*80)
    
    for idx, tpd_id in enumerate(tpd_ids, 1):
        logger.info(f"\n[{idx}/{total}] Processing: {tpd_id}")
        print(f"\n[{idx}/{total}] Processing: {tpd_id}")
        
        # Skip existing HTML files if requested
        save_path = html_base_dir / f"{tpd_id}.html"
        if skip_existing and save_path.exists() and save_path.is_file():
            logger.info(f"⊘ Skipping {tpd_id} (HTML file already exists)")
            print(f"  ⊘ Skipped (HTML file already exists)")
            skipped += 1
            continue
        
        # Small delay to be nice to the server
        time.sleep(1 + random.uniform(0, 5))
        logger.info(f"Fetching HTML from TPDdb website...")
        html_content = fetch_html_from_web(tpd_id, save_to_file=save_path)

        # Check fetch result and update stats
        if not html_content:
            logger.warning(f"⚠ Failed to fetch HTML for {tpd_id}")
            print(f"  ⚠ Failed to fetch HTML")
            fetch_failed += 1
            continue
        
        logger.info(f"✓ Successfully fetched HTML ({len(html_content)} bytes)")
        success += 1
        
        # Progress update every 10 entries
        if idx % 10 == 0:
            logger.info(f"Progress: {idx}/{total} ({(idx/total)*100:.1f}%)")
            print(f"\nProgress: {idx}/{total} ({(idx/total)*100:.1f}%) - Success: {success}, Failed: {failed}, Fetch Failed: {fetch_failed}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("BATCH EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total TPD IDs: {total}")
    logger.info(f"Successfully processed: {success}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"Failed to fetch HTML: {fetch_failed}")
    logger.info(f"Extraction failed: {failed}")
    logger.info(f"HTML files saved to: {html_base_dir}")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("BATCH EXTRACTION COMPLETE")
    print("="*80)
    print(f"Total TPD IDs: {total}")
    print(f"Successfully processed: {success}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed to fetch HTML: {fetch_failed}")
    print(f"Extraction failed: {failed}")
    print(f"HTML files saved to: {html_base_dir}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='TPDdb Batch Download - Fetch all requested TPD IDs HTML from TPDdb website.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Process all TPD IDs (fetches HTML from web)
  python scraping.py
  
  # Specify the directory to save HTML files
  python scraping.py --html-save-dir data/html
  
  # Skip already processed entries (resume)
  python scraping.py --skip-existing
  
  # Test specific IDs to verify DC50/IC50/EC50/Dmax extraction
  python scraping.py --ids TPD-O7725Z TPD-LIZKQI TPD-NMQ9M9
        """
    )
    
    parser.add_argument(
        '--protac-txt',
        type=Path,
        default=Path('data/original/PROTAC_main_table.txt'),
        help='Path to PROTAC_main_table.txt file (default: data/original/PROTAC_main_table.txt)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Process only first N entries (for testing)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip TPD IDs already in output CSV'
    )
    parser.add_argument(
        '--html-save-dir',
        type=Path,
        default=Path('data/html'),
        help='Directory to save fetched HTML files (default: data/html)'
    )
    parser.add_argument(
        '--ids',
        nargs='+',
        metavar='TPD_ID',
        help='Process only specific TPD IDs (space-separated list, e.g., --ids TPD-O7725Z TPD-LIZKQI)'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs/'),
        help='Directory to save log files (default: logs/)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='count',
        default=0,
        help='Increase verbosity level (use -vv for DEBUG, -v for INFO)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    log_file = setup_logging(
        log_dir=args.log_dir,
        log_base_name='tpddb_scraping',
        verbose=args.verbose
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Log file: {log_file}")
    print(f"\nTPDdb Extraction System")
    print(f"=" * 80)
    print(f"Mode: Fetch HTML from TPDdb website")
    print(f"Log file: {log_file}")
    print(f"HTML files will be saved to: {args.html_save_dir}")
    print(f"=" * 80)
    
    # Run batch extraction
    batch_scrape(
        protac_txt_filepath=args.protac_txt,
        limit=args.limit,
        skip_existing=args.skip_existing,
        specific_ids=args.ids
    )

if __name__ == "__main__":
    main()
