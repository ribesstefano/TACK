import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from datasets import load_dataset

from tackai import EnsemblePredictor, SampleInput


def test_ensemble_predictor():
    
    predictor = EnsemblePredictor.from_directory("ensembles/dc50_ensemble/")

    test_df = load_dataset(
        "ailab-bio/TACK",
        "DC50",
        split="train",
    ).to_pandas()

    # test_df = pd.read_csv("data/tack/protacdb_tpddb_protacpedia_protac_dc50_activities_processed.csv")
    test_df[['Value_Type', 'Value']].dropna(subset=['Value']).sample(1, random_state=42)

    test_df['Cell_Line_ID'] = test_df['Cell_Line_ID'].fillna('Unknown cell line.')
    test_df['Assay'] = test_df['Assay'].fillna('Unknown')

    test_df = test_df.iloc[:1000]  # <-- Limit to first 500 rows for testing

    batch_size = 32
    metrics_df = []

    for batch_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
        batch = test_df.iloc[batch_idx:min(batch_idx + batch_size, len(test_df))]

        # Build (sample, true_value) pairs, skipping rows with missing Value
        batch_samples = []
        batch_true_values = []
        for _, row in batch.iterrows():
            if pd.isna(row['Value']):
                continue
            sample = SampleInput(
                smiles=row['SMILES'],
                poi_name=row['POI_Name'],
                poi_sequence=row['POI_Sequence'],
                ligase_name=row['Ligase_Name'],
                cell_line=row['Cell_Line_ID'],
                treatment_time=row['Assay_Time'],
                assay_type=row['Assay'],
            )
            batch_samples.append(sample)
            batch_true_values.append(row['Value'])

        if not batch_samples:
            continue

        batch_results = predictor.predict(batch_samples, tasks=['dc50'])

        # Compare predictions with true values
        for sample, true_dc50, result in zip(batch_samples, batch_true_values, batch_results):
            result = result['dc50'].to_dict()
            metrics_df.append({
                # 'SMILES': sample.smiles,
                'True_DC50': true_dc50,
                'Predicted_DC50': result['weighted_mean'][0],
                'CI_Lower_95': result['ci_percentile_lower_95'][0],
                'CI_Upper_95': result['ci_percentile_upper_95'][0],
            })

    metrics_df = pd.DataFrame(metrics_df)

    # Calculate MAE and CI coverage
    metrics_df['Absolute_Error'] = (metrics_df['True_DC50'] - metrics_df['Predicted_DC50']).abs()
    mae = metrics_df['Absolute_Error'].mean()
    within_ci = ((metrics_df['True_DC50'] >= metrics_df['CI_Lower_95']) & (metrics_df['True_DC50'] <= metrics_df['CI_Upper_95'])).mean()
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Percentage of true DC50 values within 95% CI: {within_ci:.2%}")

    def dc50_to_pdc50(dc50):
        """Convert DC50 to pDC50 (negative log10)"""
        return -np.log10(dc50)

    metrics_df['True_pDC50'] = metrics_df['True_DC50'].apply(dc50_to_pdc50)
    metrics_df['Predicted_pDC50'] = metrics_df['Predicted_DC50'].apply(dc50_to_pdc50)
    metrics_df['CI_Lower_95_pDC50'] = metrics_df['CI_Lower_95'].apply(dc50_to_pdc50)
    metrics_df['CI_Upper_95_pDC50'] = metrics_df['CI_Upper_95'].apply(dc50_to_pdc50)

    print(metrics_df.head(30).to_markdown(index=False))

    mae = (metrics_df['True_pDC50'] - metrics_df['Predicted_pDC50']).abs().mean()
    r2 = 1 - ((metrics_df['True_pDC50'] - metrics_df['Predicted_pDC50']) ** 2).sum() / ((metrics_df['True_pDC50'] - metrics_df['True_pDC50'].mean()) ** 2).sum()
    within_ci = ((metrics_df['True_pDC50'] >= metrics_df['CI_Lower_95_pDC50']) & (metrics_df['True_pDC50'] <= metrics_df['CI_Upper_95_pDC50'])).mean()
    print(f"Mean Absolute Error (MAE) in pDC50 space: {mae:.4f}")
    print(f"R² in pDC50 space: {r2:.4f}")
    print(f"Percentage of true pDC50 values within 95% CI: {within_ci:.2%}")

    # Get the MAE for a dummy mean predictor in pDC50 space for comparison
    metrics_df['Mean_Predicted_pDC50'] = metrics_df['True_pDC50'].mean()  # Dummy predictor that always predicts the mean true pDC50
    mae_dummy = (metrics_df['True_pDC50'] - metrics_df['Mean_Predicted_pDC50']).abs().mean()
    r2_dummy = 1 - ((metrics_df['True_pDC50'] - metrics_df['Mean_Predicted_pDC50']) ** 2).sum() / ((metrics_df['True_pDC50'] - metrics_df['True_pDC50'].mean()) ** 2).sum()
    print(f"Mean Absolute Error (MAE) of dummy mean predictor in pDC50 space: {mae_dummy:.4f}")
    print(f"R² of dummy mean predictor in pDC50 space: {r2_dummy:.4f}")
    
    assert r2 > 0 and r2 > r2_dummy, f"The ensemble model shall score a R2 higher than a dummy model, got instead: R2 ensemble = {r2} vs. R2 dummy = {r2_dummy}"


if __name__ == "__main__":
    test_ensemble_predictor()