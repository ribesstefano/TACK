""" Cleaning and standardizing activity data from TPDdb 

Example of usage:

>>> from cleaning import clean_activity_data
>>> cleaned_df = clean_activity_data('path/to/activities.csv')

"""
import pandas as pd
import re


def clean_activity_data(df_or_path):
    """
    Clean and standardize activity data from TPDdb.
    
    Args:
        df_or_path: Either a pandas DataFrame or path to CSV file containing Type and Value columns
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with additional columns (original columns preserved)
    """
    # Handle both DataFrame and file path inputs
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()  # Make a copy to avoid modifying the original
    else:
        df = pd.read_csv(df_or_path)

    if 'Type' in df.columns:
        df = clean_type_column(df)
        df = normalize_units(df, base_col='Type')
    
    if 'Value' in df.columns:
        df = clean_value_column(df)
        df = normalize_units(df, base_col='Value')
    
    if 'Cell_Line' in df.columns:
        df = clean_cell_line_column(df)
        df = normalize_units(df, base_col='Cell_Line')
    
    if 'Type' in df.columns:
        df = standardize_type_base(df)
    
    return df

def clean_type_column(df):
    """Extract base type, numeric suffix, and concentration with unit from Type column"""
    
    def parse_type(type_str):
        if pd.isna(type_str):
            return None, None, None, None
        
        type_str = str(type_str).strip()
        
        # Initialize return values
        concentration = None
        concentration_unit = None
        
        # Pattern 1: "Type(concentration)" or "Type_variant(concentration)"
        # Handles: IC50(1μM), Inhibition(100nM), Dmax_1(5μM), etc.
        conc_match = re.search(r'\((\d+(?:\.\d+)?)\s*([nμµ]?M)\)', type_str)
        if conc_match:
            concentration = float(conc_match.group(1))
            concentration_unit = conc_match.group(2)
        
        # Pattern 2: "Type_variant(concentration unit Compound Concentration)"
        # Handles: Normalized BTK levels_2(10 μM Compound Concentration)
        if not concentration:
            conc_match = re.search(r'\((\d+(?:\.\d+)?)\s*([μµ]M)\s+Compound\s+Concentration\)', type_str)
            if conc_match:
                concentration = float(conc_match.group(1))
                concentration_unit = conc_match.group(2)
        
        # Pattern 3: "Type under concentration unit treatment"
        # Handles: GFP level under 0.01 μM treatment
        if not concentration:
            conc_match = re.search(r'under\s+(\d+(?:\.\d+)?)\s*([μµ]M)\s+treatment', type_str)
            if conc_match:
                concentration = float(conc_match.group(1))
                concentration_unit = conc_match.group(2)
        
        # Pattern 4: "% degradation at 1 μM"
        if not concentration:
            conc_match = re.search(r'at\s+(\d+(?:\.\d+)?)\s*([a-zA-Zμ%]+)', type_str, re.IGNORECASE)
            if conc_match:
                concentration = float(conc_match.group(1))
                concentration_unit = conc_match.group(2)
                # Remove the concentration part from type_str
                type_str = re.sub(r'\s+at\s+\d+(?:\.\d+)?\s*[a-zA-Zμ%]+', '', type_str, flags=re.IGNORECASE)
        
        # Pattern 5: "100nM Degradation"
        if not concentration:
            conc_match = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Zμ]+)\s+(\w+)', type_str, re.IGNORECASE)
            if conc_match and 'degradation' in type_str.lower():
                concentration = float(conc_match.group(1))
                concentration_unit = conc_match.group(2)
                # Remove the concentration part
                type_str = re.sub(r'^\d+(?:\.\d+)?\s*[a-zA-Zμ]+\s+', '', type_str, flags=re.IGNORECASE)
        
        # Pattern 6: "% DEGRADATION (100nM)" - extract from parentheses for degradation types
        if not concentration:
            conc_match = re.search(r'\((\d+(?:\.\d+)?)\s*([a-zA-Zμ%]+)\)', type_str)
            if conc_match and 'degradation' in type_str.lower():
                concentration = float(conc_match.group(1))
                concentration_unit = conc_match.group(2)
        
        # Pattern 7: "Degradation percentage(25nM)"
        if not concentration:
            conc_match = re.search(r'\((\d+(?:\.\d+)?)\s*([a-zA-Zμ]+)\)', type_str)
            if conc_match and 'degradation' in type_str.lower():
                concentration = float(conc_match.group(1))
                concentration_unit = conc_match.group(2)
        
        # Remove all content in parentheses
        type_str = re.sub(r'\([^)]*\)', '', type_str).strip()

        # Remove extra parenthesis if any
        type_str = type_str.replace('(', '').replace(')', '').strip()
        
        # Remove "under ... treatment" phrases
        type_str = re.sub(r'under\s+[\d.]+\s*[μµ]M\s+treatment', '', type_str, flags=re.IGNORECASE).strip()
        
        # Clean up extra spaces
        type_str = re.sub(r'\s+', ' ', type_str).strip()
        
        # Handle "Cell viability" variations
        if 'cell viability' in type_str.lower():
            type_str = re.sub(r'cell\s+viability', 'Cell viability', type_str, flags=re.IGNORECASE)
        
        # Extract numeric variant (e.g., _1, _2)
        variant = None
        match = re.search(r'[_\s](\d+)$', type_str)
        if match:
            variant = int(match.group(1))
            type_str = re.sub(r'[_\s]\d+$', '', type_str).strip()
            
        # Rename "Inhibiton" to "Inhibition"
        if 'Inhibiton' in type_str:
            type_str = type_str.replace('Inhibiton', 'Inhibition')
        
        # Final cleanup
        type_str = type_str.strip()
        
        return type_str, variant, concentration, concentration_unit
    
    # Only add columns if they don't already exist
    df[['Type_Base', 'Type_Variant', 'Type_Concentration', 'Type_Concentration_Unit']] = df['Type'].apply(
        lambda x: pd.Series(parse_type(x))
    )
    
    return df

def standardize_type_base(df):
    """Standardize Type_Base values"""
    if 'Type_Base' not in df.columns:
        return df
    
    # Map degradation variants to standard name
    degradation_variants = [
        '% DEGRADATION',
        '% degradation',
        '% Degradation',
        'Degradation percentage'
    ]
    
    df['Type_Base'] = df['Type_Base'].replace(degradation_variants, 'Degradation')
    
    return df

def clean_cell_line_column(df):
    """Clean Cell_Line column if needed"""
    # Initialize new columns only if they don't exist
    for col in ['Cell_Line_Concentration', 'Cell_Line_Concentration_Unit']:
        if col not in df.columns:
            df[col] = None

    def parse_cell_line(cell_line_str):
        if pd.isna(cell_line_str):
            return None, None
        
        cell_line_str = str(cell_line_str).strip()
        
        # Pattern: "Cell Line (concentration unit)", e.g., "Jeko-1(10µM)" or "Jeko-1(100 nM)"
        conc_match = re.search(r'\((\d+(?:\.\d+)?)\s*([nμµ]?M)\)', cell_line_str)
        if conc_match:
            concentration = float(conc_match.group(1))
            concentration_unit = conc_match.group(2)
            return concentration, concentration_unit
        
        return None, None
    
    df[['Cell_Line_Concentration', 'Cell_Line_Concentration_Unit']] = df['Cell_Line'].apply(
        lambda x: pd.Series(parse_cell_line(x))
    )
    return df

def clean_value_column(df):
    """Parse and standardize Value column"""
    
    # Initialize new columns only if they don't exist
    for col in ['Value_Category', 'Value_Mean', 'Value_Error', 'Value_Unit', 
                'Value_Operator', 'Value_Range_Min', 'Value_Range_Max']:
        if col not in df.columns:
            df[col] = None
    
    for idx, value in df['Value'].items():
        if pd.isna(value):
            continue
        
        value_str = str(value).strip()
        
        # Category 1: Letter grades (A, B, C, D, E, F, G, AA, A+, A++)
        if re.match(r'^[A-G][\+]{0,2}$', value_str):
            df.at[idx, 'Value_Category'] = 'grade'
            continue
        
        # Category 2: Symbols (+, ++, +++, ++++, -, *, **, ***)
        if re.match(r'^[\+\-\*]{1,4}$', value_str):
            df.at[idx, 'Value_Category'] = 'symbol'
            continue
        
        # Category 3: Text descriptors
        text_values = ['No Degradation', 'ND', 'N.D.', 'NT', 'TBD', 'NDE', '/', 
                       'No major killing', 'No killing', 'slightly growth promoting',
                       'No major killing, slightly growth promoting', 'No', 'AA']
        if value_str in text_values:
            df.at[idx, 'Value_Category'] = 'text'
            continue
        
        # Category 4: Multiple values (comma-separated)
        if ',' in value_str:
            df.at[idx, 'Value_Category'] = 'multiple'
            # Parse first value only
            first_val = value_str.split(',')[0].strip()
            parsed = parse_single_value(first_val)
            if parsed:
                apply_parsed_value(df, idx, parsed)
            continue
        
        # Category 5: "at" phrases (30% at 1.0 μM, 96% at 1.0 μM)
        # This extracts the value BEFORE "at" and stores concentration separately
        at_match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s+at\s+(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)', value_str, re.IGNORECASE)
        if at_match:
            df.at[idx, 'Value_Category'] = 'numeric'
            df.at[idx, 'Value_Mean'] = float(at_match.group(1))
            df.at[idx, 'Value_Unit'] = at_match.group(2)
            df.at[idx, 'Value_Concentration'] = float(at_match.group(3))
            df.at[idx, 'Value_Concentration_Unit'] = at_match.group(4)
            continue
        
        # Category 6: Numeric values (with operators, error bars, or plain)
        parsed = parse_single_value(value_str)
        if parsed:
            df.at[idx, 'Value_Category'] = 'numeric'
            apply_parsed_value(df, idx, parsed)
            continue
        
        # Category 7: Range values (10nM≤x<100nM, x<10, 0.01-0.1μM, 150-200, 5nM-15nM)
        if ('x' in value_str.lower() or 
            re.search(r'\d+[<≤].*[<≤]\d+', value_str) or 
            re.search(r'\d+\.?\d*\s*([a-zA-Zμµ%]+)?-\s*\d+\.?\d*([a-zA-Zμµ%]+)?', value_str)):
            df.at[idx, 'Value_Category'] = 'range'
            parsed = parse_range_value(value_str)
            if parsed:
                df.at[idx, 'Value_Range_Min'] = parsed['min']
                df.at[idx, 'Value_Range_Max'] = parsed['max']
                df.at[idx, 'Value_Unit'] = parsed['unit']
            continue
        
        # If nothing matched
        df.at[idx, 'Value_Category'] = 'other'
    
    return df

def parse_single_value(value_str):
    """Parse a single numeric value with optional operator, error bar, and unit"""
    
    # Clean the value string first
    value_str = value_str.strip()
    
    # Pattern 1: Error bar with optional tilde prefix (0.785±0.03μM, 67±1.4%, ~58.7±0.03%)
    match = re.match(r'^~?\s*(-?\d+\.?\d*|\d*\.\d+)\s*[±]\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': float(match.group(2)),
            'unit': match.group(3),
            'operator': None
        }
    
    # Pattern 2: Operator with optional spaces (>10μM, <100nM, ≥150nM, <=100nM, > 3.16E-07M)
    match = re.match(r'^([>≥<≤]+|<=|>=)\s*(\d+\.?\d*(?:[eE][+-]?\d+)?|\d*\.\d+)\s*([a-zA-Zμµ%/]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(2)),
            'error': None,
            'unit': match.group(3),
            'operator': match.group(1)
        }
    
    # Pattern 3: Tilde prefix (~58.7%, ~3nM, ~30.1 %)
    match = re.match(r'^~\s*(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': '~'
        }

    # Pattern 4: Numeric with exponential notation (1.2e3 nM, 3.5E-2 μM, 3.16E-07M)
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)[eE][+-]?\d+\s*([a-zA-Zμµ%/]+)?$', value_str)
    if match:
        # Extract the full number including exponent by finding where the unit starts
        numeric_part = value_str
        if match.group(2):  # If there's a unit
            numeric_part = value_str[:value_str.rfind(match.group(2))].strip()
        return {
            'mean': float(numeric_part),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    # Pattern 5: Negative numbers (-36%, -15%, -5.3%)
    match = re.match(r'^(-\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    # Pattern 6: Asterisk suffix (88.9*, 0.08nM*, 7nM*)
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*\*$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': '*'
        }
    
    # Pattern 7: Value with parenthetical annotation (0.022 (1%), 0.052 (42%))
    # Extract first number before parentheses
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*\([^)]+\)$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    # Pattern 8: Standard numeric (2.63nM, 0.701μM, 67%, 0.001µM)
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%/]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    return None

def apply_parsed_value(df, idx, parsed):
    """Apply parsed value to dataframe row"""
    df.at[idx, 'Value_Mean'] = parsed['mean']
    df.at[idx, 'Value_Error'] = parsed['error']
    df.at[idx, 'Value_Unit'] = parsed['unit']
    df.at[idx, 'Value_Operator'] = parsed['operator']

def parse_range_value(value_str):
    """Parse range values like '10nM≤x<100nM', '0.01-0.1μM', or '150-200'"""
    
    # Clean the value string first
    value_str = value_str.strip()
    
    # Pattern 1: Inequality ranges with x (10nM≤x<100nM, 1.0μM≤x<3.0μM)
    match = re.search(
        r'(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*[<≤]\s*x\s*[<≤]\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?',
        value_str
    )
    if match:
        return {
            'min': float(match.group(1)),
            'max': float(match.group(3)),
            'unit': match.group(2) or match.group(4)
        }
    
    # Pattern 2: Hyphen ranges with unit (0.01-0.1μM, 100nM-300nM, 1-5μM)
    match = re.match(r'^(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*-\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'min': float(match.group(1)),
            'max': float(match.group(3)),
            'unit': match.group(2) or match.group(4)
        }
    
    # Pattern 3: Plain number ranges without unit (150-200, 50-100)
    match = re.match(r'^(\d+\.?\d*|\d*\.\d+)\s*-\s*(\d+\.?\d*|\d*\.\d+)$', value_str)
    if match:
        return {
            'min': float(match.group(1)),
            'max': float(match.group(2)),
            'unit': None
        }
    
    # Pattern 4: Operator with inequality (≥150nM, <100, >10μM, <=100nM, >=10)
    match = re.match(r'^([>≥<≤]+|<=|>=)\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        operator = match.group(1)
        val = float(match.group(2))
        unit = match.group(3)
        
        if operator in ['>', '≥', '>=']:
            return {'min': val, 'max': None, 'unit': unit}
        elif operator in ['<', '≤', '<=']:
            return {'min': None, 'max': val, 'unit': unit}
    
    return None

def normalize_units(df, base_col='Value'):
    """Normalize units to standard forms"""
    unit_mapping = {
        'nm': 'nM',
        'nM': 'nM',
        'nmol/L': 'nM',
        'uM': 'μM',
        'µM': 'μM',
        'um': 'μM',
        'μM': 'μM',
        'M': 'M',
        '%': '%',
    }
    
    # Normalize <base>_Unit
    if f'{base_col}_Unit' in df.columns:
        df[f'{base_col}_Unit'] = df[f'{base_col}_Unit'].replace(unit_mapping)
    
    # Normalize Value_Concentration_Unit
    if f'{base_col}_Concentration_Unit' in df.columns:
        df[f'{base_col}_Concentration_Unit'] = df[f'{base_col}_Concentration_Unit'].replace(unit_mapping)
    
    return df