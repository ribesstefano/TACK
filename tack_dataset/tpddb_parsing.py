""" Implements parsing of TPD HTML files to extract various sections into pandas
DataFrames. """
import re
import argparse
import logging
from typing import Optional, Union
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from tack_dataset.tpddb_cleaning import clean_activity_data
from tack_dataset.logging_utils import setup_logging


def extract_general_information(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract General Information from TPD HTML file.
    
    Args:
        html_path (Optional[Union[str, Path]]): Path to the HTML file.
        html_content (Optional[str]): HTML content as a string. If provided, html_path is ignored.
        
    Returns:
        pandas.DataFrame: DataFrame containing general information with one row per TPD
    """
    if html_content is None:
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    elif html_path is None:
        raise ValueError("Either html_path or html_content must be provided.")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract TPD ID from filename
    tpd_id = Path(html_path).stem
    
    # Find the General Information section
    general_section = None
    for div in soup.find_all('div', class_='div-unit'):
        title = div.find('div', class_='div-title')
        if title and 'General Information of This Targeted Protein Degrader' in title.get_text():
            general_section = div
            break
    
    # Initialize data dictionary
    data = {'TPD_ID': tpd_id}
    
    if general_section:
        table = general_section.find('table')
        if table:
            rows = table.find_all('tr')
            
            for row in rows:
                th = row.find('th')
                if not th:
                    continue
                
                field_name = th.get_text(strip=True)
                
                if field_name == 'TPD Name':
                    td = row.find('td')
                    if td:
                        data['TPD_Name'] = td.get_text(strip=True)
                
                elif field_name == 'Synonyms':
                    td = row.find('td')
                    if td:
                        syn_div = td.find('div', class_='synonymous2')
                        if syn_div:
                            data['Synonyms'] = syn_div.get_text(strip=True).replace('<p>', '').replace('</p>', '')
                
                elif field_name == 'Type':
                    td = row.find('td')
                    if td:
                        data['Type'] = td.get_text(strip=True)
                
                elif field_name == 'Sub Type':
                    td = row.find('td')
                    if td:
                        data['Sub_Type'] = td.get_text(strip=True)
                
                # LYTAC-specific fields
                elif field_name == 'Target':
                    td = row.find('td')
                    if td:
                        data['Target'] = td.get_text(strip=True)
                
                elif field_name == 'Target Adapter Name':
                    tds = row.find_all('td')
                    if tds:
                        data['Target_Adapter_Name'] = tds[0].get_text(strip=True)
                
                elif field_name == 'Target Adapter Type':
                    tds = row.find_all('td')
                    if len(tds) >= 2:
                        data['Target_Adapter_Type'] = tds[1].get_text(strip=True)
                
                elif field_name == 'Linker':
                    tds = row.find_all('td')
                    if tds:
                        data['Linker'] = tds[0].get_text(strip=True)
                
                elif field_name == 'Linker Type':
                    tds = row.find_all('td')
                    if len(tds) >= 2:
                        data['Linker_Type'] = tds[1].get_text(strip=True)
                
                elif field_name == 'Lysosome-targeting Receptor':
                    td = row.find('td')
                    if td:
                        data['Lysosome_Targeting_Receptor'] = td.get_text(strip=True)
                
                elif field_name == 'LTR Adapter Name':
                    tds = row.find_all('td')
                    if tds:
                        data['LTR_Adapter_Name'] = tds[0].get_text(strip=True)
                
                elif field_name == 'LTR Adapter Type':
                    tds = row.find_all('td')
                    if len(tds) >= 2:
                        data['LTR_Adapter_Type'] = tds[1].get_text(strip=True)
                
                elif field_name == 'LTR Adapter':
                    td = row.find('td')
                    if td:
                        data['LTR_Adapter'] = td.get_text(strip=True)
                
                elif field_name == 'Structure':
                    # Extract 3D and 2D structure file paths if available
                    td = row.find('td')
                    if td:
                        # Check for 3D structure link
                        structure_3d_link = td.find('a', href=re.compile(r'/sites/files/tpd/3d/'))
                        if structure_3d_link:
                            data['Structure_3D_Path'] = structure_3d_link['href']
                        
                        # Check for 2D structure link
                        structure_2d_link = td.find('a', href=re.compile(r'/sites/files/tpd/2d/'))
                        if structure_2d_link:
                            data['Structure_2D_Path'] = structure_2d_link['href']
                        
                        # Check for image
                        img = td.find('img', src=re.compile(r'/sites/files/tpd/img/'))
                        if img:
                            data['Structure_Image_Path'] = img['src']
                
                elif field_name == 'Reference':
                    td = row.find('td')
                    if td:
                        ref_div = td.find('div', class_='breakall2')
                        if ref_div:
                            # Extract reference text
                            ref_text = ref_div.get_text(strip=True)
                            # Remove the external link icon text
                            ref_text = re.sub(r'\s*\(patent\)\s*$', '', ref_text)
                            data['Reference'] = ref_text
                            
                            # Extract reference link
                            ref_link = ref_div.find('a')
                            if ref_link and ref_link.has_attr('href'):
                                data['Reference_Link'] = ref_link['href']
                
                elif field_name == 'Chemical Identifiers':
                    td = row.find('td')
                    if td:
                        dl = td.find('dl', class_='dl-in-table')
                        if dl:
                            # Extract all chemical identifiers
                            dts = dl.find_all('dt')
                            dds = dl.find_all('dd')
                            
                            for dt, dd in zip(dts, dds):
                                identifier_name = dt.get_text(strip=True)
                                identifier_value = dd.get_text(strip=True)
                                
                                # Clean up identifier name for column naming
                                col_name = identifier_name.replace(' ', '_')
                                data[col_name] = identifier_value
                                
                                # For PubChem CID, also extract the ID number
                                if identifier_name == 'PubChem CID':
                                    pubchem_id = re.search(r'(\d+)', identifier_value)
                                    if pubchem_id:
                                        data['PubChem_CID_Number'] = pubchem_id.group(1)
                
                elif field_name == 'Properties':
                    td = row.find('td')
                    if td:
                        props_text = td.get_text(strip=True)
                        # Parse properties using regex
                        polar_area = re.search(r'(\d+\.?\d*)\s*\(Polar Area\)', props_text)
                        complexity = re.search(r'(\d+\.?\d*)\s*\(Complexity\)', props_text)
                        heavy_atom = re.search(r'(\d+)\s*\(Heavy Atom Count\)', props_text)
                        
                        if polar_area:
                            data['Polar_Area'] = float(polar_area.group(1))
                        if complexity:
                            data['Complexity'] = float(complexity.group(1))
                        if heavy_atom:
                            data['Heavy_Atom_Count'] = int(heavy_atom.group(1))
                
                elif field_name == 'Lipinski RO5':
                    # This is a multi-row field, need to look at subsequent rows
                    # Find the RO5 Violation row
                    for ro5_row in rows[rows.index(row):]:
                        ro5_tds = ro5_row.find_all('td')
                        if not ro5_tds:
                            continue
                        
                        text = ro5_tds[0].get_text(strip=True)
                        
                        if 'RO5 Violation' in text:
                            if len(ro5_tds) >= 2:
                                violation_td = ro5_tds[1]
                                if violation_td:
                                    try:
                                        data['RO5_Violation'] = int(violation_td.get_text(strip=True))
                                    except ValueError:
                                        pass
                        
                        elif 'Molecular Weight' in text:
                            if len(ro5_tds) >= 2:
                                mw_td = ro5_tds[1]
                                if mw_td:
                                    try:
                                        data['Molecular_Weight'] = float(mw_td.get_text(strip=True))
                                    except ValueError:
                                        pass
                        
                        elif 'Partition Coefficient' in text:
                            if len(ro5_tds) >= 2:
                                clogp_td = ro5_tds[1]
                                if clogp_td:
                                    try:
                                        data['ClogP'] = float(clogp_td.get_text(strip=True))
                                    except ValueError:
                                        pass
                        
                        elif 'H-bond Donors' in text:
                            if len(ro5_tds) >= 2:
                                hbd_td = ro5_tds[1]
                                if hbd_td:
                                    try:
                                        data['HBD'] = int(hbd_td.get_text(strip=True))
                                    except ValueError:
                                        pass
                        
                        elif 'H-bond Acceptors' in text:
                            if len(ro5_tds) >= 2:
                                hba_td = ro5_tds[1]
                                if hba_td:
                                    try:
                                        data['HBA'] = int(hba_td.get_text(strip=True))
                                    except ValueError:
                                        pass
                        
                        elif 'Rotatable Bonds' in text:
                            if len(ro5_tds) >= 2:
                                rb_td = ro5_tds[1]
                                if rb_td:
                                    try:
                                        data['Rotatable_Bonds'] = int(rb_td.get_text(strip=True))
                                    except ValueError:
                                        pass
    
    # Create DataFrame with one row
    df = pd.DataFrame([data])
    return df


def extract_mode_of_action(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract Mode of Action information from TPD HTML file.
    
    Args:
        html_path: Path to the HTML file
        
    Returns:
        pandas.DataFrame: DataFrame containing Mode of Action information with columns:
                         'Type', 'Name', 'Gene Name', 'Sequence', 'Function', 'Info Link'
    """
    if html_content is None:
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    elif html_path is None:
        raise ValueError("Either html_path or html_content must be provided.")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the Mode of Action section
    moa_section = None
    for div in soup.find_all('div', class_='div-unit'):
        title = div.find('div', class_='div-title')
        if title and 'Mode of Action' in title.get_text():
            moa_section = div
            break
    
    # Extract mode of action data
    moa_data = []
    
    if moa_section:
        table = moa_section.find('table')
        if table:
            rows = table.find_all('tr')
            
            current_entry = {}
            current_type = None
            
            binding_preference_data = {}
            
            for row in rows:
                # Check if this is a section header (POI or Ligase or Binding Preference)
                if row.has_attr('class') and 'abc-info' in row.get('class'):
                    # Get the type (POI or Ligase or Binding Preference)
                    td = row.find('td', class_='adc-info-detail')
                    th = row.find('th', class_='adc-info-detail')
                    header_elem = td or th
                    if header_elem:
                        type_text = header_elem.get_text(strip=True)
                        if 'POI' in type_text or 'Protein of Interest' in type_text:
                            current_type = 'POI'
                        elif 'Ligase' in type_text:
                            current_type = 'Ligase'
                        elif 'Binding Preference' in type_text:
                            current_type = 'BindingPreference'
                    continue
                
                # Extract field data
                th = row.find('th')
                if th:
                    field_name = th.get_text(strip=True)
                    
                    if field_name == 'POI Name' or field_name == 'Ligase Name':
                        # Save previous entry if it exists
                        if current_entry:
                            moa_data.append(current_entry)
                        
                        # Start a new entry
                        current_entry = {'Type': current_type}
                        
                        tds = row.find_all('td')
                        if tds:
                            # Get name from first td
                            name = tds[0].get_text(strip=True)
                            current_entry['Name'] = name
                            
                            # Get link to info page from last td
                            info_link = ''
                            if len(tds) > 1:
                                link_tag = tds[-1].find('a')
                                if link_tag and link_tag.has_attr('href'):
                                    info_link = link_tag['href']
                            current_entry['Info_Link'] = info_link
                            current_entry[f'{current_type}_ID'] = info_link.split('/')[-1]  # Extract POI ID from link
                    
                    elif field_name == 'Gene Name':
                        td = row.find('td')
                        if td:
                            current_entry['Gene_Name'] = td.get_text(strip=True)
                    
                    elif field_name == 'Sequence':
                        td = row.find('td')
                        if td:
                            # Get text from the synonymous div
                            seq_div = td.find('div', class_='synonymous')
                            if seq_div:
                                # Remove <br> tags and join
                                sequence = seq_div.get_text(separator='', strip=True)
                                current_entry['Sequence'] = sequence
                    
                    elif field_name == 'Function':
                        td = row.find('td')
                        if td:
                            # Get text from the synonymous div
                            func_div = td.find('div', class_='synonymous')
                            if func_div:
                                function = func_div.get_text(strip=True)
                                current_entry['Function'] = function
                    
                    # Extract Binding Preference (for Molecular Glues)
                    elif field_name == 'Binding Preference' and current_type == 'BindingPreference':
                        tds = row.find_all('td')
                        ths = row.find_all('th')
                        
                        # First td contains Binding Preference value
                        if len(tds) >= 1:
                            pref_div = tds[0].find('div')
                            if pref_div:
                                binding_preference_data['Binding_Preference'] = pref_div.get_text(strip=True)
                        
                        # Check if Confidence Level is in the same row
                        if len(ths) >= 2 and 'Confidence Level' in ths[1].get_text(strip=True):
                            if len(tds) >= 2:
                                # Extract just the confidence level letter (A, B, C, etc.)
                                # The tooltip-wrapper span contains the letter
                                conf_span = tds[1].find('span', class_='tooltip-wrapper')
                                if conf_span:
                                    # Get only the direct text of the span, not the tooltip content
                                    conf_text = conf_span.get_text(separator=' ', strip=True)
                                    # Extract just the first letter/word before "Hover"
                                    conf_level = conf_text.split()[0] if conf_text else ''
                                    binding_preference_data['Confidence_Level'] = conf_level
            
            # Add the last entry
            if current_entry:
                moa_data.append(current_entry)
    
    # Create DataFrame
    df = pd.DataFrame(moa_data)
    return df


def extract_target_degradation_activities(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract Target Degradation Activities information from TPD HTML file.
    
    Args:
        html_path: Path to the HTML file
        
    Returns:
        pandas.DataFrame: DataFrame containing degradation activities with columns:
                         'Activity_Number', 'Type', 'Value', 'POI_ID', 'POI_Name', 
                         'Cell_Line', 'Cell_Line_ID', 'Assay', 'Description', 'Reference'
    """
    if html_content is None:
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    elif html_path is None:
        raise ValueError("Either html_path or html_content must be provided.")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the Target Degradation Activities section
    degradation_section = None
    for div in soup.find_all('div', class_='div-unit'):
        title = div.find('div', class_='div-title')
        if title and 'Target Degradation Activities of This TPD' in title.get_text():
            degradation_section = div
            break
    
    # Extract degradation data
    activities = []
    
    if degradation_section:
        table = degradation_section.find('table')
        if table:
            rows = table.find_all('tr')
            
            current_activity = {}
            activity_number = 0
            
            for row in rows:
                # Check if this is an activity header
                if row.has_attr('class') and 'abc-info' in row.get('class'):
                    # Save previous activity if it exists
                    if current_activity:
                        activities.append(current_activity)
                    
                    # Start new activity
                    activity_number += 1
                    td = row.find('td', class_='adc-info-detail')
                    if td:
                        current_activity = {
                            'Activity_Number': activity_number,
                            'Activity_Label': td.get_text(strip=True)
                        }
                    continue
                
                # Extract field data
                th = row.find('th')
                if th:
                    field_name = th.get_text(strip=True)
                    ths = row.find_all('th')
                    tds = row.find_all('td')
                    
                    if field_name == 'Type':
                        # Type and Value are in the same row
                        if len(tds) >= 1:
                            current_activity['Type'] = tds[0].get_text(strip=True)
                        
                        # Check if Value is in the same row (2 th tags)
                        if len(ths) >= 2 and ths[1].get_text(strip=True) == 'Value':
                            if len(tds) >= 2:
                                current_activity['Value'] = tds[1].get_text(strip=True)
                    
                    elif field_name == 'Value':
                        if len(tds) >= 1:
                            current_activity['Value'] = tds[0].get_text(strip=True)
                    
                    elif field_name == 'POI ID':
                        # POI ID and POI Name are in the same row
                        if len(tds) >= 1:
                            current_activity['POI_ID'] = tds[0].get_text(strip=True)
                        
                        # Check if POI Name is in the same row (2 th tags)
                        if len(ths) >= 2 and ths[1].get_text(strip=True) == 'POI Name':
                            if len(tds) >= 2:
                                current_activity['POI_Name'] = tds[1].get_text(strip=True)
                    
                    elif field_name == 'POI Name':
                        if len(tds) >= 1:
                            current_activity['POI_Name'] = tds[0].get_text(strip=True)
                    
                    elif field_name == 'Cell Line':
                        if len(tds) >= 1:
                            current_activity['Cell_Line'] = tds[0].get_text(strip=True)
                        
                        if len(ths) >= 2 and ths[1].get_text(strip=True) == 'Cell Line ID':
                            if len(tds) >= 2:
                                current_activity['Cell_Line_ID'] = tds[1].get_text(strip=True)
                    
                    elif field_name == 'Cell Line ID':
                        if len(tds) >= 1:
                            current_activity['Cell_Line_ID'] = tds[0].get_text(strip=True)
                    
                    elif field_name == 'Assay':
                        td = row.find('td')
                        if td:
                            current_activity['Assay'] = td.get_text(strip=True)
                    
                    elif field_name == 'Description':
                        td = row.find('td')
                        if td:
                            desc_div = td.find('div', class_='synonymous')
                            if desc_div:
                                current_activity['Description'] = desc_div.get_text(strip=True)
                    
                    elif field_name == 'Reference':
                        td = row.find('td')
                        if td:
                            ref_div = td.find('div', class_='breakall2')
                            if ref_div:
                                current_activity['Reference'] = ref_div.get_text(strip=True)
            
            # Add the last activity
            if current_activity:
                activities.append(current_activity)
    
    # Create DataFrame
    df = pd.DataFrame(activities)
    return df


def extract_binding_affinities(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract Binding Affinities information from TPD HTML file.
    
    Args:
        html_path: Path to the HTML file
        html_content: HTML content as a string
        
    Returns:
        pandas.DataFrame: DataFrame containing binding affinities with columns:
                         'Activity_Number', 'Type', 'Value', 'POI_ID', 'POI_Name', 
                         'Cell_Line', 'Cell_Line_ID', 'Assay', 'Description', 'Reference'
    """
    if html_content is None:
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    elif html_path is None:
        raise ValueError("Either html_path or html_content must be provided.")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the Binding Affinities section
    binding_section = None
    for div in soup.find_all('div', class_='div-unit'):
        title = div.find('div', class_='div-title')
        if title and 'Binding Affinities of This TPD' in title.get_text():
            binding_section = div
            break
    
    # Extract binding affinity data
    activities = []
    
    if binding_section:
        table = binding_section.find('table')
        if table:
            rows = table.find_all('tr')
            
            current_activity = {}
            activity_number = 0
            
            for row in rows:
                # Check if this is an activity header
                if row.has_attr('class') and 'abc-info' in row.get('class'):
                    # Save previous activity if it exists
                    if current_activity:
                        activities.append(current_activity)
                    
                    # Start new activity
                    activity_number += 1
                    td = row.find('td', class_='adc-info-detail')
                    if td:
                        current_activity = {
                            'Activity_Number': activity_number,
                            'Activity_Label': td.get_text(strip=True)
                        }
                    continue
                
                # Extract field data
                th = row.find('th')
                if th:
                    field_name = th.get_text(strip=True)
                    ths = row.find_all('th')
                    tds = row.find_all('td')
                    
                    if field_name == 'Type':
                        # Type and Value are in the same row
                        if len(tds) >= 1:
                            current_activity['Type'] = tds[0].get_text(strip=True)
                        
                        # Check if Value is in the same row (2 th tags)
                        if len(ths) >= 2 and ths[1].get_text(strip=True) == 'Value':
                            if len(tds) >= 2:
                                current_activity['Value'] = tds[1].get_text(strip=True)
                    
                    elif field_name == 'Value':
                        if len(tds) >= 1:
                            current_activity['Value'] = tds[0].get_text(strip=True)
                    
                    elif field_name == 'POI ID':
                        # POI ID and POI Name are in the same row
                        if len(tds) >= 1:
                            current_activity['POI_ID'] = tds[0].get_text(strip=True)
                        
                        # Check if POI Name is in the same row (2 th tags)
                        if len(ths) >= 2 and ths[1].get_text(strip=True) == 'POI Name':
                            if len(tds) >= 2:
                                # Extract POI name, might have links
                                poi_link = tds[1].find('a', class_='conditional-link')
                                if poi_link:
                                    current_activity['POI_Name'] = poi_link.get_text(strip=True)
                                else:
                                    current_activity['POI_Name'] = tds[1].get_text(strip=True)
                    
                    elif field_name == 'POI Name':
                        if len(tds) >= 1:
                            poi_link = tds[0].find('a', class_='conditional-link')
                            if poi_link:
                                current_activity['POI_Name'] = poi_link.get_text(strip=True)
                            else:
                                current_activity['POI_Name'] = tds[0].get_text(strip=True)
                    
                    elif field_name == 'Cell Line':
                        if len(tds) >= 1:
                            current_activity['Cell_Line'] = tds[0].get_text(strip=True)
                        
                        # Check if Cell Line ID is in the same row
                        if len(ths) >= 2 and ths[1].get_text(strip=True) == 'Cell Line ID':
                            if len(tds) >= 2:
                                current_activity['Cell_Line_ID'] = tds[1].get_text(strip=True)
                    
                    elif field_name == 'Cell Line ID':
                        if len(tds) >= 1:
                            current_activity['Cell_Line_ID'] = tds[0].get_text(strip=True)
                    
                    elif field_name == 'Assay':
                        td = row.find('td')
                        if td:
                            current_activity['Assay'] = td.get_text(strip=True)
                    
                    elif field_name == 'Description':
                        td = row.find('td')
                        if td:
                            desc_div = td.find('div', class_='synonymous')
                            if desc_div:
                                current_activity['Description'] = desc_div.get_text(strip=True)
                    
                    elif field_name == 'Reference':
                        td = row.find('td')
                        if td:
                            ref_div = td.find('div', class_='breakall2')
                            if ref_div:
                                current_activity['Reference'] = ref_div.get_text(strip=True)
            
            # Add the last activity
            if current_activity:
                activities.append(current_activity)
    
    # Create DataFrame
    df = pd.DataFrame(activities)
    return df


def extract_cytotoxic_activities(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract cytotoxic activities information from TPDdb HTML file.
    
    Args:
        html_path: Path to the HTML file
        
    Returns:
        pandas DataFrame with columns: Activity_Number, Type, Value, POI_ID, POI_Name,
        Cell_Line, Cell_Line_ID, Assay, Description, Reference
    """
    if html_content is None:
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    elif html_path is None:
        raise ValueError("Either html_path or html_content must be provided.")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the Cytotoxic Activities section
    cytotoxic_section = None
    for div in soup.find_all('div', class_='div-unit'):
        title = div.find('div', class_='div-title')
        if title and 'Cytotoxic Activities of This TPD' in title.get_text():
            cytotoxic_section = div
            break
    
    if not cytotoxic_section:
        # Return empty DataFrame if section not found
        return pd.DataFrame(columns=['Activity_Number', 'Type', 'Value', 'POI_ID', 
                                    'POI_Name', 'Cell_Line', 'Cell_Line_ID', 
                                    'Assay', 'Description', 'Reference'])
    
    # Find the table within the section
    table = cytotoxic_section.find('table')
    if not table:
        return pd.DataFrame(columns=['Activity_Number', 'Type', 'Value', 'POI_ID', 
                                    'POI_Name', 'Cell_Line', 'Cell_Line_ID', 
                                    'Assay', 'Description', 'Reference'])
    
    activities = []
    current_activity = {}
    activity_number = None
    
    rows = table.find_all('tr')
    
    for row in rows:
        # Check if this is an activity header row
        if row.find('td', class_='adc-info-detail'):
            # Save previous activity if it exists
            if current_activity:
                activities.append(current_activity)
            
            # Start new activity
            header_text = row.find('td', class_='adc-info-detail').get_text(strip=True)
            # Extract activity number from "Cytotoxic Activities Information X"
            match = re.search(r'Cytotoxic Activities Information (\d+)', header_text)
            if match:
                activity_number = int(match.group(1))
            else:
                activity_number = len(activities) + 1
            
            current_activity = {
                'Activity_Number': activity_number,
                'Type': None,
                'Value': None,
                'POI_ID': None,
                'POI_Name': None,
                'Cell_Line': None,
                'Cell_Line_ID': None,
                'Assay': None,
                'Description': None,
                'Reference': None
            }
            continue
        
        # Skip rows without activity context
        if not current_activity:
            continue
        
        # Extract data from rows
        th = row.find('th')
        if not th:
            continue
        
        label = th.get_text(strip=True)
        
        if label == 'Type':
            # Type and Value are often in the same row
            ths = row.find_all('th')
            tds = row.find_all('td')
            
            if len(tds) >= 1:
                current_activity['Type'] = tds[0].get_text(strip=True)
            
            # Check if Value is in the same row (2 th tags)
            if len(ths) >= 2 and ths[1].get_text(strip=True) == 'Value':
                if len(tds) >= 2:
                    current_activity['Value'] = tds[1].get_text(strip=True)
        
        elif label == 'Value':
            tds = row.find_all('td')
            if len(tds) >= 1:
                current_activity['Value'] = tds[0].get_text(strip=True)
        
        elif label == 'POI ID':
            # POI ID and POI Name are often in the same row
            ths = row.find_all('th')
            tds = row.find_all('td')
            
            if len(tds) >= 1:
                current_activity['POI_ID'] = tds[0].get_text(strip=True)
            
            # Check if POI Name is in the same row (2 th tags)
            if len(ths) >= 2 and ths[1].get_text(strip=True) == 'POI Name':
                if len(tds) >= 2:
                    current_activity['POI_Name'] = tds[1].get_text(strip=True)
        
        elif label == 'POI Name':
            tds = row.find_all('td')
            if len(tds) >= 1:
                current_activity['POI_Name'] = tds[0].get_text(strip=True)
        
        elif label == 'Cell Line':
            ths = row.find_all('th')
            tds = row.find_all('td')
            
            if len(tds) >= 1:
                current_activity['Cell_Line'] = tds[0].get_text(strip=True)
            
            if len(ths) >= 2 and ths[1].get_text(strip=True) == 'Cell Line ID':
                if len(tds) >= 2:
                    current_activity['Cell_Line_ID'] = tds[1].get_text(strip=True)
        
        elif label == 'Cell Line ID':
            tds = row.find_all('td')
            if len(tds) >= 1:
                current_activity['Cell_Line_ID'] = tds[0].get_text(strip=True)
        
        elif label == 'Assay':
            td = row.find('td', colspan='6')
            if td:
                current_activity['Assay'] = td.get_text(strip=True)
        
        elif label == 'Description':
            td = row.find('td', colspan='6')
            if td:
                # Get description from the div with class 'synonymous'
                desc_div = td.find('div', class_='synonymous')
                if desc_div:
                    # Get text and clean up
                    desc_text = desc_div.get_text(strip=True)
                    # Remove trailing <p></p> markers
                    desc_text = desc_text.replace('<p></p>', '').strip()
                    current_activity['Description'] = desc_text
        
        elif label == 'Reference':
            td = row.find('td')
            if td:
                # Get reference text, may contain links
                ref_div = td.find('div', class_='breakall2')
                if ref_div:
                    current_activity['Reference'] = ref_div.get_text(strip=True)
    
    # Don't forget to add the last activity
    if current_activity:
        activities.append(current_activity)
    
    return pd.DataFrame(activities)


def extract_disease_info(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract disease information from TPD HTML file.
    
    Args:
        html_path: Path to the HTML file
        
    Returns:
        pandas.DataFrame: DataFrame containing disease information with columns:
                         'Disease Name', 'WHO ICD-11', 'Source'
    """
    if html_content is None:
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    elif html_path is None:
        raise ValueError("Either html_path or html_content must be provided.")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the disease information section
    disease_section = None
    for div in soup.find_all('div', class_='div-unit'):
        title = div.find('div', class_='div-title')
        if title and 'Disease Information of This TPD' in title.get_text():
            disease_section = div
            break
    
    # Extract disease data
    diseases = []
    if disease_section:
        table = disease_section.find('table')
        if table:
            rows = table.find_all('tr')
            
            i = 0
            while i < len(rows):
                row = rows[i]
                th = row.find('th')
                
                if th and th.get_text(strip=True) == 'Disease Name':
                    # This row has the disease name
                    td = row.find('td')
                    disease_name = td.get_text(strip=True) if td else ''
                    
                    # Next row should have WHO ICD-11 and Source
                    if i + 1 < len(rows):
                        next_row = rows[i + 1]
                        tds = next_row.find_all('td')
                        
                        who_icd = tds[0].get_text(strip=True) if len(tds) > 0 else ''
                        source = tds[1].get_text(strip=True) if len(tds) > 1 else ''
                        
                        diseases.append({
                            'Disease_Name': disease_name,
                            'WHO_ICD-11': who_icd,
                            'Source': source
                        })
                        
                        i += 2  # Skip the next row since we already processed it
                        continue
                
                i += 1
    
    # Create DataFrame
    df = pd.DataFrame(diseases)
    return df


def _clean_value(val):
    """Convert pandas NaN/None to Python None, keep other values as-is."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return val


def extract_binding_preference(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> dict:
    """
    Extract Binding Preference information from TPD HTML file (for Molecular Glues).
    
    Args:
        html_path: Path to the HTML file
        html_content: HTML content as a string
        
    Returns:
        Dictionary with 'Binding_Preference' and 'Confidence_Level' keys
    """
    if html_content is None:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    elif html_path is None:
        raise ValueError("Either html_path or html_content must be provided.")
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the Mode of Action section
    moa_section = None
    for div in soup.find_all('div', class_='div-unit'):
        title = div.find('div', class_='div-title')
        if title and 'Mode of Action' in title.get_text():
            moa_section = div
            break
    
    binding_pref_data = {
        'Binding_Preference': None,
        'Confidence_Level': None
    }
    
    if moa_section:
        table = moa_section.find('table')
        if table:
            rows = table.find_all('tr')
            in_binding_section = False
            
            for row in rows:
                # Check if this is the Binding Preference header
                if row.has_attr('class') and 'abc-info' in row.get('class'):
                    header_elem = row.find('th', class_='adc-info-detail') or row.find('td', class_='adc-info-detail')
                    if header_elem and 'Binding Preference' in header_elem.get_text(strip=True):
                        in_binding_section = True
                    continue
                
                if in_binding_section:
                    th = row.find('th')
                    if th and 'Binding Preference' in th.get_text(strip=True):
                        tds = row.find_all('td')
                        ths = row.find_all('th')
                        
                        # First td contains Binding Preference value
                        if len(tds) >= 1:
                            pref_div = tds[0].find('div')
                            if pref_div:
                                binding_pref_data['Binding_Preference'] = pref_div.get_text(strip=True)
                        
                        # Check if Confidence Level is in the same row
                        if len(ths) >= 2 and 'Confidence Level' in ths[1].get_text(strip=True):
                            if len(tds) >= 2:
                                # Extract just the confidence level letter
                                conf_span = tds[1].find('span', class_='tooltip-wrapper')
                                if conf_span:
                                    # Get text from span, extract first word
                                    conf_texts = list(conf_span.stripped_strings)
                                    if conf_texts:
                                        binding_pref_data['Confidence_Level'] = conf_texts[0]
                        break  # Found what we need
    
    return binding_pref_data


def extract_all_data(
        html_path: Optional[Union[str, Path]] = None,
        html_content: Optional[str] = None,
) -> dict:
    """
    Extract all data from TPD HTML file and return as a dictionary matching the Gemini extraction format.
    Handles multiple POIs and Ligases by joining with semicolons.
    
    Args:
        html_path: Path to the HTML file
        html_content: HTML content as a string
        
    Returns:
        Dictionary with all extracted fields in the format expected by combine_extraction_data
    """
    general_df = extract_general_information(html_path, html_content)
    moa_df = extract_mode_of_action(html_path, html_content)
    degradation_df = extract_target_degradation_activities(html_path, html_content)
    binding_affinities_df = extract_binding_affinities(html_path, html_content)
    cytotoxic_df = extract_cytotoxic_activities(html_path, html_content)
    disease_df = extract_disease_info(html_path, html_content)
    binding_pref = extract_binding_preference(html_path, html_content)
    
    # Combine degradation activities and binding affinities (they have the same structure)
    degradation_df = pd.concat([degradation_df, binding_affinities_df], ignore_index=True)
    
    general = general_df.iloc[0].to_dict() if len(general_df) > 0 else {}
    
    # Helper to clean semicolon-only strings
    def clean_semicolons_static(value_str):
        """Convert strings that are only semicolons (e.g., ';', ';;') to None"""
        if value_str is None or value_str == '':
            return None
        if all(c == ';' for c in value_str):
            return None
        return value_str
    
    # Extract ALL POIs and join with semicolons
    poi_rows = moa_df[moa_df['Type'] == 'POI']
    if len(poi_rows) > 0:
        poi_data = {
            'Name': clean_semicolons_static(';'.join([str(row['Name']) for _, row in poi_rows.iterrows() if pd.notna(row.get('Name'))])),
            'Gene_Name': clean_semicolons_static(';'.join([str(row['Gene_Name']) for _, row in poi_rows.iterrows() if pd.notna(row.get('Gene_Name'))])),
            'Sequence': clean_semicolons_static(';'.join([str(row['Sequence']) for _, row in poi_rows.iterrows() if pd.notna(row.get('Sequence'))])),
            'Function': clean_semicolons_static(';'.join([str(row['Function']) for _, row in poi_rows.iterrows() if pd.notna(row.get('Function'))])),
        }
    else:
        poi_data = {}
    
    # Extract ALL Ligases and join with semicolons
    ligase_rows = moa_df[moa_df['Type'] == 'Ligase']
    if len(ligase_rows) > 0:
        ligase_data = {
            'Name': clean_semicolons_static(';'.join([str(row['Name']) for _, row in ligase_rows.iterrows() if pd.notna(row.get('Name'))])),
            'Ligase_ID': clean_semicolons_static(';'.join([str(row['Ligase_ID']) for _, row in ligase_rows.iterrows() if pd.notna(row.get('Ligase_ID'))])),
            'Gene_Name': clean_semicolons_static(';'.join([str(row['Gene_Name']) for _, row in ligase_rows.iterrows() if pd.notna(row.get('Gene_Name'))])),
            'Sequence': clean_semicolons_static(';'.join([str(row['Sequence']) for _, row in ligase_rows.iterrows() if pd.notna(row.get('Sequence'))])),
            'Function': clean_semicolons_static(';'.join([str(row['Function']) for _, row in ligase_rows.iterrows() if pd.notna(row.get('Function'))])),
        }
    else:
        ligase_data = {}
    
    result = {
        'subtype': _clean_value(general.get('Type') or general.get('Sub_Type')),
        'inchi': _clean_value(general.get('InChI')),
        'inchikey': _clean_value(general.get('InChIKey')),
        'iupac': _clean_value(general.get('IUPAC')),
        'structure_2d': _clean_value(general.get('Structure_2D_Path')),
        'structure_3d': _clean_value(general.get('Structure_3D_Path')),
        'heavy_atom_count': _clean_value(general.get('Heavy_Atom_Count')),
        
        'poi_name': _clean_value(poi_data.get('Name')),
        'poi_gene_name': _clean_value(poi_data.get('Gene_Name')),
        'poi_sequence': _clean_value(poi_data.get('Sequence')),
        'poi_function': _clean_value(poi_data.get('Function')),
        
        'ligase_name': _clean_value(ligase_data.get('Name')),
        'ligase_id': _clean_value(ligase_data.get('Ligase_ID')),
        'ligase_gene_name': _clean_value(ligase_data.get('Gene_Name')),
        'ligase_sequence': _clean_value(ligase_data.get('Sequence')),
        'ligase_function': _clean_value(ligase_data.get('Function')),
        
        'binding_preference': _clean_value(binding_pref.get('Binding_Preference')),
        'confidence_level': _clean_value(binding_pref.get('Confidence_Level')),
        
        'dc50_values': [],
        'dc50_poi_names': [],
        'dc50_poi_ids': [],
        'dc50_cell_lines': [],
        'dc50_cell_line_ids': [],
        'dc50_assay_types': [],
        'dc50_descriptions': [],
        
        'ic50_values': [],
        'ic50_poi_names': [],
        'ic50_poi_ids': [],
        'ic50_cell_lines': [],
        'ic50_cell_line_ids': [],
        'ic50_assay_types': [],
        'ic50_descriptions': [],
        
        'ec50_values': [],
        'ec50_poi_names': [],
        'ec50_poi_ids': [],
        'ec50_cell_lines': [],
        'ec50_cell_line_ids': [],
        'ec50_assay_types': [],
        'ec50_descriptions': [],
        
        'dmax_values': [],
        'dmax_poi_names': [],
        'dmax_poi_ids': [],
        'dmax_cell_lines': [],
        'dmax_cell_line_ids': [],
        'dmax_assay_types': [],
        'dmax_descriptions': [],
        
        'disease_names': []
    }
    
    dfs_to_combine = []
    if len(degradation_df) > 0:
        dfs_to_combine.append(degradation_df)
    if len(cytotoxic_df) > 0:
        dfs_to_combine.append(cytotoxic_df)
    
    combined_activities_df = pd.concat(dfs_to_combine, ignore_index=True) if dfs_to_combine else pd.DataFrame()
    
    for _, row in combined_activities_df.iterrows():
        activity_type = str(row.get('Type', '')).upper()
        
        if 'DC50' in activity_type or 'DC 50' in activity_type:
            val = _clean_value(row.get('Value'))
            if val is not None:
                result['dc50_values'].append(val)
                result['dc50_poi_names'].append(_clean_value(row.get('POI_Name')) or '')
                result['dc50_poi_ids'].append(_clean_value(row.get('POI_ID')) or '')
                result['dc50_cell_lines'].append(_clean_value(row.get('Cell_Line')) or '')
                result['dc50_cell_line_ids'].append(_clean_value(row.get('Cell_Line_ID')) or '')
                result['dc50_assay_types'].append(_clean_value(row.get('Assay')) or '')
                result['dc50_descriptions'].append(_clean_value(row.get('Description')) or '')
        elif 'IC50' in activity_type or 'IC 50' in activity_type:
            val = _clean_value(row.get('Value'))
            if val is not None:
                result['ic50_values'].append(val)
                result['ic50_poi_names'].append(_clean_value(row.get('POI_Name')) or '')
                result['ic50_poi_ids'].append(_clean_value(row.get('POI_ID')) or '')
                result['ic50_cell_lines'].append(_clean_value(row.get('Cell_Line')) or '')
                result['ic50_cell_line_ids'].append(_clean_value(row.get('Cell_Line_ID')) or '')
                result['ic50_assay_types'].append(_clean_value(row.get('Assay')) or '')
                result['ic50_descriptions'].append(_clean_value(row.get('Description')) or '')
        elif 'EC50' in activity_type or 'EC 50' in activity_type:
            val = _clean_value(row.get('Value'))
            if val is not None:
                result['ec50_values'].append(val)
                result['ec50_poi_names'].append(_clean_value(row.get('POI_Name')) or '')
                result['ec50_poi_ids'].append(_clean_value(row.get('POI_ID')) or '')
                result['ec50_cell_lines'].append(_clean_value(row.get('Cell_Line')) or '')
                result['ec50_cell_line_ids'].append(_clean_value(row.get('Cell_Line_ID')) or '')
                result['ec50_assay_types'].append(_clean_value(row.get('Assay')) or '')
                result['ec50_descriptions'].append(_clean_value(row.get('Description')) or '')
        elif 'DMAX' in activity_type or 'D MAX' in activity_type:
            val = _clean_value(row.get('Value'))
            if val is not None:
                result['dmax_values'].append(val)
                result['dmax_poi_names'].append(_clean_value(row.get('POI_Name')) or '')
                result['dmax_poi_ids'].append(_clean_value(row.get('POI_ID')) or '')
                result['dmax_cell_lines'].append(_clean_value(row.get('Cell_Line')) or '')
                result['dmax_cell_line_ids'].append(_clean_value(row.get('Cell_Line_ID')) or '')
                result['dmax_assay_types'].append(_clean_value(row.get('Assay')) or '')
                result['dmax_descriptions'].append(_clean_value(row.get('Description')) or '')
    
    # Helper to clean up semicolon-only strings
    def clean_semicolons(value_str):
        """Convert strings that are only semicolons (e.g., ';', ';;') to None"""
        if value_str is None or value_str == '':
            return None
        # Check if the string contains only semicolons
        if all(c == ';' for c in value_str):
            return None
        return value_str
    
    # Join lists into semicolon-separated strings for single-row output
    result['dc50_values'] = clean_semicolons(';'.join(result['dc50_values']) if result['dc50_values'] else None)
    result['dc50_poi_names'] = clean_semicolons(';'.join(result['dc50_poi_names']) if result['dc50_poi_names'] else None)
    result['dc50_poi_ids'] = clean_semicolons(';'.join(result['dc50_poi_ids']) if result['dc50_poi_ids'] else None)
    result['dc50_cell_lines'] = clean_semicolons(';'.join(result['dc50_cell_lines']) if result['dc50_cell_lines'] else None)
    result['dc50_cell_line_ids'] = clean_semicolons(';'.join(result['dc50_cell_line_ids']) if result['dc50_cell_line_ids'] else None)
    result['dc50_assay_types'] = clean_semicolons(';'.join(result['dc50_assay_types']) if result['dc50_assay_types'] else None)
    result['dc50_descriptions'] = clean_semicolons(';'.join(result['dc50_descriptions']) if result['dc50_descriptions'] else None)
    
    result['ic50_values'] = clean_semicolons(';'.join(result['ic50_values']) if result['ic50_values'] else None)
    result['ic50_poi_names'] = clean_semicolons(';'.join(result['ic50_poi_names']) if result['ic50_poi_names'] else None)
    result['ic50_poi_ids'] = clean_semicolons(';'.join(result['ic50_poi_ids']) if result['ic50_poi_ids'] else None)
    result['ic50_cell_lines'] = clean_semicolons(';'.join(result['ic50_cell_lines']) if result['ic50_cell_lines'] else None)
    result['ic50_cell_line_ids'] = clean_semicolons(';'.join(result['ic50_cell_line_ids']) if result['ic50_cell_line_ids'] else None)
    result['ic50_assay_types'] = clean_semicolons(';'.join(result['ic50_assay_types']) if result['ic50_assay_types'] else None)
    result['ic50_descriptions'] = clean_semicolons(';'.join(result['ic50_descriptions']) if result['ic50_descriptions'] else None)
    
    result['ec50_values'] = clean_semicolons(';'.join(result['ec50_values']) if result['ec50_values'] else None)
    result['ec50_poi_names'] = clean_semicolons(';'.join(result['ec50_poi_names']) if result['ec50_poi_names'] else None)
    result['ec50_poi_ids'] = clean_semicolons(';'.join(result['ec50_poi_ids']) if result['ec50_poi_ids'] else None)
    result['ec50_cell_lines'] = clean_semicolons(';'.join(result['ec50_cell_lines']) if result['ec50_cell_lines'] else None)
    result['ec50_cell_line_ids'] = clean_semicolons(';'.join(result['ec50_cell_line_ids']) if result['ec50_cell_line_ids'] else None)
    result['ec50_assay_types'] = clean_semicolons(';'.join(result['ec50_assay_types']) if result['ec50_assay_types'] else None)
    result['ec50_descriptions'] = clean_semicolons(';'.join(result['ec50_descriptions']) if result['ec50_descriptions'] else None)
    
    result['dmax_values'] = clean_semicolons(';'.join(result['dmax_values']) if result['dmax_values'] else None)
    result['dmax_poi_names'] = clean_semicolons(';'.join(result['dmax_poi_names']) if result['dmax_poi_names'] else None)
    result['dmax_poi_ids'] = clean_semicolons(';'.join(result['dmax_poi_ids']) if result['dmax_poi_ids'] else None)
    result['dmax_cell_lines'] = clean_semicolons(';'.join(result['dmax_cell_lines']) if result['dmax_cell_lines'] else None)
    result['dmax_cell_line_ids'] = clean_semicolons(';'.join(result['dmax_cell_line_ids']) if result['dmax_cell_line_ids'] else None)
    result['dmax_assay_types'] = clean_semicolons(';'.join(result['dmax_assay_types']) if result['dmax_assay_types'] else None)
    result['dmax_descriptions'] = clean_semicolons(';'.join(result['dmax_descriptions']) if result['dmax_descriptions'] else None)
    
    result['disease_names'] = clean_semicolons(';'.join(disease_df['Disease_Name'].tolist()) if len(disease_df) > 0 else None)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='TPDdb Batch Parsing - Parse downloaded TPDdb HTML files into structured CSVs for further cleaning and curation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all TPD IDs (parses dowloaded HTML files into CSVs)
  python parsing.py
  
  # Process first 10 entries for testing
  python parsing.py --limit 10
  
  # Skip already processed entries (resume)
  python parsing.py --skip-existing
  
  # Test specific IDs to verify DC50/IC50/EC50/Dmax extraction
  python parsing.py --ids TPD-O7725Z TPD-LIZKQI TPD-NMQ9M9
        """
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip TPD IDs already in output CSV'
    )
    parser.add_argument(
        '--html-dir',
        type=Path,
        default=Path('data/html/'),
        help='Directory to get the HTML files (default: data/html)'
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=Path('data/parsed/'),
        help='Directory to save the parsed HTML files (default: data/parsed)'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs/'),
        help='Directory to save log files (default: logs/)'
    )
    parser.add_argument(
        '--mol-type',
        type=str,
        choices=['AUTOTAC', 'LYTAC', 'ATTEC', 'AUTAC', 'PROTAC', 'MG'],
        default='PROTAC',
        help='Specify the molecule type to filter TPD IDs'
    )
    parser.add_argument(
        '--ids',
        nargs='+',
        metavar='TPD_ID',
        help='Process only specific TPD IDs (space-separated list, e.g., --ids TPD-O7725Z TPD-LIZKQI)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='count',
        default=0,
        help='Increase verbosity level (e.g., -v for INFO, -vv for DEBUG)'
    )
    
    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(
        log_dir=args.log_dir,
        log_base_name='tpddb_parsing',
        verbose=args.verbose
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Log file: {log_file}")
    
    num_skipped = 0
    num_parsed = 0
    num_saved = 0
    
    if args.ids is None:
        # Get all TPD IDs from the HTML directory
        html_files = [f.stem for f in args.html_dir.glob("*.html")]
        # Get all TPD IDs from the filename stems, i.e., XXXXX from TPD-XXXXX.html
        tpd_ids = set()
        for filename in html_files:
            match = re.match(r'TPD-[A-Z0-9]+', filename)
            if match:
                tpd_ids.add(match.group(0))
    else:
        tpd_ids = set(args.ids)
    
    for tpd_id in tpd_ids:
        html_path = args.html_dir / f"{tpd_id}.html"
        if not html_path.exists():
            logger.warning(f"HTML file for {tpd_id} not found at {html_path}. Skipping.")
            continue
        
        logger.info(f"Parsing TPD ID: {tpd_id}...")

        # Open and read HTML content
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        extract_functions = {
            "general_info": extract_general_information,
            "mode_of_action": extract_mode_of_action,
            "degradation_activities": extract_target_degradation_activities,
            "cytotoxic_activities": extract_cytotoxic_activities,
            "disease_info": extract_disease_info,
            "binding_affinities": extract_binding_affinities,
            "binding_preference": extract_binding_preference,
        }
        
        for i, (key, extract_func) in enumerate(extract_functions.items()):
            # Setup filenames and skip if existing
            save_path = args.save_dir / key
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f"{tpd_id}_{key}.csv"
            if output_file.exists() and args.skip_existing:
                num_skipped += 1
                logger.info(f"  - {key.replace('_', ' ').title()} already exists. Skipping.")
                continue

            logger.info(f"  - Extracting {key.replace('_', ' ').title()}...")
            
            # Extract data
            df = extract_func(html_path, html_content)
            
            if key == "binding_preference":
                # Convert dict to DataFrame
                df = pd.DataFrame([df])

            # Add TPD_ID and Molecule_Type columns
            if not df.empty:
                df["TPD_ID"] = tpd_id
                if args.mol_type:
                    df['Molecule_Type'] = args.mol_type

            # Clean activity dataframes
            cols_to_clean = [
                'degradation_activities',
                'cytotoxic_activities',
                'binding_affinities',
            ]
            if key in cols_to_clean:
                if not df.empty:
                    df = clean_activity_data(df)

            # Save to CSV
            df.to_csv(output_file, index=False)
            num_saved += 1
            logger.info(f"  - Saved to {output_file}.")

        num_parsed += 1

    report = f"\nParsing complete. Parsed: {num_parsed:,} ({num_parsed / len(tpd_ids):.2%}), Skipped: {num_skipped:,} ({num_skipped / (5 * len(tpd_ids)):.2%}), Saved: {num_saved:,} ({num_saved / (5 * len(tpd_ids)):.2%})"
    print(report)
    logger.info(report)


if __name__ == "__main__":
    main()