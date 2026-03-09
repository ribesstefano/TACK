import requests
import time
import re
from typing import Optional, Union, Literal
from functools import lru_cache

import pandas as pd


@lru_cache()
def fetch_uniprot_entry(uniprot_id: str) -> Optional[dict]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        time.sleep(0.5)  # To avoid hitting the API too hard
        return r.json()
    except Exception as e:
        print(f"[UniProt] fetch failed for {uniprot_id}: {e}")
        return None

def extract_protein_info(uniprot_id: str, skip_isoforms: bool = False) -> Optional[dict]:
    """ Extracts detailed information about a protein from UniProt.
        Fetch and filter the Uniprot JSON entries.
    
    Args:
        uniprot_id (str): The UniProt ID of the protein.
        skip_isoforms (bool): If True, skips fetching isoform information.
        
    Returns:
        dict: A dictionary containing the protein information, or None if the fetch fails. List of keys:
            - 'accession': Primary accession number.
            - 'secondary_accessions': List of secondary accession numbers.
            - 'sequence': Canonical sequence of the protein.
            - 'full_names': List of full names of the protein.
            - 'short_names': List of short names of the protein.
            - 'isoforms': List of isoform information (if not skipped).
            - 'locations': List of subcellular locations.
            - 'natural_variants': List of natural variant sequences.
            - 'natural_variants_ids': List of IDs for the natural variants.
    """
    # Fetch the UniProt entry
    entry = fetch_uniprot_entry(uniprot_id)
    if not entry:
        print(f"[UniProt] {uniprot_id} fetch failed.")
        return None

    # Setup the information dictionary to return
    info = {
        'accession': entry.get('primaryAccession'),
        'secondary_accessions': entry.get('secondaryAccessions'),
        'sequence': entry.get('sequence', {}).get('value'),
        'organism': entry.get('organism', {}).get('scientificName'),
    }

    # Obtain full names and short names
    alternative_names = entry.get('proteinDescription', {}).get('alternativeNames', [])
    info['full_names'] = [n.get('fullName', {}).get('value', 'N/A') for n in alternative_names]
    info['short_names'] = [n.get('value', 'N/A') for an in alternative_names for n in an.get('shortNames', [])]

    # Parse comments for isoforms and locations in cell
    info['isoforms'] = []
    info['locations'] = []
    comments = entry.get('comments', [])
    for comment in comments:
        # Get isoforms IDs if present, they will be recursively fetched later
        if comment.get('commentType', '') == 'ALTERNATIVE PRODUCTS':
            if not skip_isoforms:
                for isoform in comment.get('isoforms', []):
                    if isoform.get('isoformIds'):
                        info['isoforms'] += isoform['isoformIds']

        # Get subcellular locations
        elif comment.get('commentType', '') == 'SUBCELLULAR LOCATION':
            for location in comment.get('subcellularLocations', []):
                location = location.get('location', {})
                if location.get('value'):
                    info['locations'].append(location['value'])
    
    # Ensure locations are lowercase and unique
    locations = []
    for loc in info['locations']:
        for l in loc.split(', '):
            l = l.strip().lower()
            if l not in locations:
                locations.append(l)
    info['locations'] = locations

    info['natural_variants'] = []
    info['natural_variants_ids'] = []
    features = entry.get('features', [])
    for feature in features:
        if feature.get('type', '') == 'Natural variant':
            loc = feature.get('location', {})
            start = loc.get('start', {}).get('value')
            end = loc.get('end', {}).get('value')
            if start is not None and end is not None:
                alt_seq_info = feature.get('alternativeSequence', {})
                original_seq = alt_seq_info.get('originalSequence', '')
                alt_seq = ''.join(alt_seq_info.get('alternativeSequences', ['']))
                if info['sequence'][start-1:end] != original_seq:
                    print(f"[WARNING] Sequence mismatch for {uniprot_id} at {start}-{end}: {info['sequence'][start-1:end]} != {original_seq}")
                natural_variant = (
                    info['sequence'][:start-1] + alt_seq + info['sequence'][end:]
                )
                info['natural_variants'].append(natural_variant)
                info['natural_variants_ids'].append(feature.get('featureId'))

    # If isoforms are not skipped, fetch their details recursively
    # NOTE: Recursion is disabled within an isoform extraction.
    info['isoforms'] = [extract_protein_info(iso_id, skip_isoforms=True) for iso_id in info['isoforms']]
    info['isoforms'] = [iso for iso in info['isoforms'] if iso is not None]
    
    return info

def clean_target(t: str) -> str:
    """ Clean the target string by removing special characters and extra spaces. """
    if pd.isnull(t):
        return t
    t = t.replace(' and ', '/').upper().strip()
    letter2unicode = {
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ',
        'epsilon': 'ε',
    }
    # Replace all greek letters
    for letter in ['alpha', 'beta', 'gamma', 'delta', 'epsilon']:
        t = t.replace(letter.upper(), letter)
        if letter in t:
            # If the greek letter is not preceded by a -, add it in front
            # If the greek letter is preceded by a space, substitute it with a hyphen
            i = t.find(letter)
            if i > 0 and t[i - 1] != '-':
                if t[i - 1] == ' ':
                    t = t[:i - 1] + '-' + t[i:]
                else:
                    t = t[:i] + '-' + t[i:]
            # If the greek letter is not at the end, and its next letter is a character and not a hyphen, add a hyphen between them
            if i + len(letter) + 1 < len(t) and t[i + len(letter)] not in ['-', ' ']:
                t = t[:i + len(letter)] + '-' + t[i + len(letter):]
            # # Finally, replace the letter with its unicode equivalent
            # t = t.replace('-' + letter, letter2unicode[letter])
            # t = t.replace(letter + '-', letter2unicode[letter])

    # If a mutation, with pattern like CharacterNumbersCharacter, is preceded by a -, substitute the - with a space
    pattern = r'-(\w+\d+\w+)'
    # Replace the pattern with a space before the mutation
    t = re.sub(pattern, r' \1', t)
    # Remove any trailing slashes or hyphens
    t = t.replace('- ', ' ')
    t = t.replace('  ', ' ')
    # Replace any mention to "fusion" or "fusion protein" with "(fusion)" at the end of the string
    if '(FUSION)' in t:
        t = t.replace('(FUSION)', '(fusion)')
    elif 'FUSION' in t or 'FUSION PROTEIN' in t:
        t = t.replace('FUSION PROTEIN', '(fusion)').replace('FUSION', '(fusion)')
    # Remove any trailing spaces
    t = t.strip()

    # Remove any mention in any case of "[-]hibit[-]"
    t = re.sub(r'[- ]?HIBIT[- ]?', '', t).strip()
    t = re.sub(r'[- ]?HiBit[- ]?', '', t).strip()
    t = re.sub(r'[- ]?Hibit[- ]?', '', t).strip()
    t = re.sub(r'[- ]?hibit[- ]?', '', t).strip()

    # Replace 'HDAC ' followed by a number with 'HDAC' followed by the number
    t = re.sub(r'HDAC (\d+)', r'HDAC\1', t).strip()

    # Remove 'HUMAN', as we already assume all targets are human by default
    t = t.replace('HUMAN', '').strip()
    t = t.replace('BROMODOMAIN', 'BD').strip()

    # # Remove ' BD' and ' BD[digit]' from the target
    # t = re.sub(r' BD\d*', '', t).strip()

    # Corner cases
    corner_cases = {
        'PDGFR-beta': 'PDGFRB',
        'CDK12-CYCLINK': 'CDK12-CYCLIN-K',
        'CDK6-CYCLIN D1': 'CDK6-CYCLIN-D1',
        'CDK6-CYCLIN D3': 'CDK6-CYCLIN-D3',        
        'CDK9-CYCT1': 'CDK9-CYCLIN-T1',
        'CDK4-CYCLIN D1': 'CDK4-CYCLIN-D1',
        'CDK4-CYCLIN D3': 'CDK4-CYCLIN-D3',
        'BTK OF HEK293 CELLS': 'BTK',
        'BPTF BROMODOMAIN': 'BPTF BD',
        'CECR2 BROMODOMAIN': 'CECR2 BD',
        'BRD9 BROMODOMAIN': 'BRD9 BD',
        'CA2': 'HCA2',
        'AR V7': 'AR-V7',
        'EGFRDEL19': 'EGFR DEL19',
        'GSK-3BATA': 'GSK3B', # 'GSK-3-beta',
        'GSK-3-beta': 'GSK3B',
        'alpha-SYN': 'alpha-SYNUCLEIN',
        'P300': 'EP300',
        'H-PGDS': 'HPGDS',
        'B-RAF': 'BRAF',
        'TCPTP': 'TC-PTP',
        'P STAT3Y705': 'p-STAT3-Y705',
        'P P38': 'p-P38',
        'HADC6': 'HDAC6',
        'BRD4-L': 'BRD4 LONG',
        'FLT-3': 'FLT3',
        'G1202R ALK': 'ALK G1202R',
        'G2019S LRRK2': 'LRRK2 G2019S',
        'HCAII': 'HCA2',
    }

    if t in corner_cases:
        return corner_cases[t]

    return t

def apply_mutation(
        seq: str,
        gene: str,
        on_error: Union[bool, Literal['raise', 'ignore']] = 'raise',
        verbose: int = 0,
) -> str:
    """ Apply the mutation to the sequence, if possible.
    
    Args:
        uniprot (str): The UniProt ID of the protein.
        gene (str): The gene name or mutation description.
        seq (str): The original protein sequence.
        on_error (str): What to do on error ('raise' or 'ignore').
        
    Returns:
        str: The mutated sequence if the mutation is valid, otherwise the original sequence.
        
    Raises:
        ValueError: If the mutation cannot be applied and `on_error` is 'raise'.
    """
    # Check if both gene and sequence are not nan
    if pd.isna(gene) or pd.isna(seq):
        return seq

    # # TODO: Just use a dictionary and replace these sequences straightaway...
    # uniprot_exceptions = {
    #     ('O60885', 'BRD4 BD1'): uniprot2sequence['O60885'],
    #     ('P25440', 'BRD2 BD2'): uniprot2sequence['P25440'],
    #     ('P10275', 'AR-V7'): uniprot2sequence['P10275'],
    #     # TODO: Not working... why???
    #     ('P00533', 'EGFR e19d'): uniprot2sequence['P10275'],
    # }
    # # Handle exceptions
    # if (uniprot, gene) in uniprot_exceptions:
    #     return uniprot_exceptions[(uniprot, gene)]

    # Use regex to get all mutations in the gene string
    if re.search(r'\b[A-Z]\d+[A-Z]\b', gene) or re.search(r'\bDEL', gene):
        mutations = re.findall(r'\b[A-Z]\d+[A-Z]\b|\bDEL\d+\b', gene.upper())
    else:
        return seq

    if verbose > 0:
        print(f'Applying mutations: {mutations} to sequence: {seq} (length: {len(seq)})')

    original_seq = seq
    del_ops = 0
    for op in mutations:
        if 'del' in op.lower():
            idx = int(op.lower().split('del')[1]) - 1
            seq = seq[:idx] + seq[idx + 1:]
            del_ops += 1
        else:
            # Replace aminoacid at a specific index
            # NOTE: The indexing starts from one, not zero.
            curr, idx, mutation = op[0].upper(), int(op[1:-1])-1, op[-1].upper()
            # NOTE: If a deletion has happened before, the index is still
            # relative to the whole sequence lenght (weird...)
            idx -= del_ops
            if verbose > 1:
                print(f'Operation: {op} on ...{seq[idx-8:idx]}[{seq[idx]} -> {mutation}]{seq[idx+1:idx+8]}...')
            if idx < 0 or idx >= len(seq):
                msg = f'Index {idx} out of bounds for sequence of length {len(seq)}.'
                if on_error == 'raise' or on_error is True:
                    raise ValueError('ERROR. ' + msg)
                else:
                    if verbose > 0:
                        print('WARNING. ' + msg + ' No mutation is applied.')
                    return original_seq

            if curr != seq[idx]:
                msg = f'Replacement at position {idx} failed. Expected "{curr}", found: "{seq[idx]}".'
                if on_error == 'raise' or on_error is True:
                    raise ValueError('ERROR. ' + msg)
                else:
                    if verbose > 0:
                        print('WARNING. ' + msg + ' No mutation is applied.')
                    return original_seq
            seq = seq[:idx] + mutation + seq[idx + 1:]

    return seq

def assign_missing_uniprot(
        row,
        df_name,
        force_replace: bool = False,
        targets_w_no_uniprot_mapping: dict = {},
        target2uniprots: dict = {},
) -> Optional[str]:
    if pd.notna(row['Uniprot']):
        return row['Uniprot']

    target = row[f'Target']
    target_parsed = row[f'Target {df_name}']

    # NOTE: The specific target column will have precedence over the generic
    # 'Target' column from the original PROTAC-DB dataframe.
    if target_parsed in targets_w_no_uniprot_mapping:
        return targets_w_no_uniprot_mapping[target_parsed]

    if target in targets_w_no_uniprot_mapping:
        return targets_w_no_uniprot_mapping[target]

    uniprots = target2uniprots.get(target_parsed, [])
    if len(uniprots) == 1:
        return uniprots[0]

    uniprots = target2uniprots.get(target, [])
    if len(uniprots) == 1:
        return uniprots[0]

    return pd.NA


def update_e3ligase_uniprot(row, e3ligase2uniprot):
    e3 = row['E3 Ligase']
    cell_species = row['Cell Species']
    current_uniprot = row['E3 Ligase Uniprot']

    if pd.isna(e3) or pd.isna(cell_species):
        return current_uniprot
    
    uniprot = e3ligase2uniprot.get(cell_species, {}).get(e3)
    if uniprot:
        return uniprot
    return current_uniprot

def update_e3ligase_sequence(row, uniprot2infos):
    uniprot_id = row['E3 Ligase Uniprot']
    current_sequence = row['E3 Ligase Sequence']

    if pd.isna(uniprot_id):
        return current_sequence
    
    info = uniprot2infos.get(uniprot_id)
    if info and 'sequence' in info:
        return info['sequence']
    return current_sequence

def get_poi_species(uniprot, uniprot2infos):
    if pd.isna(uniprot):
        return None
    infos = uniprot2infos.get(uniprot)
    if infos and 'organism' in infos:
        return infos['organism']
    return None

def get_sequence_from_uniprot(uniprot_id, uniprot2infos):
    if pd.isnull(uniprot_id):
        return None
    info = uniprot2infos.get(uniprot_id)
    if info is not None and 'sequence' in info:
        return info['sequence']
    return None

def apply_mutation_to_sequence(row):
    uniprot = row['Uniprot']
    seq = row['POI Sequence']
    target = row['Target']
    
    if pd.isnull(seq) or pd.isnull(target):
        return seq
    try:
        mutated_seq = apply_mutation(seq, target, on_error='ignore', verbose=0)
        return mutated_seq
    except ValueError as e:
        print(f"Applying mutation for {row['Uniprot']} with target '{target}'")
        print(f"Error applying mutation for {row['Uniprot']} with target '{target}': {e}")
        return seq

def map_poi_uniprot_from_species(row, species2uniprot):
    uniprot = row['Uniprot']
    poi_species = row['Cell Species']
    if pd.isna(uniprot) or pd.isna(poi_species):
        return uniprot
    if poi_species in species2uniprot and uniprot in species2uniprot[poi_species]:
        return species2uniprot[poi_species][uniprot]
    return uniprot

def map_poi_sequence_from_uniprot(row, uniprot2infos):
    uniprot = row['Uniprot']
    seq = row['POI Sequence']
    if pd.isna(uniprot):
        return seq
    if uniprot in uniprot2infos:
        return uniprot2infos[uniprot]['sequence']
    return seq

def add_location(row, uniprot2locations: dict):
    """Add the location information to the row based on the Uniprot ID."""
    uniprot = row['Uniprot']
    if pd.isna(uniprot):
        return row

    locations = uniprot2locations[uniprot]
    
    for loc, val in locations.items():
        row[f'Location: {loc}'] = val

    return row


if __name__ == "__main__":

    # Test the apply_mutation function with a known mutation and sequence
    uniprot_id = 'P00533'
    target = 'EGFR DEL19/T790M/C797S'

    seq = 'MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA'
    mutated_seq = apply_mutation(seq, target, on_error='ignore', verbose=1)

    print(f'Original sequence for {uniprot_id}: {seq}')
    print(f'Mutated sequence for {uniprot_id}:  {mutated_seq}')
    assert seq != mutated_seq, f'Sequence for {uniprot_id} was not mutated as expected: {seq} == {mutated_seq}'
    