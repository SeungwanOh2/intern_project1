from pathlib import Path
from time import time
import random
from collections import defaultdict
import argparse
import warnings
import pickle
import requests
import pickle
from copy import deepcopy
import glob

from tqdm import tqdm
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa
from Bio.PDB import PDBIO, Select
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import QED
from scipy.ndimage import gaussian_filter

from geometry_utils import get_bb_transform
from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
import constants
from constants import covalent_radii, dataset_params, ec_encoder
import utils

dataset_info = dataset_params['bindingmoad']
amino_acid_dict = dataset_info['aa_encoder']
atom_dict = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']


class Model0(Select):
    def accept_model(self, model):
        return model.id == 0


def read_label_file(csv_path):
    """
    Read BindingMOAD's label file
    Args:
        csv_path: path to 'every.csv'
    Returns:
        Nested dictionary with all ligands. First level: EC number,
            Second level: PDB ID, Third level: list of ligands. Each ligand is
            represented as a tuple (ligand name, validity, SMILES string)
    """
    ligand_dict = {}

    with open(csv_path, 'r') as f:
        for line in f.readlines():
            row = line.split(',')

            # new protein class
            if len(row[0]) > 0:
                curr_class = row[0]
                ligand_dict[curr_class] = {}
                continue

            # new protein
            if len(row[2]) > 0:
                curr_prot = row[2]
                ligand_dict[curr_class][curr_prot] = []
                continue

            # new small molecule
            if len(row[3]) > 0:
                ligand_dict[curr_class][curr_prot].append(
                    # (ligand name, validity, SMILES string)
                    [row[3], row[4], row[9]]
                )

    return ligand_dict


def compute_druglikeness(ligand_dict):
    """
    Computes RDKit's QED value and adds it to the dictionary
    Args:
        ligand_dict: nested ligand dictionary
    Returns:
        the same ligand dictionary with additional QED values
    """
    print("Computing QED values...")
    for p, m in tqdm([(p, m) for c in ligand_dict for p in ligand_dict[c]
                      for m in ligand_dict[c][p]]):
        mol = Chem.MolFromSmiles(m[2])
        if mol is None:
            mol_id = f'{p}_{m}'
            warnings.warn(f"Could not construct molecule {mol_id} from SMILES "
                          f"string '{m[2]}'")
            continue
        m.append(QED.qed(mol))
    return ligand_dict


def filter_and_flatten(ligand_dict, qed_thresh, max_occurences, seed):

    filtered_examples = []
    all_examples = [(c, p, m) for c in ligand_dict for p in ligand_dict[c]
                    for m in ligand_dict[c][p]]

    # shuffle to select random examples of ligands that occur more than
    # max_occurences times
    random.seed(seed)
    random.shuffle(all_examples)

    ligand_name_counter = defaultdict(int)
    print("Filtering examples...")
    for c, p, m in tqdm(all_examples):

        ligand_name, ligand_chain, ligand_resi = m[0].split(':')
        if m[1] == 'valid' and len(m) > 3 and m[3] > qed_thresh:
            if ligand_name_counter[ligand_name] < max_occurences:
                filtered_examples.append(
                    (c, p, m)
                )
                ligand_name_counter[ligand_name] += 1

    return filtered_examples


def ligand_list_to_dict(ligand_list):
    out_dict = defaultdict(list)
    for c, p, m in ligand_list:
        out_dict[p].append((c,m))
    return out_dict


def process_ligand_and_pocket(pdb_struct, ligand_name, ligand_chain,
                              ligand_resi, dist_cutoff, ca_only,
                              compute_quaternion=False):
    try:
        residues = {obj.id[1]: obj for obj in
                    pdb_struct[0][ligand_chain].get_residues()}
    except KeyError as e:
        raise KeyError(f'Chain {e} not found ({pdbfile}, '
                       f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    ligand = residues[ligand_resi]
    assert ligand.get_resname() == ligand_name, \
        f"{ligand.get_resname()} != {ligand_name}"

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a for a in ligand.get_atoms()
                 if (a.element.capitalize() in atom_dict or a.element != 'H')]
    lig_coords = np.array([a.get_coord() for a in lig_atoms])

    try:
        lig_one_hot = np.stack([
            np.eye(1, len(atom_dict), atom_dict[a.element.capitalize()]).squeeze()
            for a in lig_atoms
        ])
    except KeyError as e:
        raise KeyError(
            f'Ligand atom {e} not in atom dict ({pdbfile}, '
            f'{ligand_name}:{ligand_chain}:{ligand_resi})')

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)
            
    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]
    
    # Compute transform of the canonical reference frame
    n_xyz = np.array([res['N'].get_coord() for res in pocket_residues])
    ca_xyz = np.array([res['CA'].get_coord() for res in pocket_residues])
    c_xyz = np.array([res['C'].get_coord() for res in pocket_residues])

    if compute_quaternion:
        quaternion, c_alpha = get_bb_transform(n_xyz, ca_xyz, c_xyz)
        if np.any(np.isnan(quaternion)):
            raise ValueError(
                f'Invalid value in quaternion ({pdbfile}, '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    else:
        c_alpha = ca_xyz

    if ca_only:
        pocket_coords = c_alpha
        try:
            pocket_one_hot = np.stack([
                np.eye(1, len(amino_acid_dict),
                       amino_acid_dict[protein_letters_3to1[res.get_resname()]]).squeeze()
                for res in pocket_residues])
        except KeyError as e:
            raise KeyError(
                f'{e} not in amino acid dict ({pdbfile}, '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    else:
        pocket_atoms = [a for res in pocket_residues for a in res.get_atoms()
                        if (a.element.capitalize() in atom_dict or a.element != 'H')]
        pocket_coords = np.array([a.get_coord() for a in pocket_atoms])
        try:
            pocket_one_hot = np.stack([
                np.eye(1, len(atom_dict), atom_dict[a.element.capitalize()]).squeeze()
                for a in pocket_atoms
            ])
        except KeyError as e:
            raise KeyError(
                f'Pocket atom {e} not in atom dict ({pdbfile}, '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})')

    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
    }
    pocket_data = {
        'pocket_coords': pocket_coords,
        'pocket_one_hot': pocket_one_hot,
        'pocket_ids': pocket_ids,
    }
    if compute_quaternion:
        pocket_data['pocket_quaternion'] = quaternion
    return ligand_data, pocket_data


def compute_smiles_for_single_molecule(ligand_data):
    positions = np.array(ligand_data['lig_coords'])
    one_hot = ligand_data['lig_one_hot']

    atom_types = np.argmax(one_hot, axis=-1)
    positions = torch.from_numpy(positions)
    atom_types = torch.from_numpy(atom_types)
      
    mol = build_molecule(positions, atom_types, dataset_info)
    mol = rdmol_to_smiles(mol)
    return mol


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def get_bond_length_arrays(atom_mapping):
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(constants, f'bonds{i + 1}')
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


def get_lennard_jones_rm(atom_mapping):
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bond_lengths = []
            for btype in ['bonds1', 'bonds2', 'bonds3']:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bond_lengths.append(bond_dict[a1][a2])

            if len(all_bond_lengths) > 0:
                # take the shortest possible bond length because slightly larger
                # values aren't penalized as much
                bond_len = min(all_bond_lengths)
            else:
                # Replace missing values with sum of average covalent radii
                bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):

    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return atom_counts, aa_counts


def saveall(filename, pdb_and_mol_ids, lig_coords, lig_one_hot, lig_mask,
            pocket_coords, pocket_quaternion, pocket_one_hot, pocket_mask):

    np.savez(filename,
        names=pdb_and_mol_ids,
        lig_coords=lig_coords,
        lig_one_hot=lig_one_hot,
        lig_mask=lig_mask,
        pocket_coords=pocket_coords,
        pocket_quaternion=pocket_quaternion,
        pocket_one_hot=pocket_one_hot,
        pocket_mask=pocket_mask
    )
    return True


def get_ec_number_for_protein(pdb_id):
    # Define the base URL for the RCSB PDB Data API
    base_url = "https://data.rcsb.org/graphql"
    query = '{entry(entry_id: \"'+pdb_id+'\") {polymer_entities {rcsb_id rcsb_polymer_entity {pdbx_ec}}}}'
    query_alt = '{entry(entry_id: \"'+pdb_id+'\"){polymer_entities {rcsb_id rcsb_polymer_entity {rcsb_ec_lineage {depth id}}}}}'
    
    try: 
        # Make a GET request to fetch data for the specified PDB ID
        response = requests.get(f"{base_url}?query={query}")

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()

            # Extract the EC number from the protein data if available
            ec = data['data']['entry']['polymer_entities'][0]['rcsb_polymer_entity']['pdbx_ec']
            
            if ec is None:
                # get EC number by alternative query 
                response = requests.get(f"{base_url}?query={query_alt}")
                data = response.json()
                ec_hier = data['data']['entry']['polymer_entities'][0]['rcsb_polymer_entity']['rcsb_ec_lineage']
                if ec_hier is None: 
                    print(f'Failed to find ec number for {pdb_id}: no ec number for the protein')
                    return None, 0
                else: return ec_hier, 2
            else:
                return ec, 1
        else:
            print(f'Failed to find ec number for {pdb_id}: not in database')
            return None, 0
    except Exception as e:
        print(f'Failed to find ec number for {pdb_id}: {e}')
        return None, 0
    
    
def get_protein_ec_dict(data_raw, save_path, write_file=True):
    protein_ec_dict = defaultdict(list)

    # extract protein set from dataset
    for c, p, m in data_raw:
        protein_ec_dict[p] = []

    # find ec number for each protein in protein set\
    successes = 0
    for p in tqdm(protein_ec_dict.keys()):
        # ec number found by RCSB api
        ec_tmp, status = get_ec_number_for_protein(p)
        if status!=0: 
            ec_list = []
            if status==1:
                # ec number found as string 
                ec_list = ec_tmp.split(',')
            elif status==2:
                # ec number found as dictionary list
                for ec in ec_tmp: 
                    ec_list.append(ec['id'])
                    
            ec_list.sort(reverse=True)
            for ec in ec_list:
                protein_ec_dict[p].append(ec)

            successes += 1

    # save protein_ec_dict
    if write_file:
        with open(save_path, 'wb') as f:
            pickle.dump(protein_ec_dict, f)
    
    print(f'total number of proteins: {len(protein_ec_dict)}, successfully found: {successes}')
    return protein_ec_dict


def split_by_ec_number(data_raw, protein_ec_dict, save_dir='data',
                       num_train=0, num_val=300, num_test=300, ec_level=1, write_file=True):
    ec_data_idc = defaultdict(list)
    ec_stat = defaultdict(int)
    
    # filter data by whether ec numbers exist
    for idx, d in enumerate(data_raw):
        c, p, m = d
        # check if the ec number exists
        ec_list = protein_ec_dict[p]
        if len(ec_list) == 0: 
            ec_data_idc['0'].append(idx)
            ec_stat['0'] += 1
            continue

        # if multiple ec number exists for single pdb, choose the first one (temporarily)
        for ec in ec_list:
            if len(ec.split('.'))>=ec_level:
                ec = '.'.join(ec.split('.')[:ec_level])
                ec_data_idc[ec].append(idx)
                ec_stat[ec] += 1
                break

    {k:np.random.shuffle(idc) for k, idc in ec_data_idc.items()}
    ec_stat = {k:v for k,v in sorted(ec_stat.items(), key=lambda x:x[1], reverse=True)}
    n_data = len(data_raw)
    print(f"overall data: {len(data_raw)}, ec found: {sum(v for k, v in ec_stat.items() if k != '0')}")
    print(ec_stat)
    
    # split data into train, validation and test set
    data_split = {'train':[], 'val':[], 'test':[]}

    # devide indices (with ec)
    if num_train == 0:
        num_train = n_data - num_val - num_test
        
    data_split_idc_per_ec = {k: {'train': slice(0, num_train * ec_stat[k] // n_data),
                          'val': slice(num_train * ec_stat[k] // n_data, (num_val + num_train) * ec_stat[k] // n_data),
                          'test': slice((num_val + num_train) * ec_stat[k] // n_data, (num_val + num_train + num_test) * ec_stat[k] // n_data)} 
                          for k, idc in ec_data_idc.items()}
    
    
    for split in data_split.keys():
        for k, split_idc in data_split_idc_per_ec.items():
            data_split[split].extend(ec_data_idc[k][split_idc[split]])
            
        new_data_list = []
        for idx in data_split[split]:
            pocket_fn, ligand_fn = data_raw[idx][1], data_raw[idx][2][0].split('_')[-1]
            lig_name = ligand_fn.split(':')[0].split(' ')
            if len(lig_name) > 1: continue
            new_data_list.append(pocket_fn.strip() + '_' + ligand_fn.strip())
        data_split[split] = new_data_list    
        
        if write_file:
            with open(f'{save_dir}/moad_{split}.txt', 'w') as f:
                f.write(','.join(data_split[split]))
    
    return data_split, ec_stat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path)
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--qed_thresh', type=float, default=0.3)
    parser.add_argument('--max_occurences', type=int, default=50)
    parser.add_argument('--num_val', type=int, default=0)
    parser.add_argument('--num_test', type=int, default=0)
    parser.add_argument('--dist_cutoff', type=float, default=8.0)
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--make_split', action='store_true')
    parser.add_argument('--label_ec', action='store_true')
    parser.add_argument('--ec-level', type=int, default=1)
    args = parser.parse_args()

    pdbdir = args.basedir / 'BindingMOAD/'

    # Make output directory
    if args.outdir is None:
        suffix = '' if 'H' in atom_dict else '_noH'
        suffix += '_ca' if args.ca_only else '_full'
        processed_dir = Path(args.basedir, f'processed{suffix}')
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)

    ec_dict_path = processed_dir / 'ec_dict.pickle'
    if args.make_split:
        # Process the label file
        csv_path = args.basedir / 'every.csv'
        ligand_dict = read_label_file(csv_path)
        ligand_dict = compute_druglikeness(ligand_dict)
        filtered_examples = filter_and_flatten(
            ligand_dict, args.qed_thresh, args.max_occurences, args.random_seed)
        print(f'{len(filtered_examples)} examples after filtering')
        
        # find EC number for each protein
        if ec_dict_path.exists():
            with open(ec_dict_path, 'rb') as f:
                protein_ec_dict = pickle.load(f)
        else:
            protein_ec_dict = get_protein_ec_dict(filtered_examples, ec_dict_path)

        # Make data split
        num_val = int(0.01*len(filtered_examples)) if args.num_val == 0 else args.num_val
        num_test = int(0.01*len(filtered_examples)) if args.num_test == 0 else args.num_test
            
        _ = split_by_ec_number(filtered_examples, protein_ec_dict, num_val=num_val, num_test=num_test)

    else:
        with open(ec_dict_path, 'rb') as f:
            protein_ec_dict = pickle.load(f)
                
    # Use predefined data split
    data_split = {}
    for split in ['test', 'val', 'train']:
        with open(f'data/moad_{split}.txt', 'r') as f:
            pocket_ids = f.read().split(',')
        # (ec-number, protein, molecule tuple)
        data_split[split] = []
        for x in pocket_ids: 
            pocket_fn, ligand_fn = x.split('_')[0], x.split('_')[-1]
            assert len(pocket_fn)==4 and len(ligand_fn.split(':')[0].split(' ')) == 1
            ec_list = protein_ec_dict[pocket_fn]
            ec = '.'.join(ec_list[0].split('.')[:args.ec_level]) if len(ec_list) > 0 else '0'
            ec = ec_encoder[args.ec_level][ec] if ec in ec_encoder[args.ec_level] else -1
            data_split[split].append((ec, pocket_fn, (ligand_fn, )))

    n_train_before = len(data_split['train'])
    n_val_before = len(data_split['val'])
    n_test_before = len(data_split['test'])

    # Read and process PDB files
    n_samples_after = {}
    label_hist = defaultdict(int)
    for split in data_split.keys():
        lig_coords = []
        lig_one_hot = []
        lig_mask = []
        pocket_coords = []
        pocket_one_hot = []
        pocket_mask = []
        pdb_and_mol_ids = []
        receptors = []
        labels = []
        count = 0
        
        if split == 'train': train_smiles = []

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        n_tot = len(data_split[split])
        pair_dict = ligand_list_to_dict(data_split[split])

        tic = time()
        num_failed = 0
        with tqdm(total=n_tot) as pbar:
            for p in pair_dict:

                pdb_successful = set()

                # try all available .bio files
                for pdbfile in sorted(pdbdir.glob(f"{p.lower()}.bio*")):

                    # Skip if all ligands have been processed already
                    if len(pair_dict[p]) == len(pdb_successful):
                        continue

                    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)
                    struct_copy = pdb_struct.copy()

                    n_bio_successful = 0
                    for c, m in pair_dict[p]:

                        # Skip already processed ligand
                        if m[0] in pdb_successful:
                            continue

                        ligand_name, ligand_chain, ligand_resi = m[0].split(':')
                        ligand_resi = int(ligand_resi)

                        try:
                            ligand_data, pocket_data = process_ligand_and_pocket(
                                pdb_struct, ligand_name, ligand_chain, ligand_resi,
                                dist_cutoff=args.dist_cutoff, ca_only=args.ca_only)
                            if split == 'train':
                                ligand_smiles = compute_smiles_for_single_molecule(ligand_data)
                                train_smiles.append(ligand_smiles)
                        except (KeyError, AssertionError, FileNotFoundError,
                                IndexError, ValueError) as e:
                            # print(type(e).__name__, e)
                            continue

                        pdb_and_mol_ids.append(f"{p}_{m[0]}")
                        receptors.append(pdbfile.name)
                        lig_coords.append(ligand_data['lig_coords'])
                        lig_one_hot.append(ligand_data['lig_one_hot'])
                        lig_mask.append(
                            count * np.ones(len(ligand_data['lig_coords'])))
                        pocket_coords.append(pocket_data['pocket_coords'])
                        # pocket_quaternion.append(
                        #     pocket_data['pocket_quaternion'])
                        pocket_one_hot.append(pocket_data['pocket_one_hot'])
                        pocket_mask.append(
                            count * np.ones(len(pocket_data['pocket_coords'])))
                        count += 1
                        
                        if args.label_ec:
                            labels.append(c)
                            if split == 'train':
                                label_hist[c] += 1

                        pdb_successful.add(m[0])
                        n_bio_successful += 1

                        # Save additional files for affinity analysis
                        if split in {'val', 'test'}:
                        # if split in {'val', 'test', 'train'}:
                            # remove ligand from receptor
                            try:
                                struct_copy[0][ligand_chain].detach_child((f'H_{ligand_name}', ligand_resi, ' '))
                            except KeyError:
                                warnings.warn(f"Could not find ligand {(f'H_{ligand_name}', ligand_resi, ' ')} in {pdbfile}")
                                continue

                            # Create SDF file
                            atom_types = [atom_decoder[np.argmax(i)] for i in ligand_data['lig_one_hot']]
                            xyz_file = Path(pdb_sdf_dir, 'tmp.xyz')
                            utils.write_xyz_file(ligand_data['lig_coords'], atom_types, xyz_file)

                            obConversion = openbabel.OBConversion()
                            obConversion.SetInAndOutFormats("xyz", "sdf")
                            mol = openbabel.OBMol()
                            obConversion.ReadFile(mol, str(xyz_file))
                            xyz_file.unlink()

                            name = f"{p}-{pdbfile.suffix[1:]}_{m[0]}"
                            sdf_file = Path(pdb_sdf_dir, f'{name}.sdf')
                            obConversion.WriteFile(mol, str(sdf_file))

                            # specify pocket residues
                            with open(Path(pdb_sdf_dir, f'{name}.txt'), 'w') as f:
                                f.write(' '.join(pocket_data['pocket_ids']))

                    if split in {'val', 'test'} and n_bio_successful > 0:
                    # if split in {'val', 'test', 'train'} and n_bio_successful > 0:
                        # create receptor PDB file
                        pdb_file_out = Path(pdb_sdf_dir, f'{p}-{pdbfile.suffix[1:]}.pdb')
                        io = PDBIO()
                        io.set_structure(struct_copy)
                        io.save(str(pdb_file_out), select=Model0())

                pbar.update(len(pair_dict[p]))
                num_failed += (len(pair_dict[p]) - len(pdb_successful))
                pbar.set_description(f'#failed: {num_failed}')

        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_one_hot = np.concatenate(lig_one_hot, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)
        
        if args.label_ec:
            labels = np.array(labels)

        np.savez(processed_dir / f'{split}.npz', names=pdb_and_mol_ids,
                 receptors=receptors, lig_coords=lig_coords,
                 lig_one_hot=lig_one_hot, lig_mask=lig_mask,
                 pocket_coords=pocket_coords, pocket_one_hot=pocket_one_hot,
                 pocket_mask=pocket_mask,
                 labels=labels if args.label_ec else None)
        
        if split == 'train': np.save(processed_dir / 'train_smiles.npy', train_smiles)

        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic)/60.0:.2f} minutes")

    # --------------------------------------------------------------------------
    # Compute statistics & additional information
    # --------------------------------------------------------------------------
    with np.load(processed_dir / 'train.npz', allow_pickle=True) as data:
        lig_mask = data['lig_mask']
        pocket_mask = data['pocket_mask']
        lig_coords = data['lig_coords']
        lig_one_hot = data['lig_one_hot']
        pocket_one_hot = data['pocket_one_hot']

    # Joint histogram of number of ligand and pocket nodes
    n_nodes = get_n_nodes(lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)

    # Convert bond length dictionaries to arrays for batch processing
    bonds1, bonds2, bonds3 = get_bond_length_arrays(atom_dict)

    # Get bond length definitions for Lennard-Jones potential
    rm_LJ = get_lennard_jones_rm(atom_dict)

    # Get histograms of ligand and pocket node types
    atom_hist, aa_hist = get_type_histograms(lig_one_hot, pocket_one_hot,
                                             atom_dict, amino_acid_dict)

    # Create summary string
    summary_string = '# SUMMARY\n\n'
    summary_string += '# Before processing\n'
    summary_string += f'num_samples train: {n_train_before}\n'
    summary_string += f'num_samples val: {n_val_before}\n'
    summary_string += f'num_samples test: {n_test_before}\n\n'
    summary_string += '# After processing\n'
    summary_string += f"num_samples train: {n_samples_after['train']}\n"
    summary_string += f"num_samples val: {n_samples_after['val']}\n"
    summary_string += f"num_samples test: {n_samples_after['test']}\n\n"
    summary_string += '# Info\n'
    summary_string += f"'atom_encoder': {atom_dict}\n"
    summary_string += f"'atom_decoder': {list(atom_dict.keys())}\n"
    summary_string += f"'aa_encoder': {amino_acid_dict}\n"
    summary_string += f"'aa_decoder': {list(amino_acid_dict.keys())}\n"
    summary_string += f"'bonds1': {bonds1.tolist()}\n"
    summary_string += f"'bonds2': {bonds2.tolist()}\n"
    summary_string += f"'bonds3': {bonds3.tolist()}\n"
    summary_string += f"'lennard_jones_rm': {rm_LJ.tolist()}\n"
    summary_string += f"'atom_hist': {atom_hist}\n"
    summary_string += f"'aa_hist': {aa_hist}\n"
    summary_string += f"'n_nodes': {n_nodes.tolist()}\n"
    
    if args.label_ec:
        summary_string += f"'label_hist': {label_hist}\n"
        with open(processed_dir / 'label_hist.pickle', 'wb') as f:
            pickle.dump(label_hist, f)

    # Write summary to text file
    with open(processed_dir / 'summary.txt', 'w') as f:
        f.write(summary_string)

    # Print summary
    print(summary_string)
