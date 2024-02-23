### Binding MOAD
#### Data preparation
Download the dataset
```bash
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
wget http://www.bindingmoad.org/files/csv/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```

Process the raw data using
``` bash
python -W ignore process_bindingmoad.py <bindingmoad_dir>
```
Add the `--ca_only` flag to create a dataset with $C_\alpha$ pocket representation.
If you need EC number contained in the processed data, add '--label_ec'.

## Training
For training diffusion model:
```bash
python -u train_diffusion.py --config configs/moad_ca_joint.yml
```

For training classifier model:
```bash
python -u train_classifier.py --config configs/moad_ca_joint_clf.yml
```

You can find pretrained models in 'models' directory.

### Sample molecules for a given pocket
To sample small molecules for a given pocket with a trained model use the following command:
```bash
python generate_ligands.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outdir <output_dir> --resi_list <list_of_pocket_residue_ids>
```
For example:
```bash
python generate_ligands.py last.ckpt --pdbfile 1abc.pdb --outdir results/ --resi_list A:1 A:2 A:3 A:4 A:5 A:6 A:7 
```
Alternatively, the binding pocket can also be specified based on a reference ligand in the same PDB file:
```bash 
python generate_ligands.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outdir <output_dir> --ref_ligand <chain>:<resi>
```

Optional flags:
| Flag | Description |
|------|-------------|
| `--n_samples` | Number of sampled molecules |
| `--num_nodes_lig` | Size of sampled molecules |
| `--timesteps` | Number of denoising steps for inference |
| `--all_frags` | Keep all disconnected fragments |
| `--sanitize` | Sanitize molecules (invalid molecules will be removed if this flag is present) |
| `--relax` | Relax generated structure in force field |
| `--resamplings` | Inpainting parameter (doesn't apply if conditional model is used) |
| `--jump_length` | Inpainting parameter (doesn't apply if conditional model is used) |

### Sample molecules for all pockets in the test set
`test.py` can be used to sample molecules for the entire testing set:
```bash
python test.py <checkpoint>.ckpt --test_dir <bindingmoad_dir>/processed_noH/test/ --outdir <output_dir> --sanitize
```
There are different ways to determine the size of sampled molecules. 
- `--fix_n_nodes`: generates ligands with the same number of nodes as the reference molecule
- `--n_nodes_bias <int>`: samples the number of nodes randomly and adds this bias
- `--n_nodes_min <int>`: samples the number of nodes randomly but clamps it at this value

Other optional flags are equivalent to `generate_ligands.py`. 

### Fix substructures
`inpaint.py` can be used for partial ligand redesign with the conditionally trained model, e.g.:
```bash 
python inpaint.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outdir <output_dir> --ref_ligand <chain>:<resi> --fix_atoms C1 N6 C5 C12
```
`--add_n_nodes` controls the number of newly generated nodes

### Analysis
First, convert all protein PDB files to PDBQT files using MGLTools
```bash
conda activate mgltools
cd analysis
python docking_py27.py <bindingmoad_dir>/processed_noH/test/ <output_dir> bindingmoad
cd ..
conda deactivate
```
Then, compute QuickVina scores:
```bash
conda activate sbdd-env
python analysis/docking.py --pdbqt_dir <docking_py27_outdir> --sdf_dir <test_outdir> --out_dir <qvina_outdir> --write_csv --write_dict
```


