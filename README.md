# P-GUTS

### Installation

1. **Set up the environment**  
  Create a fresh Conda or Mamba environment. If an `environment.yml` file is provided, use it to set up dependencies:
  ```bash
  mamba env create -f environment.yml
  ```

2. **Install PyG dependencies**  
  Some PyTorch Geometric (PyG) dependencies need to be installed separately. Use the following command:
  ```bash
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
  ```

3. **Fix a known issue in the `tsl` library**  
  The code relies on the `tsl` library, which has a minor bug (see [GitHub Issue #15](https://github.com/Graph-Machine-Learning-Group/spin/issues/15)).  

  **Error Location:**
  ```
  File "/home/a/miniforge3/envs/spin_env/lib/python3.8/site-packages/tsl/data/batch.py", line 46, in static_graph_collate
  ```

  **Quick Fix:**  
  Edit the `static_graph_collate` function in the file mentioned above. Replace:
  ```python
     for k in elem.keys:
  ```
  with:
  ```python
     for k in elem.pattern.keys():
  ```

### Running

To execute the code, use the `experiments.run_imputation` script along with a YAML configuration file.

#### Example: Running on the `air36` dataset
Run the following command:
```bash
python -m experiments.run_imputation --config imputation/air36/pguts.yaml --dataset-name air36
```

#### Running on other datasets
To use other datasets, replace `air36` in the command above with one of the following:
- `air`
- `la_block`
- `bay_block`

#### Configuration Tweaks
You can modify the YAML configuration files in the `configuration` folder. For example, to use pooling factors of 3 and 6, set:
```yaml
factor_t: [3, 6]
```
