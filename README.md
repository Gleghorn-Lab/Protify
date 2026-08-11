<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Gleghorn-Lab/Protify">
    <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/github_banner.png" alt="Logo">
  </a>

  <h3 align="center">Protify</h3>

  <p align="center">
    A low code solution for computationally predicting the properties of chemicals.
    <br />
    <a href="https://github.com/Gleghorn-Lab/Protify/tree/main/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Gleghorn-Lab/Protify">View Demo</a>
    &middot;
    <a href="https://github.com/Gleghorn-Lab/Protify/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/Gleghorn-Lab/Protify/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#why-protify">Why Protify?</a></li>
        <li><a href="#current-key-features">Current Key Features</a></li>
        <li><a href="#support-protifys-development">Support Protify's Development</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li>
      <a href="#hyperparameter-optimization">Hyperparameter Optimization</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#quick-start">Quick Start</a></li>
        <li><a href="#configuration">Configuration</a></li>
        <li><a href="#cli-arguments">CLI Arguments</a></li>
        <li><a href="#available-metrics">Available Metrics</a></li>
        <li><a href="#workflow">Workflow</a></li>
        <li><a href="#best-practices">Best Practices</a></li>
      </ul>
    </li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#cite">Cite</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Protify is an open source platform designed to simplify and democratize workflows for chemical language models. With Protify, deep learning models can be trained to predict chemical properties at the click of a button, without requiring extensive coding knowledge or computational resources.

### Why Protify?

- **Benchmark multiple models efficiently**: Need to evaluate 10 different protein language models against 15 diverse datasets with publication-ready figures? Protify makes this possible without writing a single line of code.
- **Flexible for all skill levels**: Build custom pipelines with code or use our no-code interface depending on your needs and expertise.
- **Accessible computing**: No GPU? No problem. Synthyra offers precomputed embeddings for many popular datasets, which Protify can download for analysis with scikit-learn on your laptop.
- **Cost-effective solutions**: Run GPU training jobs through the [Synthyra API](https://synthyra.com) with a simple API key. No GPU setup required. Get your key at [synthyra.com](https://synthyra.com).

Protify is currently in beta. We're actively working to enhance features and documentation to meet our ambitious goals.

## Currently Supported Models

<details>
  <summary>Click to expand model list</summary>
  
  pLM - Protein Language Model

  | Model Name | Description | Size (parameters) | Type |
  |------------|-------------|------|------|
  | ESM2-8 | Very small pLM from Meta AI that learns evolutionary information from millions of protein sequences. | 8M | pLM |
  | ESM2-35 | Small-sized pLM trained on evolutionary data. | 35M | pLM |
  | ESM2-150 | Medium-sized pLM with improved protein structure prediction capabilities. | 150M | pLM |
  | ESM2-650 | Large pLM offering state-of-the-art performance on many protein prediction tasks. | 650M | pLM |
  | ESM2-3B | Largest ESM2 pLM with exceptional capability for protein structure and function prediction. | 3B | pLM |
  | ESMC-300 | pLM optimized for representation learning. | 300M | pLM |
  | ESMC-600 | Larger pLM for representations. | 600M | pLM |
  | ESMC-6B | 6B ESM++/ESMC pLM for high-capacity representations. | 6.352B | pLM |
  | ProtBert | BERT-based pLM trained on protein sequences from UniRef. | 420M | pLM |
  | ProtBert-BFD | BERT-based pLM trained on BFD database with improved performance. | 420M | pLM |
  | ProtT5 | T5-based pLM capable of both encoding and generation tasks. | 3B | pLM |
  | ANKH-Base | Base version of the ANKH pLM focused on protein structure understanding. | 400M | pLM |
  | ANKH-Large | Large version of the ANKH pLM with improved structural predictions. | 1.2B | pLM |
  | ANKH2-Large | Improved second generation ANKH pLM. | 1.2B | pLM |
  | GLM2-150 | Medium-sized general language model adapted for protein sequences. | 150M | pLM |
  | GLM2-650 | Large general language model adapted for protein sequences. | 650M | pLM |
  | GLM2-GAIA | Specialized GLM pLM fine-tuned with contrastive learning. | 650M | pLM |
  | DPLM-150 | Diffusion pLM focused on protein structure. | 150M | pLM |
  | DPLM-650 | Larger diffusion pLM focused on protein structure. | 650M | pLM |
  | DPLM-3B | Largest deep protein language model in the DPLM family. | 3B | pLM |
  | DPLM2-150 | Second-generation diffusion protein language model, 150M parameters. | 150M | pLM |
  | DPLM2-650 | Second-generation diffusion protein language model, 650M parameters. | 650M | pLM |
  | DPLM2-3B | Second-generation diffusion protein language model, 3B parameters. | 3B | pLM |
  | DSM-150 | Diffusion sequence model 150 parameter version. | 150M | pLM |
  | DSM-650 | Diffusion sequence model 650 parameter version. | 650M | pLM |
  | DSM-PPI | DSM model optimized for protein-protein interactions. | Varies | pLM |
  | E1-150 | Profluent E1 protein language model, 150M parameters. | 150M | pLM |
  | E1-300 | Profluent E1 protein language model, 300M parameters. | 300M | pLM |
  | E1-600 | Profluent E1 protein language model, 600M parameters. | 600M | pLM |
  | AMPLIFY-120 | AMPLIFY protein language model, 120M parameters. | 120M | pLM |
  | AMPLIFY-350 | AMPLIFY protein language model, 350M parameters. | 350M | pLM |
  | CaLM | Codon-aware language model from MultiMolecule. | Varies | pLM |
  | OneHot-Protein | One-hot encoding baseline for protein sequences. | N/A | Baseline |
  | OneHot-DNA | One-hot encoding baseline for DNA sequences. | N/A | Baseline |
  | OneHot-RNA | One-hot encoding baseline for RNA sequences. | N/A | Baseline |
  | OneHot-Codon | One-hot encoding baseline for codon sequences. | N/A | Baseline |
  | Random | Baseline model with randomly initialized weights, serving as a negative control. | Varies | Negative control |
  | Random-Transformer | Randomly initialized transformer model serving as a homology-based control. | Varies | Homology control |
  | Random-ESM2-8 | Randomly initialized ESM2-8M architecture. | 8M | Negative control |
  | Random-ESM2-35 | Randomly initialized ESM2-35M architecture. | 35M | Negative control |
  | Random-ESM2-150 | Randomly initialized ESM2-150M architecture. | 150M | Negative control |
  | Random-ESM2-650 | Randomly initialized ESM2-650M architecture. | 650M | Negative control |
</details>

## Currently Supported Datasets

<details>
  <summary>Click to expand dataset list</summary>
  
  BC - Binary Classification | SLC - Single-Label Classification | MLC - Multi-Label Classification | R - Regression

  TC - Tokenwise classification | TR - Tokenwise regression

  | Dataset Name | Description | Type | Task | Tokenwise | Multiple inputs |
  |--------------|-------------|------|------|-----------|-------------|
  | EC | Enzyme Commission numbers dataset for predicting enzyme function classification. | MLC | Protein function prediction | No | No |
  | GO-CC | Gene Ontology Cellular Component dataset for predicting protein localization in cells. | MLC | Protein localization prediction | No | No |
  | GO-BP | Gene Ontology Biological Process dataset for predicting protein involvement in biological processes. | MLC | Protein function prediction | No | No |
  | GO-MF | Gene Ontology Molecular Function dataset for predicting protein molecular functions. | MLC | Protein function prediction | No | No |
  | MB | Metal ion binding dataset for predicting protein-metal interactions. | BC | Protein-metal binding prediction | No | No |
  | DeepLoc-2 | Binary classification dataset for predicting protein localization in 2 categories. | BC | Protein localization prediction | No | No |
  | DeepLoc-10 | Multi-class classification dataset for predicting protein localization in 10 categories. | MCC | Protein localization prediction | No | No |
  | Subcellular | Dataset for predicting subcellular localization of proteins. | MCC | Protein localization prediction | No | No |
  | enzyme-kcat | Dataset for predicting enzyme catalytic rate constants (kcat). | R | Enzyme kinetics prediction | No | No |
  | solubility | Dataset for predicting protein solubility properties. | BC | Protein solubility prediction | No | No |
  | localization | Dataset for predicting subcellular localization of proteins. | MCC | Protein localization prediction | No | No |
  | temperature-stability | Dataset for predicting protein stability at different temperatures. | BC | Protein stability prediction | No | No |
  | optimal-temperature | Dataset for predicting the optimal temperature for protein function. | R | Protein property prediction | No | No |
  | optimal-ph | Dataset for predicting the optimal pH for protein function. | R | Protein property prediction | No | No |
  | material-production | Dataset for predicting protein suitability for material production. | BC | Protein application prediction | No | No |
  | fitness-prediction | Dataset for predicting protein fitness in various environments. | BC | Protein fitness prediction | No | No |
  | number-of-folds | Dataset for predicting the number of structural folds in proteins. | BC | Protein structure prediction | No | No |
  | cloning-clf | Dataset for predicting protein suitability for cloning operations. | BC | Protein engineering prediction | No | No |
  | stability-prediction | Dataset for predicting overall protein stability. | BC | Protein stability prediction | No | No |
  | SecondaryStructure-3 | Dataset for predicting protein secondary structure in 3 classes. | MCC | Protein structure prediction | Yes | No |
  | SecondaryStructure-8 | Dataset for predicting protein secondary structure in 8 classes. | MCC | Protein structure prediction | Yes | No |
  | fluorescence-prediction | Dataset for predicting protein fluorescence properties. | R | Protein property prediction | Yes | No |
  | plastic | Dataset for predicting protein capability for plastic degradation. | BC | Enzyme function prediction | No | No |
  | gold-ppi | Gold standard dataset for protein-protein interaction prediction. | SLC | PPI prediction | No | Yes |
  | human-ppi-saprot | Human protein-protein interaction dataset from SAProt paper. | SLC | PPI prediction | No | Yes |
  | human-ppi-pinui | Human protein-protein interaction dataset from PiNUI. | SLC | PPI prediction | No | Yes |
  | yeast-ppi-pinui | Yeast protein-protein interaction dataset from PiNUI. | SLC | PPI prediction | No | Yes |
  | peptide-HLA-MHC-affinity | Dataset for predicting peptide binding affinity to HLA/MHC complexes. | SLC | Binding affinity prediction | No | Yes |
  | shs27-ppi-raw | Raw SHS27k with single-label labels. | SLC | PPI type prediction | No | Yes |
  | shs148-ppi-raw | Raw SHS148k with single-label labels. | SLC | PPI type prediction | No | Yes |
  | shs27-ppi-random | SHS27k  | MLC | PPI prediction | No | Yes |
  | shs148-ppi-random | SHS148k CD-Hit 40%, multi-label lables, randomized data splits. | MLC | PPI type prediction | No | Yes |
  | shs27-ppi-dfs | SHS27k CD-Hit 40%, multi-label lables, data splits via depth first search. | MLC | PPI type prediction | No | Yes |
  | shs148-ppi-dfs | SHS148k CD-Hit 40%, multi-label lables, data splits via depth first search. | MLC | PPI type prediction | No | Yes |
  | shs27-ppi-bfs | SHS27k CD-Hit 40%, multi-label lables, data splits via breadth first search. | MLC | PPI type prediction | No | Yes |
  | shs148-ppi-bfs | SHS148k CD-Hit 40%, multi-label lables, data splits via breadth first search. | MLC | PPI type prediction | No | Yes |
  | string-ppi-random | STRING CD-Hit 40%, multi-label lables, randomized data splits. | MLC | PPI type prediction | No | Yes |
  | string-ppi-dfs | STRING CD-Hit 40%, multi-label lables, data splits via depth first search. | MLC | PPI type prediction | No | Yes |
  | string-ppi-bfs | STRING CD-Hit 40%, multi-label lables, data splits via breadth first search. | MLC | PPI type prediction | No | Yes |
  | ppi-mutation-effect | Compare wild type, mutated, and target sequence to determine if PPI is stronger or not. | SLC | PPI effect prediction | No | Yes |
  | PPA-ppi | Protein-Protein Affinity dataset from Bindwell. | R | protein-protein affinity prediction | No | Yes |
  | foldseek-fold | Dataset for protein fold classification using Foldseek. | MCC | Protein structure prediction | No | No |
  | foldseek-inverse | Inverse protein fold prediction dataset. | MCC | Protein structure prediction | No | No |
  | ec-active | Dataset for predicting active enzyme classes. | MCC | Enzyme function prediction | No | No |
  | taxon_domain | Taxonomic classification at domain level. | MCC | Taxonomic prediction | No | No |
  | taxon_kingdom | Taxonomic classification at kingdom level. | MCC | Taxonomic prediction | No | No |
  | taxon_phylum | Taxonomic classification at phylum level. | MCC | Taxonomic prediction | No | No |
  | taxon_class | Taxonomic classification at class level. | MCC | Taxonomic prediction | No | No |
  | taxon_order | Taxonomic classification at order level. | MCC | Taxonomic prediction | No | No |
  | taxon_family | Taxonomic classification at family level. | MCC | Taxonomic prediction | No | No |
  | taxon_genus | Taxonomic classification at genus level. | MCC | Taxonomic prediction | No | No |
  | taxon_species | Taxonomic classification at species level. | MCC | Taxonomic prediction | No | No |
  | diff_phylogeny | Differential phylogeny dataset. | Various | Phylogeny prediction | No | No |
  | plddt | AlphaFold pLDDT confidence score prediction. | TR | Confidence prediction | Yes | No |
  | realness | Protein realness dataset. | BC | Authenticity prediction | No | No |
  | million_full | Large-scale enzyme variant dataset, from Millionfull preprint October 2025 | R | Protein fitness prediction | No | No |
  | soluprot | Protein solubility prediction from SoluProt. | BC | Protein solubility prediction | No | No |
  | ecoli_expression | E. coli protein expression level prediction. | BC | Expression prediction | No | No |
  | KSMoFinder | Kinase substrate prediction dataset. | SLC | Enzyme function prediction | No | Yes |
  | KSMoFinder-clustered | Clustered kinase substrate prediction dataset. | SLC | Enzyme function prediction | No | Yes |
  | plm-interact | PLM-Interact human PPI cross-species dataset. | SLC | PPI prediction | No | Yes |
</details>

For more details about supported models and datasets, including programmatic access and command-line utilities, see the [Resource Listing Documentation](docs/resource_listing.md).

### Current Key Features

- **Multiple interfaces**: Run experiments via an intuitive GUI, CLI, or prepared YAML files
- **Efficient embeddings**: Leverage fast and efficient embeddings from 46+ PLMs (ESM2, ESMC, E1, ProtBert, ProtT5, ANKH, GLM2, DPLM, DPLM2, DSM, AMPLIFY, CaLM, and more) via [FastPLMs](https://github.com/Synthyra/FastPLMs)
- **Sparse autoencoder features**: Read ESMC through the Biohub sparse autoencoders with `ESMC-300-SAE`, `ESMC-600-SAE`, or `ESMC-6B-SAE`, choosing any published layer, sparsity, and codebook width. FastPLMs runs the SAE and returns sparse per-residue features; Protify reduces them to one vector per protein and stores wide codebooks sparsely, so a 131072-wide codebook costs roughly a ninth of dense storage. See [models and embeddings](docs/models_and_embeddings.md)
- **Estimator probes**: `--probe_type xgboost`, `lightgbm`, or `random_forest` fit gradient-boosted or forest models on pooled embeddings through the same pipeline, cache, and results table as the neural probes. Axis-aligned splits read sparse autoencoder features directly
- **Flexible model training**: Probe-only (frozen PLM), full fine-tuning, hybrid probing, LoRA via PEFT, and vectorized multi-seed MLP probes
- **Parallel probe sweeps**: Train many seeded pooled MLP probes in one vectorized pass with `--parallel_probe_runs`, including run-specific deterministic shuffles and static preflight planning for workstation launches
- **Automated model selection**: Find optimal scikit-learn models for your data with LazyPredict, enhanced by automatic hyperparameter optimization
- **Hyperparameter optimization**: Integrated Weights & Biases sweeps that conducts a hyperparameter search and trains the final version based on the best hyperparameters
- **Complete reproducibility**: Every session generates a detailed log that can be used to reproduce your entire workflow
- **Publication-ready visualizations**: Generate cross-model and dataset comparisons, confidence interval plots, dimensionality reductions, and [multi-algorithm clustering diagnostics](docs/clustering.md) from saved embeddings
- **Extensive dataset support**: Access 60+ protein datasets by default, or easily integrate your own local or private datasets
- **Advanced interaction modeling**: Support for protein-protein interaction datasets
  - Coming soon: Protein-small molecule interaction capabilities
- **ProteinGym zero-shot benchmarking**: Run zero-shot DMS scoring against 217+ ProteinGym substitution assays with multiple scoring methods
- **Cloud compute**: Run GPU jobs via the Synthyra API or self-hosted Modal compute
- **Multi-GPU embedding**: Split sequences across GPUs for parallel embedding generation
- **SQL embedding storage**: Compact binary blob storage with async writer for efficient embedding caching

### Support Protify's Development

Help us grow by sharing online, starring our repository, or contributing through our [bounty program](https://gleghornlab.notion.site/1de62a314a2e808bb6fdc1e714725900?v=1de62a314a2e80389ed7000c97c1a709&pvs=4).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Installation
From pip

`pip install Protify`

To get started locally
```bash
git clone https://github.com/Gleghorn-Lab/Protify.git
cd Protify
git submodule update --init --remote --recursive
python -m pip install -r requirements.txt
cd src/protify
```

With a Python VM (linux)
```bash
git clone https://github.com/Gleghorn-Lab/Protify.git
cd Protify
git submodule update --init --remote --recursive
chmod +x setup_protify.sh
./setup_protify.sh
source ~/protify_venv/bin/activate
cd src/protify
```

With Docker

Run all commands from the **repository root** on your host (no need to `cd src/protify`). Build the image once, then run the CLI with the project mounted at `/workspace` and the container working directory set to `/workspace/src/protify` so that `python -m main` runs correctly.

**Linux / macOS:**
```bash
git clone https://github.com/Gleghorn-Lab/Protify.git
cd Protify
git submodule update --init --remote --recursive
docker build -t protify .
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify python -m main --data_names EC --model_names ESM2 --probe_type transformer --max_length 128 --probe_batch_size 4
```

**Windows (PowerShell or cmd):**
```bash
git clone https://github.com/Gleghorn-Lab/Protify.git
cd Protify
git submodule update --init --remote --recursive
docker build -t protify .
docker run --rm -it --gpus all -v "%CD%":/workspace -w /workspace/src/protify protify py -m main --data_names EC --model_names ESM2 --probe_type transformer --max_length 128 --probe_batch_size 4
```

Omit the `--data_names` and other CLI args to see help, or add any options you need. Output paths like `--log_dir` and `--results_dir` are relative to `/workspace/src/protify`; use e.g. `--log_dir /workspace/logs` to write at project root. Note: you may need `sudo` before the docker commands on Linux.

### Optional: xformers for AMPLIFY

If you plan to use the AMPLIFY model with the `--use_xformers` flag for memory-efficient attention, you'll need to install xformers separately:

```bash
pip install xformers
```

**Note for aarch64 systems**: The standard pip installation is not supported on aarch64 architectures. If you encounter installation errors, use this alternative method:

```bash
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers --no-build-isolation
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

<details>
  <summary>Toggle </summary><br />
  
  To launch the gui, run
  
  ```bash
  python -m gui
  ```
  
  It's recommended to use the user interface alongside an open terminal, as helpful messages and progressbars will show in the terminal while you press the GUI buttons.
 
  ### Run jobs in the cloud via Synthyra API (recommended)

  The easiest way to run Protify on GPUs is through the Synthyra API. No local GPU or cloud provider setup required.

  1. Get your API key at [synthyra.com](https://synthyra.com)
  2. In the GUI **Info** tab, enter your Synthyra API key
  3. Configure your data, models, probe, and trainer settings
  4. Click **Submit Remote Run** to launch your job on cloud GPUs
  5. Use **Poll Status** (or auto polling) to monitor job progress
  6. Click **Fetch Logs/Results/Plots** to download artifacts locally

  From the CLI:
  ```bash
  python -m main --synthyra_api_key YOUR_KEY --model_names ESM2-8 --data_names EC
  ```

  Notes:
  - Pass your Hugging Face and W&B tokens in the GUI Info tab so remote runs can authenticate for gated model access, experiment tracking, and Hub uploads.

  ### Parallel multi-seed MLP probes

  When embeddings are already cached, pooled sequence-level MLP probes can train many seeds together instead of launching one Hugging Face Trainer run per seed:

  ```bash
  python -m main --model_names ESM2-8 --data_names DeepLoc-2 --num_runs 8 --parallel_probe_runs --parallel_probe_batch_mode run_specific --parallel_probe_index_strategy permutation --parallel_probe_max_group_size 8
  ```

  Protify infers `num_labels` from the dataset during normal `python -m main` runs. Do not pass `--num_labels` to `main`; that flag only exists on the synthetic benchmark and static planning helper scripts.

  Before a larger workstation sweep, generate a no-training launch plan:

  ```bash
  python -m scripts.plan_parallel_probes --model_names ESM2-8 --data_names DeepLoc-2 --input_size 640 --num_runs 8 --parallel_batch_mode run_specific --parallel_index_strategy permutation --parallel_max_group_size 8 --probe_batch_size 512 --train_dataset_size 5473
  ```

  See [Probes and training](docs/probes_and_training.md#num_runs-aggregation) for eligibility, validation, telemetry, and comparison commands.

  ### Run jobs on Modal (advanced / self-hosted)

  For users who want to manage their own Modal compute directly, Protify also supports a dedicated **Modal** tab in the Tk GUI.

  1. Install Modal locally (same environment as Protify):

  ```bash
  python -m pip install modal
  ```

  2. Provide Modal credentials in the GUI Info tab:
     - Preferred: `modal_token_id` + `modal_token_secret`
     - Backward-compatible: `modal_api_key` in `token_id:token_secret` format

  3. In the **Modal** tab:
     - Set `Modal App Name` and backend path (`src/protify/modal_backend.py`)
     - Select GPU type and timeout
     - Click **Deploy Modal Backend**

  4. Click **Submit Remote Run** to launch the current GUI configuration on Modal.

  5. Use **Poll Status** (or auto polling) to monitor:
     - job status (`PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `TERMINATED`, `TIMEOUT`)
     - current phase and heartbeat age
     - live log tail

  6. Click **Fetch Logs/Results/Plots** to download artifacts to your local `modal_artifacts` directory (configurable in the Modal tab).

  Notes:
  - Pass your Hugging Face and W&B tokens in the GUI Info tab so remote runs can authenticate for gated model access, experiment tracking, and Hub uploads.
  - The Modal backend keeps heartbeat/status updates in persistent storage so the local GUI can detect stalled or failed jobs.
  
  ### An example workflow
  
  Here, we will compare various protein models against a random vector baseline (negative control) and random transformer (homology based control).
  
  1.) Start the session
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/1.PNG">
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/2.PNG" width="500">
  
  2.) Select the models you would like to benchmark
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/3.PNG" width="500">
  
  3.) Select the datasets you are interested in. Here we chose Enzyme Commission numbers (multi-label classification), metal-ion binding (binary classification), solubility (binary classification), catalytic rate (kcat, regression), and protein localization (DeepLoc-2, binary classification).
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/4.PNG" width="500">
  
  4.) Embed the proteins in the selected datasets. If your machine does not have a GPU, you can download precomputed embeddings for many common sequences.
    Note: If you download embeddings, it will be faster to use the scikit model tab than the probe tab
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/5.PNG" width="500">
  
  5.) Select which probe and configuration you would like. Here, we will use a simple MLP probe, a small neural network. It is the **fastest** (by a large margin) but worst performing option (by a small margin usually).
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/6.PNG" width="500">
  
  6.) Select your settings for training. Like most of the tabs, the defaults are pretty good. If you need information about what setting does what, the `?` button provides a helpful note. The documentations has more extensive information
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/7.PNG" width="500">
  
  This will train your models!
  
  7.) After training, you can render helpful visualizations by passing the log ID from before. If you forget it, you can look for the file generated in the `logs` folder.
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/8.PNG" width="500">
  
  Here's a sample of the many plots produced. You can find them all inside `plots/your_log_id/*`
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/9.png" width="500">
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/10.png" width="500">
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/11.png" width="500">
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/13.png" width="500">

  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/12.png" width="500">
  
  8.) Need to replicate your findings for a report or paper? Just input the generated log into the replay tab
  
  <img src="https://github.com/Gleghorn-Lab/Protify/blob/main/assets/example_workflow/14.PNG" width="500">

  To run the same session from the command line instead, you would simply execute
  ```
  python -m main --model_names ESM2-8 ESM2-35 ESMC-300 ProtBert ANKH-Base Random Random-Transformer --data_names EC DeepLoc-2 enzyme-kcat MB solubility --patience 3
  ```

  With Docker (from the repo root on your host):
  ```
  docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify python -m main --model_names ESM2-8 ESM2-35 ESMC-300 ProtBert ANKH-Base Random Random-Transformer --data_names EC DeepLoc-2 enzyme-kcat MB solubility --patience 3
  ```
  On Windows use `-v "%CD%":/workspace` and `py -m main` instead of `python -m main`.

  ### Using custom or non-preset models

  Instead of preset `--model_names`, you can use `--model_paths` and `--model_types` to load any HuggingFace or local model directly. The two lists are positionally paired and mutually exclusive with `--model_names`.

  Load a custom model via `AutoModel`:
  ```
  python -m main --model_paths answerdotai/ModernBERT-base --model_types custom --data_names DeepLoc-2
  ```

  Load a fine-tuned version of a known architecture:
  ```
  python -m main --model_paths my-org/finetuned-esm2 --model_types esm2 --data_names DeepLoc-2
  ```

  Mix multiple models of different types:
  ```
  python -m main --model_paths Synthyra/ESM2-8M answerdotai/ModernBERT-base --model_types esm2 custom --data_names DeepLoc-2
  ```

  Valid `--model_types` values: `esm2`, `dsm`, `esmc`, `protbert`, `prott5`, `ankh`, `glm`, `dplm`, `dplm2`, `protclm`, `onehot`, `amplify`, `e1`, `calm`, `custom`, `random`.

  Or, set up a yaml file with your desired settings (so you don't have to type out everything in the CLI)
  ```
  python -m main --yaml_path yamls/your_custom_yaml_path.yaml
  ```
  Replaying from the CLI is just as simple
  ```
  python -m main --replay_path logs/your_log_id.txt
  ```

  ### CLI auto-run on Modal (advanced)

  For self-hosted Modal compute, pass Modal credentials as CLI arguments to run remotely on Modal instead of local execution. Most users should use `--synthyra_api_key` instead (see above).

  ```bash
  python -m main \
    --modal_token_id YOUR_MODAL_TOKEN_ID \
    --modal_token_secret YOUR_MODAL_TOKEN_SECRET \
    --rebuild_modal \
    --delete_modal_embeddings \
    --model_names ESM2-8 \
    --data_names EC
  ```

  Behavior:
  - Triggers only when Modal credentials are explicitly provided in CLI args (`--modal_token_id` / `--modal_token_secret` or `--modal_api_key`).
  - `--rebuild_modal` forces a fresh deploy of `src/protify/modal_backend.py` before execution.
  - `--delete_modal_embeddings` clears `/data/embeddings` on the Modal volume before job submission (or can be run as a maintenance-only command when no dataset/proteingym run is requested).
  - Automatically deploys `src/protify/modal_backend.py` if submit/delete fails before app/function lookup succeeds.
  - Submits the remote job, streams incremental remote logs to your local terminal, and uses the Modal `FunctionCall` result as the terminal source of truth.
  - Requires the currently deployed backend to provide `get_job_log_delta` (no legacy log-tail fallback).
  - On completion, fetches artifacts and exits with:
    - `0` on success
    - non-zero on failed/terminated/timeout jobs
  - Saves fetched artifacts to `modal_artifacts/<job_id>` by default.

</details>

## Hyperparameter Optimization

Protify uses Weights & Biases (W&B) for automated hyperparameter optimization across all training modes (neural network probes, full fine-tuning, and hybrid probes). This feature allows you to systematically search for optimal hyperparameters to maximize model performance.

### Overview

- **Multiple search methods**: Choose from Bayesian, grid, or random search
- **Flexible configuration**: Customize which hyperparameters to optimize via the YAML file
- **Automatic best model selection**: After sweep completion, the best configuration is automatically applied and used for final training
- **Comprehensive logging**: All trials are logged to W&B and saved locally as CSV files for later analysis
- **Works with all training modes**: Neural network probes, full fine-tuning, and hybrid probes

### Quick Start

To use W&B hyperparameter optimization, simply add the `--use_wandb_hyperopt` flag to any run:

```bash
python -m main \
  --use_wandb_hyperopt \
  --model_names ESM2-8 ESM2-35 \
  --data_names DeepLoc-2 MB
```

That's it! The sweep will use default settings (Bayesian search, 10 trials, minimize validation loss). To customize these defaults, see the Configuration section below.

### Configuration
 
Hyperparameter sweeps are configured via the `sweep.yaml` file (located in `src/protify/yamls/sweep.yaml`), as well as the CLI. Here's the default configuration:

```yaml
early_terminate: {type: hyperband, min_iter: 10}  # early stopping (stops underperforming trials)

parameters:
  # Common parameters for all training modes
  lr:
    distribution: log_uniform_values
    min: 0.000001
    max: 0.01
  weight_decay:
    distribution: log_uniform_values
    min: 0.000001
    max: 0.1
  
  # Full finetuning specific parameters
  base_batch_size:
    values: [4, 8, 16]
  base_grad_accum:
    values: [4, 8, 16]
  
  # Probe specific parameters
  probe_batch_size:
    values: [16, 32, 64, 128]
  dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
  hidden_size:
    values: [512, 1024, 2048, 4096, 8192]
  n_layers:
    values: [1, 2, 3]
  probe_pooling_types:
    values: ["cls", "mean"]
  
  # Transformer probe specific parameters
  transformer_dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
  classifier_dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
  pre_ln:
    values: [True, False]
  classifier_dim:
    values: [4096, 8192]
  head_size:  # attention head dimension; n_heads = hidden_size // head_size
    values: [64, 128, 256]
  # `n_heads` is deprecated as a sweep key. Old configs that use it are auto-migrated
  # to `head_size = hidden_size // n_heads` with a DeprecationWarning.
  
  # LoRA parameters
  lora_r:
    values: [8, 16, 32]
  lora_alpha:
    values: [16, 32, 64]
  lora_dropout:
    values: [0.0, 0.01, 0.05]
```    
### CLI Arguments

The following arguments control hyperparameter optimization:

- `--use_wandb_hyperopt`: Enable W&B hyperparameter optimization (flag)
- `--wandb_api_key`: Your Weights & Biases API key (alternatively set via environment variable, or the terminal when prompted)
- `--wandb_project`: W&B project name (default: "Protify")
- `--wandb_entity`: W&B team/user entity (optional)
- `--sweep_config_path`: Path to sweep configuration YAML (default: "yamls/sweep.yaml")
- `--sweep_count`: Number of trials to run (default: 10)
- `--sweep_method`: Search method - "bayes", "grid", or "random" (default: "bayes")
- `--sweep_metric_cls`: Classification metric to optimize during sweep (default: "eval_loss")
- `--sweep_metric_reg`: Regression metric to optimize during sweep (default: "eval_loss")
- `--sweep_goal`: Optimization goal - "maximize" or "minimize" (default: "minimize")

- **NOTE**: The default optimization strategy is to minimize the eval_loss. If you would like to choose a different metric via the `--sweep_metric_cls` and `--sweep_metric_reg` arguments, make sure you ALSO change `--sweep_goal` to **maximize**.

Common metrics you can optimize for:

- **Classification**: `eval_accuracy`, `eval_mcc`, `eval_f1`, `eval_precision`, `eval_recall`
- **Regression**: `eval_r_squared`, `eval_pearson_rho`, `eval_spearman_rho`

### Workflow

1. **Initialization**: Protify reads your sweep configuration and initializes a W&B sweep
2. **Trials**: For each trial, W&B suggests hyperparameters based on the search method
3. **Training**: A model is trained with the suggested hyperparameters
4. **Evaluation**: Validation metrics are computed and reported to W&B
5. **Selection**: After all trials, the best configuration is automatically selected
6. **Final training**: The model is retrained with the best hyperparameters and evaluated on the test set
7. **Results**: All results and plots are reported based on final test set performance. Top 5 W&B trials are saved to a CSV file (`logs/YOUR_ID_sweep_DATASET_MODEL.csv`) 
### ProteinGym Benchmarking

Protify includes a zero-shot pipeline for the ProteinGym DMS benchmark with a standardized performance summary.

- Run zero-shot scoring on ProteinGym substitutions
  ```console
  python -m main --proteingym \
    --model_names ESM2-8 ESM2-35 ProtBert \
    --dms_ids all \
    --scoring_method masked_marginal \
    --scoring_window optimal \
  ```
  - Outputs per-assay CSVs at `results/proteingym/*__zs_masked_marginal.csv`
  - After scoring, a standardized performance summary is written to `results/proteingym/benchmark_performance/`
    - This summary exactly matches the format expected by the ProteinGym repository for adding scores for a new model (ready to use in a PR)

  Available options
  - dms_ids. By default, all 217 substitution assays are used. You can specify DMS_ids by name to only use a subset.
  - scoring_method
    - masked_marginal (default): Mask mutated positions in the wild-type window; score Δlog p(mutant) − log p(wildtype) at those positions
    - wildtype_marginal: Unmasked wild-type context; score Δlog p(mutant) − log p(wildtype) at mutated positions
    - mutant_marginal: Unmasked mutant context; score Δlog p(mutant) − log p(wildtype) at mutated positions
    - pll: Pseudo log-likelihood obtained by masking each position and summing true-token log-likelihoods (indels are scored with a length-normalized PLL across windows)
    - global_log_prob: Unmasked log-probability of the entire mutated sequence/window
  - scoring_window
    - optimal (default): Single window (≤ model context) centered around the mutation barycenter
    - sliding: Non-overlapping contiguous windows across the full sequence (default for indels)

- Compare performance & time for each scoring method for one or more models
  ```console
  python -m main --proteingym --compare_scoring_methods \
    --model_names ESM2-650 \
    --dms_ids AACC1_PSEAI_Dandage_2018 A4_HUMAN_Seuma_2022 \
    --results_dir results
  ```
  - Saves a summary CSV to `results/scoring_methods_comparison.csv`


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Testing

Tests live under `src/protify/testing_suite/`. Run them with:

```bash
py -m pytest src/protify/testing_suite/ -v
```

The suite covers metrics, pooling, probe construction, model components, loss functions, blob serialization, data utilities, seed reproducibility, scikit-learn integration, packaged probe export, and attention backends.

For Docker-based testing (recommended for GPU-dependent tests):

```bash
docker build -t protify .
docker run --rm --gpus all -v "${PWD}":/workspace -w /workspace protify python -m pytest src/protify/testing_suite/ -v
```

To run only CPU tests (no GPU required):

```bash
docker run --rm -v "${PWD}":/workspace -w /workspace protify python -m pytest src/protify/testing_suite/ -v -m "not gpu and not slow"
```

For vectorized multi-seed MLP probe changes, run the focused Docker suite from `src/protify`:

```bash
docker run --rm --gpus all -v "${PWD}":/workspace -e PYTHONPATH=/workspace -w /workspace/src/protify protify python -m pytest testing_suite/test_parallel_probe_benchmark.py testing_suite/test_parallel_probe_batches.py testing_suite/test_parallel_probe_compare.py testing_suite/test_parallel_probe_hardware_monitor.py testing_suite/test_parallel_probe_launch_manifest_runner.py testing_suite/test_parallel_probe_plan.py testing_suite/test_parallel_linear_probe.py testing_suite/test_parallel_probe_logger.py testing_suite/test_parallel_probe_preflight.py testing_suite/test_trainer_arguments.py -v
```

See [docs/testing.md](docs/testing.md) for full details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

We work with a [bounty system](https://gleghornlab.notion.site/1de62a314a2e808bb6fdc1e714725900?v=1de62a314a2e80389ed7000c97c1a709&pvs=4). You can find bounties on this page. Contributing bounties will get you listed on the Protify consortium and potentially coauthorship on published papers involving the framework.

Simply open a pull request with the bounty ID in the title to claim one. For additional features not on the bounty list simply use a descriptive title.

For bugs and general suggestions please use [GitHub issues](https://github.com/Gleghorn-Lab/Protify/issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Transformers][Transformers-badge]][Transformers-url]
* [![Datasets][Datasets-badge]][Datasets-url]
* [![PEFT][PEFT-badge]][PEFT-url]
* [![scikit-learn][Scikit-learn-badge]][Scikit-learn-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![SciPy][SciPy-badge]][SciPy-url]
* [![Einops][Einops-badge]][Einops-url]
* [![PAUC][PAUC-badge]][PAUC-url]
* [![LazyPredict][LazyPredict-badge]][LazyPredict-url]
* [![Weights & Biases][WandB-badge]][WandB-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the Apache 2.0 license. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

info@synthyra.com

[Synthyra](https://synthyra.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Cite

If you use this package, please cite the following papers. (Coming soon)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Gleghorn-Lab/Protify.svg?style=for-the-badge
[contributors-url]: https://github.com/Gleghorn-Lab/Protify/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Gleghorn-Lab/Protify.svg?style=for-the-badge
[forks-url]: https://github.com/Gleghorn-Lab/Protify/network/members
[stars-shield]: https://img.shields.io/github/stars/Gleghorn-Lab/Protify.svg?style=for-the-badge
[stars-url]: https://github.com/Gleghorn-Lab/Protify/stargazers
[issues-shield]: https://img.shields.io/github/issues/Gleghorn-Lab/Protify.svg?style=for-the-badge
[issues-url]: https://github.com/Gleghorn-Lab/Protify/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/synthyra
[product-screenshot]: assets/screenshot.png

[Transformers-badge]: https://img.shields.io/badge/Hugging%20Face-Transformers-FF6C44?style=for-the-badge&logo=Huggingface&logoColor=white  
[Transformers-url]: https://github.com/huggingface/transformers

[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white  
[PyTorch-url]: https://github.com/pytorch/pytorch

[Datasets-badge]: https://img.shields.io/badge/Hugging%20Face-Datasets-0078D4?style=for-the-badge&logo=Huggingface&logoColor=white  
[Datasets-url]: https://github.com/huggingface/datasets

[Scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white  
[Scikit-learn-url]: https://github.com/scikit-learn/scikit-learn

[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white  
[NumPy-url]: https://github.com/numpy/numpy

[SciPy-badge]: https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white  
[SciPy-url]: https://github.com/scipy/scipy

[PAUC-badge]: https://img.shields.io/badge/PAUC-Package-4B8BBE?style=for-the-badge&logo=python&logoColor=white  
[PAUC-url]: https://pypi.org/project/pauc

[LazyPredict-badge]: https://img.shields.io/badge/LazyPredict-Modeling-4B8BBE?style=for-the-badge&logo=python&logoColor=white  
[LazyPredict-url]: https://github.com/shankarpandala/lazypredict

[PEFT-badge]: https://img.shields.io/badge/PEFT-HuggingFace-713196?style=for-the-badge&logo=Huggingface&logoColor=white  
[PEFT-url]: https://github.com/huggingface/peft

[Einops-badge]: https://img.shields.io/badge/Einops-Transform-4B8BBE?style=for-the-badge&logo=python&logoColor=white  
[Einops-url]: https://github.com/arogozhnikov/einops

[WandB-badge]: https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white  
[WandB-url]: https://wandb.ai
