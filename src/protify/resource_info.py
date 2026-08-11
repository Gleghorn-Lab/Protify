#!/usr/bin/env python
"""
Utility script to list Protify supported models and datasets.
"""
import argparse
import sys


model_descriptions = {
    'ESM2-8': {
        'description': 'Small protein language model (8M parameters) from Meta AI that learns evolutionary information from millions of protein sequences.',
        'size': '8M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-35': {
        'description': 'Medium-sized protein language model (35M parameters) trained on evolutionary data.',
        'size': '35M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-150': {
        'description': 'Large protein language model (150M parameters) with improved protein structure prediction capabilities.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-650': {
        'description': 'Very large protein language model (650M parameters) offering state-of-the-art performance on many protein prediction tasks.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-3B': {
        'description': 'Largest ESM2 protein language model (3B parameters) with exceptional capability for protein structure and function prediction.',
        'size': '3B parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'Random': {
        'description': 'Baseline model with randomly initialized weights, serving as a negative control.',
        'size': 'Varies',
        'type': 'Baseline control',
        'citation': 'N/A'
    },
    'Random-Transformer': {
        'description': 'Randomly initialized transformer model serving as a homology-based control.',
        'size': 'Varies',
        'type': 'Baseline control',
        'citation': 'N/A'
    },
    'ESMC-300': {
        'description': 'Protein language model optimized for classification tasks with 300M parameters.',
        'size': '300M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'ESMC-600': {
        'description': 'Larger protein language model for classification with 600M parameters.',
        'size': '600M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'ESMC-6B': {
        'description': 'Large ESM++ protein language model corresponding to ESMC 6B.',
        'size': '6.352B parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'ESMC-300-SAE': {
        'description': 'ESMC 300M read through a Biohub sparse autoencoder, giving sparse codebook features per residue. Defaults to layer 23, k 64, codebook 8192.',
        'size': '300M parameters plus the sparse autoencoder codebook',
        'type': 'Protein language model sparse autoencoder',
        'citation': 'biohub/ESMC-SAE-Overview'
    },
    'ESMC-600-SAE': {
        'description': 'ESMC 600M read through a Biohub sparse autoencoder. Defaults to layer 27, k 64, codebook 8192.',
        'size': '600M parameters plus the sparse autoencoder codebook',
        'type': 'Protein language model sparse autoencoder',
        'citation': 'biohub/ESMC-SAE-Overview'
    },
    'ESMC-6B-SAE': {
        'description': 'ESMC 6B read through a Biohub sparse autoencoder. Defaults to layer 60, k 64, codebook 8192.',
        'size': '6.352B parameters plus the sparse autoencoder codebook',
        'type': 'Protein language model sparse autoencoder',
        'citation': 'biohub/ESMC-SAE-Overview'
    },
    'ProtBert': {
        'description': 'BERT-based protein language model trained on protein sequences from UniRef.',
        'size': '420M parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ProtBert-BFD': {
        'description': 'BERT-based protein language model trained on BFD database with improved performance.',
        'size': '420M parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ProtT5': {
        'description': 'T5-based protein language model capable of both encoding and generation tasks.',
        'size': '3B parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ProtT5-XL-UniRef50-full-prec': {
        'description': 'Extra large T5-based protein language model trained on UniRef50 with full precision.',
        'size': '11B parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ANKH-Base': {
        'description': 'Base version of the ANKH protein language model focused on protein structure understanding.',
        'size': '400M parameters',
        'type': 'Protein language model',
        'citation': 'Choromanski et al. (2022). ANKH: Optimized Protein Language Model Unlocks General-Purpose Modelling.'
    },
    'ANKH-Large': {
        'description': 'Large version of the ANKH protein language model with improved structural predictions.',
        'size': '1.2B parameters',
        'type': 'Protein language model',
        'citation': 'Choromanski et al. (2022). ANKH: Optimized Protein Language Model Unlocks General-Purpose Modelling.'
    },
    'ANKH2-Large': {
        'description': 'Improved second generation ANKH protein language model.',
        'size': '1.2B parameters',
        'type': 'Protein language model',
        'citation': 'Choromanski et al. (2022). ANKH: Optimized Protein Language Model Unlocks General-Purpose Modelling.'
    },
    'GLM2-150': {
        'description': 'Medium-sized general language model adapted for protein sequences.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'GLM2-650': {
        'description': 'Large general language model adapted for protein sequences.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'GLM2-GAIA': {
        'description': 'Specialized GLM protein language model with GAIA architecture improvements.',
        'size': '1B+ parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DPLM-150': {
        'description': 'Deep protein language model with 150M parameters focused on protein structure.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DPLM-650': {
        'description': 'Larger deep protein language model with 650M parameters.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DPLM-3B': {
        'description': 'Largest deep protein language model in the DPLM family with 3B parameters.',
        'size': '3B parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DSM-150': {
        'description': 'Deep language model for proteins with 150M parameters.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DSM-650': {
        'description': 'Deep language model for proteins with 650M parameters.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'CARBON-500M': {
        'description': 'Decoder-only autoregressive DNA language model (6-mer tokenization) for sequence-level genomic tasks.',
        'size': '500M parameters',
        'type': 'DNA language model',
        'citation': 'HuggingFaceBio/Carbon-500M'
    },
    'CARBON-3B': {
        'description': 'Decoder-only autoregressive DNA language model (6-mer tokenization) for sequence-level genomic tasks.',
        'size': '3B parameters',
        'type': 'DNA language model',
        'citation': 'HuggingFaceBio/Carbon-3B'
    },
    'CARBON-8B': {
        'description': 'Decoder-only autoregressive DNA language model (6-mer tokenization) for sequence-level genomic tasks.',
        'size': '8B parameters',
        'type': 'DNA language model',
        'citation': 'HuggingFaceBio/Carbon-8B'
    }
}


dataset_descriptions = {
    'EC': {
        'description': 'Enzyme Commission numbers dataset for predicting enzyme function classification.',
        'type': 'Multi-label classification',
        'task': 'Protein function prediction',
        'citation': 'Gleghorn Lab'
    },
    'GO-CC': {
        'description': 'Gene Ontology Cellular Component dataset for predicting protein localization in cells.',
        'type': 'Multi-label classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'GO-BP': {
        'description': 'Gene Ontology Biological Process dataset for predicting protein involvement in biological processes.',
        'type': 'Multi-label classification',
        'task': 'Protein function prediction',
        'citation': 'Gleghorn Lab'
    },
    'GO-MF': {
        'description': 'Gene Ontology Molecular Function dataset for predicting protein molecular functions.',
        'type': 'Multi-label classification',
        'task': 'Protein function prediction',
        'citation': 'Gleghorn Lab'
    },
    'MB': {
        'description': 'Metal ion binding dataset for predicting protein-metal interactions.',
        'type': 'Classification',
        'task': 'Protein-metal binding prediction',
        'citation': 'Gleghorn Lab'
    },
    'DeepLoc-2': {
        'description': 'Binary classification dataset for predicting protein localization in 2 categories.',
        'type': 'Binary classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'DeepLoc-10': {
        'description': 'Multi-class classification dataset for predicting protein localization in 10 categories.',
        'type': 'Multi-class classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'enzyme-kcat': {
        'description': 'Dataset for predicting enzyme catalytic rate constants (kcat).',
        'type': 'Regression',
        'task': 'Enzyme kinetics prediction',
        'citation': 'Gleghorn Lab'
    },
    'solubility': {
        'description': 'Dataset for predicting protein solubility properties.',
        'type': 'Binary classification',
        'task': 'Protein solubility prediction',
        'citation': 'Gleghorn Lab'
    },
    'localization': {
        'description': 'Dataset for predicting subcellular localization of proteins.',
        'type': 'Multi-class classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'temperature-stability': {
        'description': 'Dataset for predicting protein stability at different temperatures.',
        'type': 'Binary classification',
        'task': 'Protein stability prediction',
        'citation': 'Gleghorn Lab'
    },
    'peptide-HLA-MHC-affinity': {
        'description': 'Dataset for predicting peptide binding affinity to HLA/MHC complexes.',
        'type': 'Protein-protein interaction',
        'task': 'Binding affinity prediction',
        'citation': 'Gleghorn Lab'
    },
    'optimal-temperature': {
        'description': 'Dataset for predicting the optimal temperature for protein function.',
        'type': 'Regression',
        'task': 'Protein property prediction',
        'citation': 'Gleghorn Lab'
    },
    'optimal-ph': {
        'description': 'Dataset for predicting the optimal pH for protein function.',
        'type': 'Regression',
        'task': 'Protein property prediction',
        'citation': 'Gleghorn Lab'
    },
    'optimal-ph-rigor': {
        'description': 'Rigorously curated variant of the optimal pH dataset for predicting the optimal pH for protein function.',
        'type': 'Regression',
        'task': 'Protein property prediction',
        'citation': 'Gleghorn Lab'
    },
    'material-production': {
        'description': 'Dataset for predicting protein suitability for material production.',
        'type': 'Classification',
        'task': 'Protein application prediction',
        'citation': 'Gleghorn Lab'
    },
    'fitness-prediction': {
        'description': 'Dataset for predicting protein fitness in various environments.',
        'type': 'Classification',
        'task': 'Protein fitness prediction',
        'citation': 'Gleghorn Lab'
    },
    'number-of-folds': {
        'description': 'Dataset for predicting the number of structural folds in proteins.',
        'type': 'Classification',
        'task': 'Protein structure prediction',
        'citation': 'Gleghorn Lab'
    },
    'cloning-clf': {
        'description': 'Dataset for predicting protein suitability for cloning operations.',
        'type': 'Classification',
        'task': 'Protein engineering prediction',
        'citation': 'Gleghorn Lab'
    },
    'stability-prediction': {
        'description': 'Dataset for predicting overall protein stability.',
        'type': 'Classification',
        'task': 'Protein stability prediction',
        'citation': 'Gleghorn Lab'
    },
    'human-ppi': {
        'description': 'Dataset for predicting human protein-protein interactions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Gleghorn Lab'
    },
    'SecondaryStructure-3': {
        'description': 'Dataset for predicting protein secondary structure in 3 classes.',
        'type': 'Token-wise classification',
        'task': 'Protein structure prediction',
        'citation': 'Gleghorn Lab'
    },
    'SecondaryStructure-8': {
        'description': 'Dataset for predicting protein secondary structure in 8 classes.',
        'type': 'Token-wise classification',
        'task': 'Protein structure prediction',
        'citation': 'Gleghorn Lab'
    },
    'fluorescence-prediction': {
        'description': 'Dataset for predicting protein fluorescence properties.',
        'type': 'Token-wise regression',
        'task': 'Protein property prediction',
        'citation': 'Gleghorn Lab'
    },
    'plastic': {
        'description': 'Dataset for predicting protein capability for plastic degradation.',
        'type': 'Classification',
        'task': 'Enzyme function prediction',
        'citation': 'Gleghorn Lab'
    },
    'gold-ppi': {
        'description': 'Gold standard dataset for protein-protein interaction prediction.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Synthyra/bernett_gold_ppi'
    },
    'human-ppi-pinui': {
        'description': 'Human protein-protein interaction dataset from PiNUI.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Gleghorn Lab'
    },
    'yeast-ppi-pinui': {
        'description': 'Yeast protein-protein interaction dataset from PiNUI.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Gleghorn Lab'
    },
    'shs27-ppi': {
        'description': 'SHS27k dataset containing 27,000 protein-protein interactions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Synthyra/SHS27k'
    },
    'shs148-ppi': {
        'description': 'SHS148k dataset containing 148,000 protein-protein interactions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Synthyra/SHS148k'
    },
    'PPA-ppi': {
        'description': 'Protein-Protein Affinity dataset for quantitative binding predictions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI affinity prediction',
        'citation': 'Synthyra/ProteinProteinAffinity'
    },
}


def list_models(show_standard_only: bool = False) -> None:
    """List available models with descriptions if available"""
    try:
        from .base_models.get_base_models import currently_supported_models, standard_models
        from .base_models.model_descriptions import model_descriptions
        
        if show_standard_only:
            models_to_show = standard_models
            print("\n=== Standard Models ===\n")
        else:
            models_to_show = currently_supported_models
            print("\n=== All Supported Models ===\n")
        
        # Calculate maximum widths for formatting
        max_name_len = max(len(name) for name in models_to_show)
        max_type_len = max(len(model_descriptions.get(name, {}).get('type', 'Unknown')) for name in models_to_show if name in model_descriptions)
        max_size_len = max(len(model_descriptions.get(name, {}).get('size', 'Unknown')) for name in models_to_show if name in model_descriptions)
        
        # Print header
        print(f"{'Model':<{max_name_len+2}}{'Type':<{max_type_len+2}}{'Size':<{max_size_len+2}}Description")
        print("-" * (max_name_len + max_type_len + max_size_len + 50))
        
        # Print model information
        for model_name in models_to_show:
            if model_name in model_descriptions:
                model_info = model_descriptions[model_name]
                print(f"{model_name:<{max_name_len+2}}{model_info.get('type', 'Unknown'):<{max_type_len+2}}{model_info.get('size', 'Unknown'):<{max_size_len+2}}{model_info.get('description', 'No description available')}")
            else:
                print(f"{model_name:<{max_name_len+2}}{'Unknown':<{max_type_len+2}}{'Unknown':<{max_size_len+2}}No description available")
    
    except ImportError as e:
        print(f"Error loading model information: {e}")
        print("\n=== Models ===\n")
        try:
            from .base_models.get_base_models import currently_supported_models, standard_models
            
            if show_standard_only:
                for model_name in standard_models:
                    print(f"- {model_name}")
            else:
                for model_name in currently_supported_models:
                    print(f"- {model_name}")
        except ImportError:
            print("Could not load model lists. Please check your installation.")


def list_datasets(show_standard_only: bool = False) -> None:
    """List available datasets with descriptions if available"""
    try:
        from .data.supported_datasets import supported_datasets, standard_data_benchmark
        from .data.dataset_descriptions import dataset_descriptions
        
        if show_standard_only:
            datasets_to_show = {name: supported_datasets[name] for name in standard_data_benchmark if name in supported_datasets}
            print("\n=== Standard Benchmark Datasets ===\n")
        else:
            datasets_to_show = supported_datasets
            print("\n=== All Supported Datasets ===\n")
        
        # Calculate maximum widths for formatting
        max_name_len = max(len(name) for name in datasets_to_show)
        max_type_len = max(len(dataset_descriptions.get(name, {}).get('type', 'Unknown')) for name in datasets_to_show if name in dataset_descriptions)
        max_task_len = max(len(dataset_descriptions.get(name, {}).get('task', 'Unknown')) for name in datasets_to_show if name in dataset_descriptions)
        
        # Print header
        print(f"{'Dataset':<{max_name_len+2}}{'Type':<{max_type_len+2}}{'Task':<{max_task_len+2}}Description")
        print("-" * (max_name_len + max_type_len + max_task_len + 50))
        
        # Print dataset information
        for dataset_name in datasets_to_show:
            if dataset_name in dataset_descriptions:
                dataset_info = dataset_descriptions[dataset_name]
                print(f"{dataset_name:<{max_name_len+2}}{dataset_info.get('type', 'Unknown'):<{max_type_len+2}}{dataset_info.get('task', 'Unknown'):<{max_task_len+2}}{dataset_info.get('description', 'No description available')}")
            else:
                print(f"{dataset_name:<{max_name_len+2}}{'Unknown':<{max_type_len+2}}{'Unknown':<{max_task_len+2}}No description available")
    
    except ImportError as e:
        print(f"Error loading dataset information: {e}")
        print("\n=== Datasets ===\n")
        try:
            from .data.supported_datasets import supported_datasets, standard_data_benchmark
            
            if show_standard_only:
                for dataset_name in standard_data_benchmark:
                    if dataset_name in supported_datasets:
                        print(f"- {dataset_name}: {supported_datasets[dataset_name]}")
            else:
                for dataset_name, dataset_source in supported_datasets.items():
                    print(f"- {dataset_name}: {dataset_source}")
        except ImportError:
            print("Could not load dataset lists. Please check your installation.")


def main() -> None:
    """Main function to run the script from command line"""
    parser = argparse.ArgumentParser(description='List Protify supported models and datasets')
    parser.add_argument('--models', action='store_true', help='List supported models')
    parser.add_argument('--datasets', action='store_true', help='List supported datasets')
    parser.add_argument('--standard-only', action='store_true', help='Show only standard models/datasets')
    parser.add_argument('--all', action='store_true', help='Show both models and datasets')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1 or args.all:
        list_models(args.standard_only)
        print("\n" + "="*80 + "\n")
        list_datasets(args.standard_only)
        return
    
    if args.models:
        list_models(args.standard_only)
    
    if args.datasets:
        list_datasets(args.standard_only)


if __name__ == "__main__":
    main()