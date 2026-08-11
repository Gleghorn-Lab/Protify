

all_presets_with_paths = {
    # ESM2 models (from esm2.py)
    'ESM2-8': 'Synthyra/ESM2-8M',
    'ESM2-35': 'Synthyra/ESM2-35M',
    'ESM2-150': 'Synthyra/ESM2-150M',
    'ESM2-650': 'Synthyra/ESM2-650M',
    'ESM2-3B': 'Synthyra/ESM2-3B',
    # DSM models (from esm2.py)
    'DSM-150': 'GleghornLab/DSM_150',
    'DSM-650': 'GleghornLab/DSM_650',
    'DSM-PPI': 'Synthyra/DSM_ppi_full',
    # ESMC models (from esmc.py)
    'ESMC-300': 'Synthyra/ESMplusplus_small',
    'ESMC-600': 'Synthyra/ESMplusplus_large',
    'ESMC-6B': 'Synthyra/ESMplusplus_6B',
    # ESMC sparse autoencoders (from esmc_sae.py). These aliases expand to a fully
    # qualified name such as ESMC-300-SAE-l23-k64-c8192; the path here is the ESM++
    # backbone the SAE reads, since the codebook weights come from a separate Biohub
    # repository that resolve_sae_selection picks.
    'ESMC-300-SAE': 'Synthyra/ESMplusplus_small',
    'ESMC-600-SAE': 'Synthyra/ESMplusplus_large',
    'ESMC-6B-SAE': 'Synthyra/ESMplusplus_6B',
    # E1 models (from e1.py)
    'E1-150': 'Synthyra/Profluent-E1-150M',
    'E1-300': 'Synthyra/Profluent-E1-300M',
    'E1-600': 'Synthyra/Profluent-E1-600M',
    # ProtBert models (from protbert.py)
    'ProtBert': 'Rostlab/prot_bert',
    'ProtBert-BFD': 'Rostlab/prot_bert_bfd',
    # ProtT5 models (from prott5.py)
    'ProtT5': 'Rostlab/prot_t5_xl_half_uniref50-enc',
    'ProtT5-XL-UniRef50-full-prec': 'Rostlab/prot_t5_xl_uniref50',
    'ProtT5-XXL-UniRef50': 'Rostlab/prot_t5_xxl_uniref50',
    'ProtT5-XL-BFD': 'Rostlab/prot_t5_xl_bfd',
    'ProtT5-XXL-BFD': 'Rostlab/prot_t5_xxl_bfd',
    # ANKH models (from ankh.py)
    'ANKH-Base': 'Synthyra/ANKH_base',
    'ANKH-Large': 'Synthyra/ANKH_large',
    'ANKH2-Large': 'Synthyra/ANKH2_large',
    # GLM2 models (from glm.py)
    'GLM2-150': 'tattabio/gLM2_150M',
    'GLM2-650': 'tattabio/gLM2_650M',
    'GLM2-GAIA': 'tattabio/gLM2_650M_embed',
    # DPLM models (from dplm.py)
    'DPLM-150': 'Synthyra/DPLM-150M',
    'DPLM-650': 'Synthyra/DPLM-650M',
    'DPLM-3B': 'Synthyra/DPLM-3B',
    # DPLM2 models (from dplm2.py)
    'DPLM2-150': 'Synthyra/DPLM2-150M',
    'DPLM2-650': 'Synthyra/DPLM2-650M',
    'DPLM2-3B': 'Synthyra/DPLM2-3B',
    # AMPLIFY models (from amplify.py)
    'AMPLIFY-120': 'GleghornLab/AMPLIFY_120M',
    'AMPLIFY-350': 'GleghornLab/AMPLIFY_350M',
    # Random models (from random.py)
    'Random': 'random',
    'Random-Transformer': 'facebook/esm2_t12_35M_UR50D',
    'Random-ESM2-8': 'facebook/esm2_t6_8M_UR50D',
    'Random-ESM2-35': 'facebook/esm2_t12_35M_UR50D',
    'Random-ESM2-150': 'facebook/esm2_t30_150M_UR50D',
    'Random-ESM2-650': 'facebook/esm2_t36_650M_UR50D',
    # OneHot models (from one_hot.py) - internal implementations
    'OneHot-Protein': 'OneHot-Protein',
    'OneHot-DNA': 'OneHot-DNA',
    'OneHot-RNA': 'OneHot-RNA',
    'OneHot-Codon': 'OneHot-Codon',
    # CaLM models (from calm.py)
    'CaLM': 'multimolecule/calm',
}


currently_supported_models = [
    'ESM2-8',
    'ESM2-35',
    'ESM2-150',
    'ESM2-650',
    'ESM2-3B',
    'Random',
    'Random-Transformer',
    'Random-ESM2-8',
    'Random-ESM2-35', # same as Random-Transformer
    'Random-ESM2-150',
    'Random-ESM2-650',
    'ESMC-300',
    'ESMC-600',
    'ESMC-6B',
    'ESMC-300-SAE',
    'ESMC-600-SAE',
    'ESMC-6B-SAE',
    'E1-150',
    'E1-300',
    'E1-600',
    'ProtBert',
    'ProtBert-BFD',
    'ProtT5',
    'ProtT5-XL-UniRef50-full-prec',
    'ProtT5-XXL-UniRef50',
    'ProtT5-XL-BFD',
    'ProtT5-XXL-BFD',
    'ANKH-Base',
    'ANKH-Large',
    'ANKH2-Large',
    'GLM2-150',
    'GLM2-650',
    'GLM2-GAIA',
    'DPLM-150',
    'DPLM-650',
    'DPLM-3B',
    'DPLM2-150',
    'DPLM2-650',
    'DPLM2-3B',
    'DSM-150',
    'DSM-650',
    'DSM-PPI',
    'OneHot-Protein',
    'OneHot-DNA',
    'OneHot-RNA',
    'OneHot-Codon',
    'AMPLIFY-120',
    'AMPLIFY-350',
    'CaLM',
]

standard_models = [
    'AMPLIFY-120',
    'AMPLIFY-350',
    'GLM2-150',
    'GLM2-650',
    'GLM2-GAIA',
    'DSM-150',
    'DSM-650',
    'DSM-PPI',
    'E1-150',
    'E1-300',
    'E1-600',
    'ESM2-8',
    'ESM2-35',
    'ESM2-150',
    'ESM2-650',
    'ESM2-3B',
    'ESMC-300',
    'ESMC-600',
    'ProtBert-BFD',
    'ProtT5',
    'ANKH-Base',
    'ANKH-Large',
    'ANKH2-Large',
    'DPLM-150',
    'DPLM-650',
    'DPLM-3B',
    'DPLM2-150',
    'DPLM2-650',
    'DPLM2-3B',
    'Random',
    'Random-Transformer',
    'OneHot-Protein',
]

experimental_models = []
