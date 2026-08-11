import entrypoint_setup

import os
import tkinter as tk
import argparse
import base64
import json
import queue
import subprocess
import sys
import traceback
import webbrowser
from types import SimpleNamespace
from tkinter import ttk, messagebox, filedialog
from concurrent.futures import ThreadPoolExecutor

from base_models.get_base_models import BaseModelArguments, standard_models
from data.supported_datasets import supported_datasets, standard_data_benchmark, internal_datasets
from embedder import EmbeddingArguments
from probes.estimator_probe import is_estimator_probe
from probes.get_probe import ProbeArguments
from probes.trainers import TrainerArguments
from main import MainProcess
from data.data_mixin import DataArguments
from cloud_backend import get_or_create_cloud_backend
from utils import print_message, print_done, print_title, expand_dms_ids_all
from visualization.plot_result import create_plots
from benchmarks.proteingym.compare_scoring_methods import compare_scoring_methods
from hyperopt_utils import HyperoptModule


class BackgroundTask:
    def __init__(self, target, *args, **kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        self._complete = False
        
    def run(self):
        try:
            self.result = self.target(*self.args, **self.kwargs)
        except Exception as e:
            self.error = e
            print_message(f"Error in background task: {str(e)}")
            traceback.print_exc()
        finally:
            self._complete = True
    
    @property
    def complete(self):
        return self._complete


class GUI(MainProcess):
    def __init__(self, master):
        super().__init__(argparse.Namespace(), GUI=True)  # Initialize MainProcess with empty namespace
        self.master = master
        self.master.title("Settings GUI")
        self.master.geometry("600x800")

        icon = tk.PhotoImage(file="protify_logo.png")
        # Set the window icon
        self.master.iconphoto(True, icon)

        # Dictionary to store Tkinter variables for settings
        self.settings_vars = {}

        # Create the Notebook widget
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for each settings tab
        self.info_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.embed_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.probe_tab = ttk.Frame(self.notebook)
        self.trainer_tab = ttk.Frame(self.notebook)
        self.wandb_tab = ttk.Frame(self.notebook)
        self.cloud_tab = ttk.Frame(self.notebook)
        self.scikit_tab = ttk.Frame(self.notebook)
        self.replay_tab = ttk.Frame(self.notebook)
        self.viz_tab = ttk.Frame(self.notebook)
        self.proteingym_tab = ttk.Frame(self.notebook)

        # Add tabs to the notebook
        self.notebook.add(self.info_tab, text="Info")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.embed_tab, text="Embedding")
        self.notebook.add(self.probe_tab, text="Probe")
        self.notebook.add(self.trainer_tab, text="Trainer")
        self.notebook.add(self.wandb_tab, text="W&B Sweep")
        self.notebook.add(self.cloud_tab, text="Cloud")
        self.notebook.add(self.proteingym_tab, text="ProteinGym")
        self.notebook.add(self.scikit_tab, text="Scikit")
        self.notebook.add(self.replay_tab, text="Replay")
        self.notebook.add(self.viz_tab, text="Visualization")

        # Build these lines
        self.task_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.current_task = None
        self.cloud_polling_active = False
        
        # Start the queue checker
        self.check_task_queue()

        # Build each tab
        self.build_info_tab()
        self.build_model_tab()
        self.build_data_tab()
        self.build_embed_tab()
        self.build_probe_tab()
        self.build_trainer_tab()
        self.build_wandb_tab()
        self.build_cloud_tab()
        self.build_proteingym_tab()
        self.build_scikit_tab()
        self.build_replay_tab()
        self.build_viz_tab()

    def check_task_queue(self):
        """Periodically check for completed background tasks"""
        if self.current_task and self.current_task.complete:
            if self.current_task.error:
                print_message(f"Task failed: {self.current_task.error}")
            self.current_task = None
            
        if not self.current_task and not self.task_queue.empty():
            self.current_task = self.task_queue.get()
            self.thread_pool.submit(self.current_task.run)
        
        # Schedule next check
        self.master.after(100, self.check_task_queue)
    
    def run_in_background(self, target, *args, **kwargs):
        """Queue a task to run in background"""
        task = BackgroundTask(target, *args, **kwargs)
        self.task_queue.put(task)
        return task

    def _open_url(self, url):
        """Open a URL in the default web browser"""
        webbrowser.open_new_tab(url)
        
    def build_info_tab(self):
        # Create a frame for IDs
        id_frame = ttk.LabelFrame(self.info_tab, text="Identification")
        id_frame.pack(fill="x", padx=10, pady=5)

        # Huggingface Username
        ttk.Label(id_frame, text="Huggingface Username:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["huggingface_username"] = tk.StringVar(value="Synthyra")
        entry_huggingface_username = ttk.Entry(id_frame, textvariable=self.settings_vars["huggingface_username"], width=30)
        entry_huggingface_username.grid(row=0, column=1, padx=10, pady=5)
        self.add_help_button(id_frame, 0, 2, "Your Hugging Face username for model downloads and uploads.")

        # Huggingface token
        ttk.Label(id_frame, text="Huggingface Token:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["huggingface_token"] = tk.StringVar(value="")
        entry_huggingface_token = ttk.Entry(id_frame, textvariable=self.settings_vars["huggingface_token"], width=30)
        entry_huggingface_token.grid(row=1, column=1, padx=10, pady=5)
        self.add_help_button(id_frame, 1, 2, "Your Hugging Face API token for accessing gated or private models.")

        # Wandb API key 
        ttk.Label(id_frame, text="Wandb API Key:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["wandb_api_key"] = tk.StringVar(value="")
        entry_wandb_api_key = ttk.Entry(id_frame, textvariable=self.settings_vars["wandb_api_key"], width=30)
        entry_wandb_api_key.grid(row=2, column=1, padx=10, pady=5)
        self.add_help_button(id_frame, 2, 2, "Your Weights & Biases API key for experiment tracking.")

        # Synthyra API key
        ttk.Label(id_frame, text="Synthyra API Key:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["synthyra_api_key"] = tk.StringVar(value="")
        entry_synthyra_api_key = ttk.Entry(id_frame, textvariable=self.settings_vars["synthyra_api_key"], width=30)
        entry_synthyra_api_key.grid(row=3, column=1, padx=10, pady=5)
        self.add_help_button(id_frame, 3, 2, "Your Synthyra API key for accessing premium features.")

        # Cloud API key
        ttk.Label(id_frame, text="Cloud API Key:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_api_key"] = tk.StringVar(value="")
        entry_cloud_api_key = ttk.Entry(id_frame, textvariable=self.settings_vars["cloud_api_key"], width=30, show="*")
        entry_cloud_api_key.grid(row=4, column=1, padx=10, pady=5)
        self.add_help_button(id_frame, 4, 2, "API key for cloud backend execution.")

        # Create a frame for paths
        paths_frame = ttk.LabelFrame(self.info_tab, text="Paths")
        paths_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(paths_frame, text='Home Directory:').grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["home_dir"] = tk.StringVar(value=os.getcwd())
        entry_home_dir = ttk.Entry(paths_frame, textvariable=self.settings_vars["home_dir"], width=30)
        entry_home_dir.grid(row=0, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 0, 2, "Home directory for Protify.")

        # HF Home directory
        ttk.Label(paths_frame, text="HF Home Directory:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["hf_home"] = tk.StringVar(value="")
        entry_hf_home = ttk.Entry(paths_frame, textvariable=self.settings_vars["hf_home"], width=30)
        entry_hf_home.grid(row=1, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 1, 2, "Customize the HuggingFace cache directory. Leave empty to use default.")

        # Log directory
        ttk.Label(paths_frame, text="Log Directory:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["log_dir"] = tk.StringVar(value="logs")
        entry_log_dir = ttk.Entry(paths_frame, textvariable=self.settings_vars["log_dir"], width=30)
        entry_log_dir.grid(row=2, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 2, 2, "Directory where log files will be stored.")

        # Results directory
        ttk.Label(paths_frame, text="Results Directory:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["results_dir"] = tk.StringVar(value="results")
        entry_results_dir = ttk.Entry(paths_frame, textvariable=self.settings_vars["results_dir"], width=30)
        entry_results_dir.grid(row=3, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 3, 2, "Directory where results data will be stored.")

        # Model save directory
        ttk.Label(paths_frame, text="Model Save Directory:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["model_save_dir"] = tk.StringVar(value="weights")
        entry_model_save = ttk.Entry(paths_frame, textvariable=self.settings_vars["model_save_dir"], width=30)
        entry_model_save.grid(row=4, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 4, 2, "Directory where trained models will be saved.")

        ttk.Label(paths_frame, text="Plots Directory:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["plots_dir"] = tk.StringVar(value="plots")
        entry_plots_dir = ttk.Entry(paths_frame, textvariable=self.settings_vars["plots_dir"], width=30)
        entry_plots_dir.grid(row=5, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 5, 2, "Directory where plots and visualizations will be saved.")

        # Embedding save directory
        ttk.Label(paths_frame, text="Embedding Save Directory:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["embedding_save_dir"] = tk.StringVar(value="embeddings")
        entry_embed_save = ttk.Entry(paths_frame, textvariable=self.settings_vars["embedding_save_dir"], width=30)
        entry_embed_save.grid(row=6, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 6, 2, "Directory where computed embeddings will be saved.")

        # Download directory
        ttk.Label(paths_frame, text="Download Directory:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["download_dir"] = tk.StringVar(value="Synthyra/vector_embeddings")
        entry_download = ttk.Entry(paths_frame, textvariable=self.settings_vars["download_dir"], width=30)
        entry_download.grid(row=7, column=1, padx=10, pady=5)
        self.add_help_button(paths_frame, 7, 2, "HuggingFace repository path for downloading pre-computed embeddings.")

        # button to start logging
        start_logging_button = ttk.Button(self.info_tab, text="Start session", command=self._session_start)
        start_logging_button.pack(pady=10)
        
        # Add logo and website link at the bottom of the info tab
        try:
            original_logo = tk.PhotoImage(file="synthyra_logo.png")
            # Make logo even smaller (subsample by factor of 3)
            logo = original_logo.subsample(3, 3)
            
            # Create frame to hold logo and button side by side
            bottom_frame = ttk.Frame(self.info_tab)
            bottom_frame.pack(pady=(10, 20), fill="x")
            
            # Place logo on the left side
            logo_label = ttk.Label(bottom_frame, image=logo, cursor="hand2")
            logo_label.image = logo  # Keep a reference to prevent garbage collection
            logo_label.pack(side=tk.LEFT, padx=(20, 10))
            # Bind click event to the logo
            logo_label.bind("<Button-1>", lambda e: self._open_url("https://synthyra.com"))
            
            # Add a "Visit Website" button on the right side
            visit_btn = ttk.Button(
                bottom_frame,
                text="Visit Synthyra.com",
                command=lambda: self._open_url("https://synthyra.com"),
                style="Link.TButton"
            )
            
            # Create a special style for the link button
            style = ttk.Style()
            style.configure("Link.TButton", font=("Helvetica", 12), foreground="blue")
            
            visit_btn.pack(side=tk.LEFT, padx=(10, 20), pady=10)
            
        except Exception as e:
            print_message(f"Error setting up logo and link: {str(e)}")

    def build_model_tab(self):
        ttk.Label(self.model_tab, text="Model Names:").grid(row=0, column=0, padx=10, pady=5, sticky="nw")

        self.model_listbox = tk.Listbox(self.model_tab, selectmode="extended", height=24)
        for model_name in standard_models:
            self.model_listbox.insert(tk.END, model_name)
        self.model_listbox.grid(row=0, column=1, padx=10, pady=5, sticky="nw")
        self.add_help_button(self.model_tab, 0, 2, "Select the language models to use for embedding. Multiple models can be selected.")

        ttk.Label(self.model_tab, text="Model DType:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["model_dtype"] = tk.StringVar(value="bf16")
        combo_model_dtype = ttk.Combobox(
            self.model_tab,
            textvariable=self.settings_vars["model_dtype"],
            values=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
            state="readonly",
        )
        combo_model_dtype.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.model_tab, 1, 2, "Data type used when loading base models.")

        ttk.Label(self.model_tab, text="Use xformers:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["use_xformers"] = tk.BooleanVar(value=False)
        check_use_xformers = ttk.Checkbutton(self.model_tab, variable=self.settings_vars["use_xformers"])
        check_use_xformers.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.model_tab, 2, 2, "Enable memory-efficient xformers attention where supported.")

        run_button = ttk.Button(self.model_tab, text="Select Models", command=self._select_models)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_data_tab(self):
        ttk.Label(self.data_tab, text="Max Sequence Length:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["max_length"] = tk.IntVar(value=2048)
        spin_max_length = ttk.Spinbox(self.data_tab, from_=1, to=32768, textvariable=self.settings_vars["max_length"])
        spin_max_length.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.data_tab, 0, 2, "Maximum length of sequences (in tokens) to process.")

        ttk.Label(self.data_tab, text="Trim Sequences:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["trim"] = tk.BooleanVar(value=False)
        check_trim = ttk.Checkbutton(self.data_tab, variable=self.settings_vars["trim"])
        check_trim.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.data_tab, 1, 2, "Whether to trim sequences to the specified max length.")

        ttk.Label(self.data_tab, text="Delimiter:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["delimiter"] = tk.StringVar(value=",")
        entry_delimiter = ttk.Entry(self.data_tab, textvariable=self.settings_vars["delimiter"], width=5)
        entry_delimiter.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.data_tab, 2, 2, "Character used to separate columns in CSV data files.")

        ttk.Label(self.data_tab, text="Column Names (comma-separated):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["col_names"] = tk.StringVar(value="seqs,labels")
        entry_col_names = ttk.Entry(self.data_tab, textvariable=self.settings_vars["col_names"], width=20)
        entry_col_names.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.data_tab, 3, 2, "Names of columns in data files, separate with commas.")

        ttk.Label(self.data_tab, text="Multi-Column Sequences (space-separated):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["multi_column"] = tk.StringVar(value="")
        entry_multi_column = ttk.Entry(self.data_tab, textvariable=self.settings_vars["multi_column"], width=20)
        entry_multi_column.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.data_tab, 4, 2, "If set, list of sequence column names to combine per sample (space-separated). Leave empty if not using multi-column sequences.")

        ttk.Label(self.data_tab, text="Local Data Directories (comma-separated):").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["data_dirs"] = tk.StringVar(value="")
        entry_data_dirs = ttk.Entry(self.data_tab, textvariable=self.settings_vars["data_dirs"], width=30)
        entry_data_dirs.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        browse_data_dir_button = ttk.Button(self.data_tab, text="Browse", command=self._browse_data_dir)
        browse_data_dir_button.grid(row=5, column=2, padx=5, pady=5)
        self.add_help_button(self.data_tab, 5, 3, "Optional local dataset directories. Multiple paths can be comma-separated.")

        ttk.Label(self.data_tab, text="AA -> DNA:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["aa_to_dna"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.data_tab, variable=self.settings_vars["aa_to_dna"]).grid(row=6, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.data_tab, text="AA -> RNA:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["aa_to_rna"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.data_tab, variable=self.settings_vars["aa_to_rna"]).grid(row=7, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.data_tab, text="DNA -> AA:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["dna_to_aa"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.data_tab, variable=self.settings_vars["dna_to_aa"]).grid(row=8, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.data_tab, text="RNA -> AA:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["rna_to_aa"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.data_tab, variable=self.settings_vars["rna_to_aa"]).grid(row=9, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.data_tab, text="Codon -> AA:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["codon_to_aa"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.data_tab, variable=self.settings_vars["codon_to_aa"]).grid(row=10, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.data_tab, text="AA -> Codon:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["aa_to_codon"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.data_tab, variable=self.settings_vars["aa_to_codon"]).grid(row=11, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.data_tab, text="Random Pair Flipping:").grid(row=12, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["random_pair_flipping"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.data_tab, variable=self.settings_vars["random_pair_flipping"]).grid(row=12, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.data_tab, 12, 2, "Randomly flip paired inputs during training for pair datasets.")

        ttk.Label(self.data_tab, text="Dataset Names:").grid(row=13, column=0, padx=10, pady=5, sticky="nw")
        self.data_listbox = tk.Listbox(self.data_tab, selectmode="extended", height=20, width=25)
        for dataset_name in supported_datasets:
            if dataset_name not in internal_datasets:
                self.data_listbox.insert(tk.END, dataset_name)
        self.data_listbox.grid(row=13, column=1, padx=10, pady=5, sticky="nw")
        self.add_help_button(self.data_tab, 13, 2, "Select datasets to use. Multiple datasets can be selected.")

        run_button = ttk.Button(self.data_tab, text="Get Data", command=self._get_data)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_embed_tab(self):
        # batch_size
        ttk.Label(self.embed_tab, text="Batch Size:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["batch_size"] = tk.IntVar(value=4)
        spin_batch_size = ttk.Spinbox(self.embed_tab, from_=1, to=1024, textvariable=self.settings_vars["batch_size"])
        spin_batch_size.grid(row=1, column=1, padx=10, pady=5)
        self.add_help_button(self.embed_tab, 1, 2, "Number of sequences to process at once during embedding.")

        # num_workers
        ttk.Label(self.embed_tab, text="Num Workers:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_workers"] = tk.IntVar(value=0)
        spin_num_workers = ttk.Spinbox(self.embed_tab, from_=0, to=64, textvariable=self.settings_vars["num_workers"])
        spin_num_workers.grid(row=2, column=1, padx=10, pady=5)
        self.add_help_button(self.embed_tab, 2, 2, "Number of worker processes for data loading. 0 means main process only.")

        # download_embeddings
        ttk.Label(self.embed_tab, text="Download Embeddings:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["download_embeddings"] = tk.BooleanVar(value=False)
        check_download = ttk.Checkbutton(self.embed_tab, variable=self.settings_vars["download_embeddings"])
        check_download.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.embed_tab, 3, 2, "Whether to download pre-computed embeddings from HuggingFace instead of computing them.")

        # matrix_embed
        ttk.Label(self.embed_tab, text="Matrix Embedding:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["matrix_embed"] = tk.BooleanVar(value=False)
        check_matrix = ttk.Checkbutton(self.embed_tab, variable=self.settings_vars["matrix_embed"])
        check_matrix.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.embed_tab, 4, 2, "Whether to use matrix embedding (full embedding matrices) instead of pooled embeddings.")

        # pooling_types
        ttk.Label(self.embed_tab, text="Pooling Types (comma-separated):").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["embedding_pooling_types"] = tk.StringVar(value="mean, var")
        entry_pooling = ttk.Entry(self.embed_tab, textvariable=self.settings_vars["embedding_pooling_types"], width=20)
        entry_pooling.grid(row=5, column=1, padx=10, pady=5)
        self.add_help_button(self.embed_tab, 5, 2, "Types of pooling to apply to embeddings, separate with commas.")
        
        ttk.Label(self.embed_tab, text="Options: mean, max, min, norm, prod, median, std, var, cls, parti").grid(row=6, column=0, columnspan=2, padx=10, pady=2, sticky="w")

        ttk.Label(self.embed_tab, text="Hidden State Index:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["embedding_hidden_state_index"] = tk.IntVar(value=-1)
        spin_hidden_state = ttk.Spinbox(self.embed_tab, from_=-1, to=100, textvariable=self.settings_vars["embedding_hidden_state_index"])
        spin_hidden_state.grid(row=7, column=1, padx=10, pady=5)
        self.add_help_button(self.embed_tab, 7, 2, "-1 uses the final hidden state. Non-negative values select a hidden-state tuple index.")

        # embed_dtype
        ttk.Label(self.embed_tab, text="Embedding DType:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["embed_dtype"] = tk.StringVar(value="float32")
        combo_dtype = ttk.Combobox(
            self.embed_tab,
            textvariable=self.settings_vars["embed_dtype"],
            values=["float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"]
        )
        combo_dtype.grid(row=8, column=1, padx=10, pady=5)
        self.add_help_button(self.embed_tab, 8, 2, "Data type to use for storing embeddings (affects precision and size).")

        # sql
        ttk.Label(self.embed_tab, text="Use SQL:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sql"] = tk.BooleanVar(value=False)
        check_sql = ttk.Checkbutton(self.embed_tab, variable=self.settings_vars["sql"])
        check_sql.grid(row=9, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.embed_tab, 9, 2, "Whether to use SQL database for storing embeddings instead of files.")

        run_button = ttk.Button(self.embed_tab, text="Embed sequences to disk", command=self._get_embeddings)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_probe_tab(self):
        # Probe Type
        ttk.Label(self.probe_tab, text="Probe Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_type"] = tk.StringVar(value="mlp")
        combo_probe = ttk.Combobox(
            self.probe_tab,
            textvariable=self.settings_vars["probe_type"],
            values=["mlp", "transformer", "lyra", "xgboost", "lightgbm", "random_forest"]
        )
        combo_probe.grid(row=0, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 0, 2, "Probe to train on the embeddings: mlp, transformer, lyra, xgboost, lightgbm, or random_forest.")

        # Tokenwise
        ttk.Label(self.probe_tab, text="Tokenwise:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["tokenwise"] = tk.BooleanVar(value=False)
        check_tokenwise = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["tokenwise"])
        check_tokenwise.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 1, 2, "Whether to use token-wise prediction (operate on each token) instead of sequence-level.")

        # Pre Layer Norm
        ttk.Label(self.probe_tab, text="Pre Layer Norm:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["pre_ln"] = tk.BooleanVar(value=True)
        check_pre_ln = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["pre_ln"])
        check_pre_ln.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 2, 2, "Whether to use pre-layer normalization in transformer architecture.")

        # Number of Layers
        ttk.Label(self.probe_tab, text="Number of Layers:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["n_layers"] = tk.IntVar(value=1)
        spin_n_layers = ttk.Spinbox(self.probe_tab, from_=1, to=100, textvariable=self.settings_vars["n_layers"])
        spin_n_layers.grid(row=3, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 3, 2, "Number of layers in the probe architecture.")

        # Hidden Dimension
        ttk.Label(self.probe_tab, text="Hidden Dimension:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["hidden_size"] = tk.IntVar(value=8192)
        spin_hidden_size = ttk.Spinbox(self.probe_tab, from_=1, to=10000, textvariable=self.settings_vars["hidden_size"])
        spin_hidden_size.grid(row=4, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 4, 2, "Size of hidden dimension in the probe model.")

        # Dropout
        ttk.Label(self.probe_tab, text="Dropout:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["dropout"] = tk.DoubleVar(value=0.2)
        spin_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=1.0, increment=0.1, textvariable=self.settings_vars["dropout"])
        spin_dropout.grid(row=5, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 5, 2, "Dropout probability for regularization (0.0-1.0).")

        # Transformer Probe Settings
        ttk.Label(self.probe_tab, text="=== Transformer Probe Settings ===").grid(row=6, column=0, columnspan=2, pady=10)

        # FF Dimension
        ttk.Label(self.probe_tab, text="Classifier Dimension:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["classifier_size"] = tk.IntVar(value=4096)
        spin_classifier_size = ttk.Spinbox(self.probe_tab, from_=1, to=10000, textvariable=self.settings_vars["classifier_size"])
        spin_classifier_size.grid(row=8, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 8, 2, "Dimension of the classifier/feedforward layer in transformer probe.")

        # Classifier Dropout
        ttk.Label(self.probe_tab, text="Classifier Dropout:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["classifier_dropout"] = tk.DoubleVar(value=0.2)
        spin_class_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=1.0, increment=0.1, textvariable=self.settings_vars["classifier_dropout"])
        spin_class_dropout.grid(row=9, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 9, 2, "Dropout probability in the classifier layer (0.0-1.0).")

        # Number of Heads
        ttk.Label(self.probe_tab, text="Number of Heads:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["n_heads"] = tk.IntVar(value=4)
        spin_n_heads = ttk.Spinbox(self.probe_tab, from_=1, to=32, textvariable=self.settings_vars["n_heads"])
        spin_n_heads.grid(row=10, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 10, 2, "Number of attention heads in transformer probe.")

        # Rotary
        ttk.Label(self.probe_tab, text="Rotary:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["rotary"] = tk.BooleanVar(value=True)
        check_rotary = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["rotary"])
        check_rotary.grid(row=11, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 11, 2, "Whether to use rotary position embeddings in transformer.")

        # Pooling Types
        ttk.Label(self.probe_tab, text="Pooling Types (comma-separated):").grid(row=12, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_pooling_types"] = tk.StringVar(value="mean, var")
        entry_pooling = ttk.Entry(self.probe_tab, textvariable=self.settings_vars["probe_pooling_types"], width=20)
        entry_pooling.grid(row=12, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 12, 2, "Types of pooling to use in the probe model, separate with commas.")
        
        # Transformer Dropout
        ttk.Label(self.probe_tab, text="Transformer Dropout:").grid(row=13, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["transformer_dropout"] = tk.DoubleVar(value=0.1)
        spin_transformer_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=1.0, increment=0.1, textvariable=self.settings_vars["transformer_dropout"])
        spin_transformer_dropout.grid(row=13, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 13, 2, "Dropout probability in the transformer layers (0.0-1.0).")
        
        # Attention Backend
        ttk.Label(self.probe_tab, text="Attention Backend:").grid(row=14, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["attention_backend"] = tk.StringVar(value="flex")
        backend_combo = ttk.Combobox(
            self.probe_tab,
            textvariable=self.settings_vars["attention_backend"],
            values=["kernels", "flex", "sdpa"],
            state="readonly",
            width=17,
        )
        backend_combo.grid(row=14, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 14, 2, "Select the attention backend for transformer-style probes.")

        # Padding Strategy
        ttk.Label(self.probe_tab, text="Padding:").grid(row=15, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["padding"] = tk.StringVar(value="max_length")
        padding_combo = ttk.Combobox(
            self.probe_tab,
            textvariable=self.settings_vars["padding"],
            values=["max_length", "longest"],
            state="readonly",
            width=17,
        )
        padding_combo.grid(row=15, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 15, 2, "Padding strategy. 'max_length' (recommended) pads all sequences to max_length for optimal torch.compile + flex attention performance. 'longest' pads to the longest sequence in each batch.")

        # Use Bias
        ttk.Label(self.probe_tab, text="Use Bias:").grid(row=16, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["use_bias"] = tk.BooleanVar(value=False)
        check_use_bias = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["use_bias"])
        check_use_bias.grid(row=16, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 16, 2, "Use bias terms in probe linear layers.")

        # Add Token IDs
        ttk.Label(self.probe_tab, text="Add Token IDs:").grid(row=16, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["add_token_ids"] = tk.BooleanVar(value=False)
        check_add_token_ids = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["add_token_ids"])
        check_add_token_ids.grid(row=16, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 16, 2, "Add learned token type IDs for pair tasks.")

        # Save Model
        ttk.Label(self.probe_tab, text="Save Model:").grid(row=17, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["save_model"] = tk.BooleanVar(value=False)
        check_save_model = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["save_model"])
        check_save_model.grid(row=19, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 19, 2, "Whether to save the trained probe model to disk.")

        # Push Raw Probe
        ttk.Label(self.probe_tab, text="Push Raw Probe:").grid(row=20, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["push_raw_probe"] = tk.BooleanVar(value=False)
        check_push_raw_probe = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["push_raw_probe"])
        check_push_raw_probe.grid(row=20, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 20, 2, "With Save Model, push raw probe class to Hub (load with e.g. Class.from_pretrained(repo_id)) instead of packaged AutoModel.")

        # Production Model
        ttk.Label(self.probe_tab, text="Production Model:").grid(row=21, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["production_model"] = tk.BooleanVar(value=False)
        check_prod_model = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["production_model"])
        check_prod_model.grid(row=21, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 21, 2, "Whether to prepare the model for production deployment.")

        # LoRA Settings Section
        ttk.Label(self.probe_tab, text="=== LoRA Settings ===").grid(row=22, column=0, columnspan=2, pady=10)
        
        # Lora checkbox
        ttk.Label(self.probe_tab, text="Use LoRA:").grid(row=23, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lora"] = tk.BooleanVar(value=False)
        check_lora = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["lora"])
        check_lora.grid(row=23, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.probe_tab, 23, 2, "Whether to use Low-Rank Adaptation (LoRA) for fine-tuning.")

        # LoRA r
        ttk.Label(self.probe_tab, text="LoRA r:").grid(row=24, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lora_r"] = tk.IntVar(value=8)
        spin_lora_r = ttk.Spinbox(self.probe_tab, from_=1, to=128, textvariable=self.settings_vars["lora_r"])
        spin_lora_r.grid(row=24, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 24, 2, "Rank parameter r for LoRA (lower = more efficient, higher = more expressive).")

        # LoRA alpha
        ttk.Label(self.probe_tab, text="LoRA alpha:").grid(row=25, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lora_alpha"] = tk.DoubleVar(value=32.0)
        spin_lora_alpha = ttk.Spinbox(self.probe_tab, from_=1.0, to=128.0, increment=1.0, textvariable=self.settings_vars["lora_alpha"])
        spin_lora_alpha.grid(row=25, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 25, 2, "Alpha parameter for LoRA, controls update scale.")

        # LoRA dropout
        ttk.Label(self.probe_tab, text="LoRA dropout:").grid(row=26, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lora_dropout"] = tk.DoubleVar(value=0.01)
        spin_lora_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=0.5, increment=0.01, textvariable=self.settings_vars["lora_dropout"])
        spin_lora_dropout.grid(row=26, column=1, padx=10, pady=5)
        self.add_help_button(self.probe_tab, 26, 2, "Dropout probability for LoRA layers (0.0-0.5).")
        
        # Add a button to create the probe
        run_button = ttk.Button(self.probe_tab, text="Save Probe Arguments", command=self._create_probe_args)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_trainer_tab(self):
        # Hybrid Probe checkbox
        ttk.Label(self.trainer_tab, text="Hybrid Probe:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["hybrid_probe"] = tk.BooleanVar(value=False)
        check_hybrid_probe = ttk.Checkbutton(self.trainer_tab, variable=self.settings_vars["hybrid_probe"])
        check_hybrid_probe.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 0, 2, "Whether to use hybrid probe (trains the base model together with the probe).")

        # Full finetuning checkbox
        ttk.Label(self.trainer_tab, text="Full Finetuning:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["full_finetuning"] = tk.BooleanVar(value=False)
        check_full_ft = ttk.Checkbutton(self.trainer_tab, variable=self.settings_vars["full_finetuning"])
        check_full_ft.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 1, 2, "Whether to perform full finetuning of the entire model.")

        # num_epochs
        ttk.Label(self.trainer_tab, text="Number of Epochs:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_epochs"] = tk.IntVar(value=200)
        spin_num_epochs = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["num_epochs"])
        spin_num_epochs.grid(row=2, column=1, padx=10, pady=5)
        self.add_help_button(self.trainer_tab, 2, 2, "Number of training epochs (complete passes through the dataset).")

        # probe_batch_size
        ttk.Label(self.trainer_tab, text="Probe Batch Size:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_batch_size"] = tk.IntVar(value=64)
        spin_probe_batch_size = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["probe_batch_size"])
        spin_probe_batch_size.grid(row=3, column=1, padx=10, pady=5)
        self.add_help_button(self.trainer_tab, 3, 2, "Batch size for probe training.")

        # base_batch_size
        ttk.Label(self.trainer_tab, text="Base Batch Size:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["base_batch_size"] = tk.IntVar(value=4)
        spin_base_batch_size = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["base_batch_size"])
        spin_base_batch_size.grid(row=4, column=1, padx=10, pady=5)
        self.add_help_button(self.trainer_tab, 4, 2, "Batch size for base model training.")

        # probe_grad_accum
        ttk.Label(self.trainer_tab, text="Probe Grad Accum:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_grad_accum"] = tk.IntVar(value=1)
        spin_probe_grad_accum = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["probe_grad_accum"])
        spin_probe_grad_accum.grid(row=5, column=1, padx=10, pady=5)
        self.add_help_button(self.trainer_tab, 5, 2, "Gradient accumulation steps for probe training.")

        # base_grad_accum
        ttk.Label(self.trainer_tab, text="Base Grad Accum:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["base_grad_accum"] = tk.IntVar(value=8)
        spin_base_grad_accum = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["base_grad_accum"])
        spin_base_grad_accum.grid(row=6, column=1, padx=10, pady=5)
        self.add_help_button(self.trainer_tab, 6, 2, "Gradient accumulation steps for base model training.")

        # lr
        ttk.Label(self.trainer_tab, text="Learning Rate:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lr"] = tk.DoubleVar(value=1e-4)
        spin_lr = ttk.Spinbox(self.trainer_tab, from_=1e-6, to=1e-2, increment=1e-5, textvariable=self.settings_vars["lr"])
        spin_lr.grid(row=7, column=1, padx=10, pady=5)
        self.add_help_button(self.trainer_tab, 7, 2, "Learning rate for optimizer. Controls step size during training.")

        ttk.Label(self.trainer_tab, text="Probe Learning Rate:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_lr"] = tk.StringVar(value="")
        entry_probe_lr = ttk.Entry(self.trainer_tab, textvariable=self.settings_vars["probe_lr"], width=20)
        entry_probe_lr.grid(row=8, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 8, 2, "Optional probe-phase learning rate. Leave blank to use Learning Rate.")

        ttk.Label(self.trainer_tab, text="Base Learning Rate:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["base_lr"] = tk.StringVar(value="")
        entry_base_lr = ttk.Entry(self.trainer_tab, textvariable=self.settings_vars["base_lr"], width=20)
        entry_base_lr.grid(row=9, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 9, 2, "Optional base-model phase learning rate. Leave blank to use Learning Rate.")

        ttk.Label(self.trainer_tab, text="LR Scheduler:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lr_scheduler"] = tk.StringVar(value="cosine")
        combo_lr_scheduler = ttk.Combobox(
            self.trainer_tab,
            textvariable=self.settings_vars["lr_scheduler"],
            values=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"],
            state="readonly",
            width=20,
        )
        combo_lr_scheduler.grid(row=10, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 10, 2, "Learning-rate scheduler passed to Hugging Face TrainingArguments.")

        ttk.Label(self.trainer_tab, text="Optimizer:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["optimizer"] = tk.StringVar(value="adamw_torch")
        combo_optimizer = ttk.Combobox(
            self.trainer_tab,
            textvariable=self.settings_vars["optimizer"],
            values=["adamw_torch", "adamw_torch_fused", "adamw_hf", "adafactor", "sgd"],
            width=20,
        )
        combo_optimizer.grid(row=11, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 11, 2, "Optimizer name passed to Hugging Face TrainingArguments.")

        # weight_decay
        ttk.Label(self.trainer_tab, text="Weight Decay:").grid(row=12, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["weight_decay"] = tk.DoubleVar(value=0.00)
        spin_weight_decay = ttk.Spinbox(self.trainer_tab, from_=0.0, to=1.0, increment=0.01, textvariable=self.settings_vars["weight_decay"])
        spin_weight_decay.grid(row=12, column=1, padx=10, pady=5)
        self.add_help_button(self.trainer_tab, 12, 2, "L2 regularization factor to prevent overfitting (0.0-1.0).")

        # patience
        ttk.Label(self.trainer_tab, text="Patience:").grid(row=13, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["patience"] = tk.IntVar(value=1)
        spin_patience = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["patience"])
        spin_patience.grid(row=13, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 13, 2, "Number of epochs with no improvement after which training will stop.")

        # Random Seed
        ttk.Label(self.trainer_tab, text="Random Seed:").grid(row=14, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["seed"] = tk.IntVar(value=42)
        spin_seed = ttk.Spinbox(self.trainer_tab, from_=0, to=10000, textvariable=self.settings_vars["seed"])
        spin_seed.grid(row=14, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 14, 2, "Random seed for reproducibility of experiments.")

        # Read Scaler
        ttk.Label(self.trainer_tab, text="Read Scaler:").grid(row=15, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["read_scaler"] = tk.IntVar(value=100)
        spin_read_scaler = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["read_scaler"])
        spin_read_scaler.grid(row=15, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 15, 2, "Read scaler for SQL storage (multiplier for batch size when reading from SQL database).")

        # Deterministic
        ttk.Label(self.trainer_tab, text="Deterministic:").grid(row=16, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["deterministic"] = tk.BooleanVar(value=False)
        check_deterministic = ttk.Checkbutton(self.trainer_tab, variable=self.settings_vars["deterministic"])
        check_deterministic.grid(row=16, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 16, 2, "Enable deterministic behavior for reproducibility (will slow down training).")

        # Number of Runs
        ttk.Label(self.trainer_tab, text="Number of Runs:").grid(row=17, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_runs"] = tk.IntVar(value=1)
        spin_num_runs = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["num_runs"])
        spin_num_runs.grid(row=17, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.trainer_tab, 17, 2, "Train multiple runs with different seeds and aggregate metrics.")

        run_button = ttk.Button(self.trainer_tab, text="Run trainer", command=self._run_trainer)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_wandb_tab(self):
        ttk.Label(self.wandb_tab, text="Use W&B Hyperopt:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["use_wandb_hyperopt"] = tk.BooleanVar(value=False)
        check_use_wandb_hyperopt = ttk.Checkbutton(self.wandb_tab, variable=self.settings_vars["use_wandb_hyperopt"])
        check_use_wandb_hyperopt.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.wandb_tab, 0, 2, "Enable Weights & Biases hyperparameter sweeps.")

        ttk.Label(self.wandb_tab, text="W&B Project:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["wandb_project"] = tk.StringVar(value="Protify")
        entry_wandb_project = ttk.Entry(self.wandb_tab, textvariable=self.settings_vars["wandb_project"], width=30)
        entry_wandb_project.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.wandb_tab, text="W&B Entity (optional):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["wandb_entity"] = tk.StringVar(value="")
        entry_wandb_entity = ttk.Entry(self.wandb_tab, textvariable=self.settings_vars["wandb_entity"], width=30)
        entry_wandb_entity.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.wandb_tab, text="Sweep Config Path:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sweep_config_path"] = tk.StringVar(value="yamls/sweep.yaml")
        entry_sweep_config_path = ttk.Entry(self.wandb_tab, textvariable=self.settings_vars["sweep_config_path"], width=30)
        entry_sweep_config_path.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        browse_sweep_path_button = ttk.Button(self.wandb_tab, text="Browse", command=self._browse_sweep_config)
        browse_sweep_path_button.grid(row=3, column=2, padx=5, pady=5)

        ttk.Label(self.wandb_tab, text="Sweep Count:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sweep_count"] = tk.IntVar(value=10)
        spin_sweep_count = ttk.Spinbox(self.wandb_tab, from_=1, to=10000, textvariable=self.settings_vars["sweep_count"])
        spin_sweep_count.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.wandb_tab, text="Sweep Method:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sweep_method"] = tk.StringVar(value="bayes")
        combo_sweep_method = ttk.Combobox(
            self.wandb_tab,
            textvariable=self.settings_vars["sweep_method"],
            values=["bayes", "grid", "random"],
            state="readonly",
        )
        combo_sweep_method.grid(row=5, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.wandb_tab, text="Sweep Metric (Classification):").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sweep_metric_cls"] = tk.StringVar(value="eval_loss")
        entry_sweep_metric_cls = ttk.Entry(self.wandb_tab, textvariable=self.settings_vars["sweep_metric_cls"], width=30)
        entry_sweep_metric_cls.grid(row=6, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.wandb_tab, text="Sweep Metric (Regression):").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sweep_metric_reg"] = tk.StringVar(value="eval_loss")
        entry_sweep_metric_reg = ttk.Entry(self.wandb_tab, textvariable=self.settings_vars["sweep_metric_reg"], width=30)
        entry_sweep_metric_reg.grid(row=7, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.wandb_tab, text="Sweep Goal:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sweep_goal"] = tk.StringVar(value="minimize")
        combo_sweep_goal = ttk.Combobox(
            self.wandb_tab,
            textvariable=self.settings_vars["sweep_goal"],
            values=["maximize", "minimize"],
            state="readonly",
        )
        combo_sweep_goal.grid(row=8, column=1, padx=10, pady=5, sticky="w")

        run_button = ttk.Button(self.wandb_tab, text="Save W&B Settings", command=self._save_wandb_settings)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_cloud_tab(self):
        ttk.Label(self.cloud_tab, text="Cloud URL:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_url"] = tk.StringVar(value="https://api.synthyra.com")
        entry_cloud_url = ttk.Entry(self.cloud_tab, textvariable=self.settings_vars["cloud_url"], width=40)
        entry_cloud_url.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.cloud_tab, text="GPU Type:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_gpu_type"] = tk.StringVar(value="A10")
        combo_cloud_gpu_type = ttk.Combobox(
            self.cloud_tab,
            textvariable=self.settings_vars["cloud_gpu_type"],
            values=["H200", "H100", "A100-80GB", "A100", "L40S", "A10", "L4", "T4"],
            state="readonly",
        )
        combo_cloud_gpu_type.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.cloud_tab, text="Runtime Timeout (seconds):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_timeout_seconds"] = tk.IntVar(value=86400)
        spin_cloud_timeout = ttk.Spinbox(self.cloud_tab, from_=60, to=604800, textvariable=self.settings_vars["cloud_timeout_seconds"])
        spin_cloud_timeout.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.cloud_tab, text="Poll Interval (seconds):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_poll_interval"] = tk.IntVar(value=5)
        spin_cloud_poll_interval = ttk.Spinbox(self.cloud_tab, from_=1, to=600, textvariable=self.settings_vars["cloud_poll_interval"])
        spin_cloud_poll_interval.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.cloud_tab, text="Current Job ID:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_job_id"] = tk.StringVar(value="")
        entry_cloud_job_id = ttk.Entry(self.cloud_tab, textvariable=self.settings_vars["cloud_job_id"], width=30)
        entry_cloud_job_id.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.cloud_tab, text="Artifact Output Directory:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_artifacts_dir"] = tk.StringVar(value="cloud_artifacts")
        entry_cloud_artifacts_dir = ttk.Entry(self.cloud_tab, textvariable=self.settings_vars["cloud_artifacts_dir"], width=30)
        entry_cloud_artifacts_dir.grid(row=5, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.cloud_tab, text="Auto Poll:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["cloud_auto_poll"] = tk.BooleanVar(value=True)
        check_cloud_auto_poll = ttk.Checkbutton(self.cloud_tab, variable=self.settings_vars["cloud_auto_poll"])
        check_cloud_auto_poll.grid(row=6, column=1, padx=10, pady=5, sticky="w")

        submit_button = ttk.Button(self.cloud_tab, text="Submit Remote Run", command=self._cloud_submit_run)
        submit_button.grid(row=7, column=0, padx=10, pady=10, sticky="w")

        poll_button = ttk.Button(self.cloud_tab, text="Poll Status", command=self._cloud_poll_status)
        poll_button.grid(row=7, column=1, padx=10, pady=10, sticky="w")

        cancel_button = ttk.Button(self.cloud_tab, text="Cancel Run", command=self._cloud_cancel_run)
        cancel_button.grid(row=8, column=0, padx=10, pady=5, sticky="w")

        start_auto_poll_button = ttk.Button(self.cloud_tab, text="Start Auto Poll", command=self._cloud_start_auto_poll)
        start_auto_poll_button.grid(row=8, column=1, padx=10, pady=5, sticky="w")

        stop_auto_poll_button = ttk.Button(self.cloud_tab, text="Stop Auto Poll", command=self._cloud_stop_auto_poll)
        stop_auto_poll_button.grid(row=9, column=0, padx=10, pady=5, sticky="w")

        fetch_button = ttk.Button(self.cloud_tab, text="Fetch Results", command=self._cloud_fetch_artifacts)
        fetch_button.grid(row=9, column=1, padx=10, pady=5, sticky="w")

    def build_proteingym_tab(self):
        # ProteinGym Checkbox
        ttk.Label(self.proteingym_tab, text="Run ProteinGym:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["proteingym"] = tk.BooleanVar(value=False)
        check_proteingym = ttk.Checkbutton(self.proteingym_tab, variable=self.settings_vars["proteingym"])
        check_proteingym.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.proteingym_tab, 0, 2, "Enable ProteinGym zero-shot evaluation.")

        # DMS IDs
        ttk.Label(self.proteingym_tab, text="DMS IDs (space-separated or 'all'):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["dms_ids"] = tk.StringVar(value="all")
        entry_dms_ids = ttk.Entry(self.proteingym_tab, textvariable=self.settings_vars["dms_ids"], width=30)
        entry_dms_ids.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.proteingym_tab, 1, 2, "List of DMS IDs to evaluate, or 'all'.")

        # Mode
        ttk.Label(self.proteingym_tab, text="Mode:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["mode"] = tk.StringVar(value="benchmark")
        combo_mode = ttk.Combobox(
            self.proteingym_tab,
            textvariable=self.settings_vars["mode"],
            values=["benchmark", "indels", "multiples", "singles"]
        )
        combo_mode.grid(row=2, column=1, padx=10, pady=5)
        self.add_help_button(self.proteingym_tab, 2, 2, "ProteinGym zero-shot mode.")

        # Scoring Method
        ttk.Label(self.proteingym_tab, text="Scoring Method:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scoring_method"] = tk.StringVar(value="masked_marginal")
        combo_scoring_method = ttk.Combobox(
            self.proteingym_tab,
            textvariable=self.settings_vars["scoring_method"],
            values=["masked_marginal", "mutant_marginal", "wildtype_marginal", "pll", "global_log_prob"]
        )
        combo_scoring_method.grid(row=3, column=1, padx=10, pady=5)
        self.add_help_button(self.proteingym_tab, 3, 2, "Scoring method for zero-shot evaluation.")

        # Scoring Window
        ttk.Label(self.proteingym_tab, text="Scoring Window:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scoring_window"] = tk.StringVar(value="optimal")
        combo_scoring_window = ttk.Combobox(
            self.proteingym_tab,
            textvariable=self.settings_vars["scoring_window"],
            values=["optimal", "sliding"]
        )
        combo_scoring_window.grid(row=4, column=1, padx=10, pady=5)
        self.add_help_button(self.proteingym_tab, 4, 2, "Windowing strategy for scoring.")

        # Batch Size
        ttk.Label(self.proteingym_tab, text="Batch Size:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["pg_batch_size"] = tk.IntVar(value=32)
        spin_pg_batch_size = ttk.Spinbox(self.proteingym_tab, from_=1, to=1024, textvariable=self.settings_vars["pg_batch_size"])
        spin_pg_batch_size.grid(row=5, column=1, padx=10, pady=5)
        self.add_help_button(self.proteingym_tab, 5, 2, "Batch size for ProteinGym scoring.")

        # Compare Scoring Methods
        ttk.Label(self.proteingym_tab, text="Compare Scoring Methods:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["compare_scoring_methods"] = tk.BooleanVar(value=False)
        check_compare = ttk.Checkbutton(self.proteingym_tab, variable=self.settings_vars["compare_scoring_methods"])
        check_compare.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.proteingym_tab, 6, 2, "Compare different scoring methods across models and DMS assays.")

        ttk.Label(self.proteingym_tab, text="Score Only:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["score_only"] = tk.BooleanVar(value=False)
        check_score_only = ttk.Checkbutton(self.proteingym_tab, variable=self.settings_vars["score_only"])
        check_score_only.grid(row=7, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(self.proteingym_tab, 7, 2, "Skip scoring and run benchmark report generation on existing results.")

        run_button = ttk.Button(self.proteingym_tab, text="Run ProteinGym", command=self._run_proteingym)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_scikit_tab(self):
        # Create a frame for scikit settings
        scikit_frame = ttk.LabelFrame(self.scikit_tab, text="Scikit-Learn Settings")
        scikit_frame.pack(fill="x", padx=10, pady=5)
        
        # Use Scikit
        ttk.Label(scikit_frame, text="Use Scikit:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["use_scikit"] = tk.BooleanVar(value=False)
        check_scikit = ttk.Checkbutton(scikit_frame, variable=self.settings_vars["use_scikit"])
        check_scikit.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(scikit_frame, 0, 2, "Whether to use scikit-learn models instead of neural networks.")

        # Scikit Iterations
        ttk.Label(scikit_frame, text="Scikit Iterations:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_n_iter"] = tk.IntVar(value=10)
        spin_scikit_n_iter = ttk.Spinbox(scikit_frame, from_=1, to=1000, textvariable=self.settings_vars["scikit_n_iter"])
        spin_scikit_n_iter.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(scikit_frame, 1, 2, "Number of iterations for iterative scikit-learn models.")

        # Scikit CV Folds
        ttk.Label(scikit_frame, text="Scikit CV Folds:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_cv"] = tk.IntVar(value=3)
        spin_scikit_cv = ttk.Spinbox(scikit_frame, from_=1, to=10, textvariable=self.settings_vars["scikit_cv"])
        spin_scikit_cv.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(scikit_frame, 2, 2, "Number of cross-validation folds for model evaluation.")

        # Scikit Random State
        ttk.Label(scikit_frame, text="Scikit Random State:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_random_state"] = tk.IntVar(value=42)
        spin_scikit_rand = ttk.Spinbox(scikit_frame, from_=0, to=10000, textvariable=self.settings_vars["scikit_random_state"])
        spin_scikit_rand.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(scikit_frame, 3, 2, "Random seed for scikit-learn models to ensure reproducibility.")

        # Scikit Model Name
        ttk.Label(scikit_frame, text="Scikit Model Name (optional):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_model_name"] = tk.StringVar(value="")
        entry_scikit_name = ttk.Entry(scikit_frame, textvariable=self.settings_vars["scikit_model_name"], width=30)
        entry_scikit_name.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(scikit_frame, 4, 2, "Optional name for the scikit-learn model. Leave blank to use default.")
        
        # Number of Jobs/Processors
        ttk.Label(scikit_frame, text="Number of Jobs:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["n_jobs"] = tk.IntVar(value=1)
        spin_n_jobs = ttk.Spinbox(scikit_frame, from_=1, to=32, textvariable=self.settings_vars["n_jobs"])
        spin_n_jobs.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(scikit_frame, 5, 2, "Number of CPU cores to use for parallel processing. Use -1 for all cores.")

        run_button = ttk.Button(self.scikit_tab, text="Run Scikit Models", command=self._run_scikit)
        run_button.pack(pady=(20, 10))

    def build_replay_tab(self):
        # Create a frame for replay settings
        replay_frame = ttk.LabelFrame(self.replay_tab, text="Log Replay Settings")
        replay_frame.pack(fill="x", padx=10, pady=5)

        # Replay log path
        ttk.Label(replay_frame, text="Replay Log Path:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["replay_path"] = tk.StringVar(value="")
        entry_replay = ttk.Entry(replay_frame, textvariable=self.settings_vars["replay_path"], width=40)
        entry_replay.grid(row=0, column=1, padx=10, pady=5)
        self.add_help_button(replay_frame, 0, 2, "Path to the log file to replay. Use Browse button to select a file.")

        # Browse button for selecting log file
        browse_button = ttk.Button(replay_frame, text="Browse", command=self._browse_replay_log)
        browse_button.grid(row=0, column=2, padx=5, pady=5)

        # Start replay button
        replay_button = ttk.Button(replay_frame, text="Start Replay", command=self._start_replay)
        replay_button.grid(row=1, column=0, columnspan=3, pady=20)

    def build_viz_tab(self):
        # Create a frame for visualization settings
        viz_frame = ttk.LabelFrame(self.viz_tab, text="Visualization Settings")
        viz_frame.pack(fill="x", padx=10, pady=5)

        # Result ID entry
        ttk.Label(viz_frame, text="Result ID:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["result_id"] = tk.StringVar(value="")
        entry_result_id = ttk.Entry(viz_frame, textvariable=self.settings_vars["result_id"], width=30)
        entry_result_id.grid(row=0, column=1, padx=10, pady=5)
        self.add_help_button(viz_frame, 0, 2, "ID of the result to visualize. Will look for results/{result_id}.tsv")

        # Results file path
        ttk.Label(viz_frame, text="Results File:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["results_file"] = tk.StringVar(value="")
        entry_results_file = ttk.Entry(viz_frame, textvariable=self.settings_vars["results_file"], width=30)
        entry_results_file.grid(row=1, column=1, padx=10, pady=5)
        
        # Browse button for selecting results file directly
        browse_button = ttk.Button(viz_frame, text="Browse", command=self._browse_results_file)
        browse_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Use current run checkbox
        ttk.Label(viz_frame, text="Use Current Run:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["use_current_run"] = tk.BooleanVar(value=True)
        check_current_run = ttk.Checkbutton(viz_frame, variable=self.settings_vars["use_current_run"])
        check_current_run.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.add_help_button(viz_frame, 2, 2, "Use results from the current run.")

        # Output directory for plots
        ttk.Label(viz_frame, text="Output Directory:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["viz_output_dir"] = tk.StringVar(value="plots")
        entry_output_dir = ttk.Entry(viz_frame, textvariable=self.settings_vars["viz_output_dir"], width=30)
        entry_output_dir.grid(row=3, column=1, padx=10, pady=5)
        self.add_help_button(viz_frame, 3, 2, "Directory where plots will be saved.")


        # Generate plots button
        generate_button = ttk.Button(viz_frame, text="Generate Plots", command=self._generate_plots)
        generate_button.grid(row=99, column=0, columnspan=3, pady=20)

    def add_help_button(self, parent, row, column, help_text):
        """Add a small help button that displays information when clicked"""
        help_button = ttk.Button(parent, text="?", width=2, 
                                command=lambda: messagebox.showinfo("Help", help_text))
        help_button.grid(row=row, column=column, padx=(0,5), pady=5)
        return help_button

    def _selected_model_dtype(self):
        dtype_name = self.settings_vars["model_dtype"].get()
        assert dtype_name in self.dtype_map, f"Unsupported model dtype: {dtype_name}"
        return self.dtype_map[dtype_name]

    def _selected_embed_dtype(self):
        dtype_name = self.settings_vars["embed_dtype"].get()
        assert dtype_name in self.dtype_map, f"Unsupported embedding dtype: {dtype_name}"
        return self.dtype_map[dtype_name]

    def _browse_data_dir(self):
        data_dir = filedialog.askdirectory(title="Select Data Directory")
        if not data_dir:
            return
        existing = self.settings_vars["data_dirs"].get().strip()
        if not existing:
            self.settings_vars["data_dirs"].set(data_dir)
            return
        existing_parts = [path.strip() for path in existing.split(",") if path.strip()]
        if data_dir not in existing_parts:
            existing_parts.append(data_dir)
        self.settings_vars["data_dirs"].set(", ".join(existing_parts))

    def _browse_sweep_config(self):
        filename = filedialog.askopenfilename(
            title="Select W&B Sweep Config",
            filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*.*")),
        )
        if filename:
            self.settings_vars["sweep_config_path"].set(filename)

    def _save_wandb_settings(self):
        print_message("Saving W&B sweep settings...")
        self.full_args.use_wandb_hyperopt = self.settings_vars["use_wandb_hyperopt"].get()
        self.full_args.wandb_project = self.settings_vars["wandb_project"].get().strip() or "Protify"
        wandb_entity = self.settings_vars["wandb_entity"].get().strip()
        self.full_args.wandb_entity = wandb_entity if wandb_entity else None
        self.full_args.sweep_config_path = self.settings_vars["sweep_config_path"].get().strip() or "yamls/sweep.yaml"
        self.full_args.sweep_count = self.settings_vars["sweep_count"].get()
        self.full_args.sweep_method = self.settings_vars["sweep_method"].get()
        self.full_args.sweep_metric_cls = self.settings_vars["sweep_metric_cls"].get().strip() or "eval_loss"
        self.full_args.sweep_metric_reg = self.settings_vars["sweep_metric_reg"].get().strip() or "eval_loss"
        self.full_args.sweep_goal = self.settings_vars["sweep_goal"].get()

        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        if "log_file" in self.__dict__:
            self._write_args()
        print_message("W&B sweep settings saved")
        print_done()

    def _resolve_repo_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _get_cloud_backend(self):
        """Get or create a CloudBackend from the GUI settings."""
        api_key = self.settings_vars["cloud_api_key"].get().strip()
        base_url = self.settings_vars["cloud_url"].get().strip() or None
        backend = get_or_create_cloud_backend(api_key=api_key, base_url=base_url)
        assert backend is not None, "No cloud API key provided. Enter your API key in the Info tab."
        return backend

    def _collect_cloud_run_config(self):
        selected_model_indices = self.model_listbox.curselection()
        selected_models = [self.model_listbox.get(i) for i in selected_model_indices]
        if len(selected_models) == 0:
            selected_models = standard_models

        selected_dataset_indices = self.data_listbox.curselection()
        selected_datasets = [self.data_listbox.get(i) for i in selected_dataset_indices]
        data_dirs_str = self.settings_vars["data_dirs"].get().strip()
        data_dirs = [path.strip() for path in data_dirs_str.split(",") if path.strip()]

        run_proteingym = self.settings_vars["proteingym"].get()
        if (len(selected_datasets) == 0) and (len(data_dirs) == 0) and (not run_proteingym):
            selected_datasets = standard_data_benchmark

        col_names = [name.strip() for name in self.settings_vars["col_names"].get().split(",") if name.strip()]
        multi_column_raw = self.settings_vars["multi_column"].get().strip()
        if multi_column_raw:
            multi_column = multi_column_raw.split()
        else:
            multi_column = None

        embedding_pooling = [item.strip() for item in self.settings_vars["embedding_pooling_types"].get().split(",") if item.strip()]
        probe_pooling = [item.strip() for item in self.settings_vars["probe_pooling_types"].get().split(",") if item.strip()]

        dms_ids_raw = self.settings_vars["dms_ids"].get().strip()
        if dms_ids_raw.lower() == "all":
            dms_ids = ["all"]
        else:
            dms_ids = [item.strip() for item in dms_ids_raw.split() if item.strip()]

        wandb_entity = self.settings_vars["wandb_entity"].get().strip()
        if wandb_entity == "":
            wandb_entity = None

        scikit_model_name = self.settings_vars["scikit_model_name"].get().strip()
        if scikit_model_name == "":
            scikit_model_name = None

        hf_home = self.settings_vars["hf_home"].get().strip()
        if hf_home == "":
            hf_home = None

        config = {
            "hf_username": self.settings_vars["huggingface_username"].get().strip() or "Synthyra",
            "hf_token": self.settings_vars["huggingface_token"].get().strip() or None,
            "wandb_api_key": self.settings_vars["wandb_api_key"].get().strip() or None,
            "synthyra_api_key": self.settings_vars["synthyra_api_key"].get().strip() or None,
            "hf_home": hf_home,
            "log_dir": self.settings_vars["log_dir"].get().strip() or "logs",
            "results_dir": self.settings_vars["results_dir"].get().strip() or "results",
            "model_save_dir": self.settings_vars["model_save_dir"].get().strip() or "weights",
            "embedding_save_dir": self.settings_vars["embedding_save_dir"].get().strip() or "embeddings",
            "download_dir": self.settings_vars["download_dir"].get().strip() or "Synthyra/vector_embeddings",
            "plots_dir": self.settings_vars["plots_dir"].get().strip() or "plots",
            "replay_path": None,
            "data_names": selected_datasets,
            "data_dirs": data_dirs,
            "delimiter": self.settings_vars["delimiter"].get(),
            "col_names": col_names,
            "max_length": self.settings_vars["max_length"].get(),
            "trim": self.settings_vars["trim"].get(),
            "multi_column": multi_column,
            "aa_to_dna": self.settings_vars["aa_to_dna"].get(),
            "aa_to_rna": self.settings_vars["aa_to_rna"].get(),
            "dna_to_aa": self.settings_vars["dna_to_aa"].get(),
            "rna_to_aa": self.settings_vars["rna_to_aa"].get(),
            "codon_to_aa": self.settings_vars["codon_to_aa"].get(),
            "aa_to_codon": self.settings_vars["aa_to_codon"].get(),
            "random_pair_flipping": self.settings_vars["random_pair_flipping"].get(),
            "model_names": selected_models,
            "model_paths": None,
            "model_types": None,
            "model_dtype": self.settings_vars["model_dtype"].get(),
            "use_xformers": self.settings_vars["use_xformers"].get(),
            "embedding_batch_size": self.settings_vars["batch_size"].get(),
            "embedding_num_workers": self.settings_vars["num_workers"].get(),
            "num_workers": self.settings_vars["num_workers"].get(),
            "download_embeddings": self.settings_vars["download_embeddings"].get(),
            "matrix_embed": self.settings_vars["matrix_embed"].get(),
            "embedding_pooling_types": embedding_pooling,
            "embedding_hidden_state_index": self.settings_vars["embedding_hidden_state_index"].get(),
            "save_embeddings": True,
            "embed_dtype": self.settings_vars["embed_dtype"].get(),
            "sql": self.settings_vars["sql"].get(),
            "probe_type": self.settings_vars["probe_type"].get(),
            "probe_n_jobs": -1,
            "sae_layer": None,
            "sae_k": None,
            "sae_codebook_dim": None,
            "tokenwise": self.settings_vars["tokenwise"].get(),
            "hidden_size": self.settings_vars["hidden_size"].get(),

            "dropout": self.settings_vars["dropout"].get(),
            "n_layers": self.settings_vars["n_layers"].get(),
            "pre_ln": self.settings_vars["pre_ln"].get(),
            "classifier_size": self.settings_vars["classifier_size"].get(),
            "transformer_dropout": self.settings_vars["transformer_dropout"].get(),
            "classifier_dropout": self.settings_vars["classifier_dropout"].get(),
            "n_heads": self.settings_vars["n_heads"].get(),
            "rotary": self.settings_vars["rotary"].get(),
            "attention_backend": self.settings_vars["attention_backend"].get(),
            "probe_pooling_types": probe_pooling,
            "use_bias": self.settings_vars["use_bias"].get(),
            "save_model": self.settings_vars["save_model"].get(),
            "push_raw_probe": self.settings_vars["push_raw_probe"].get(),
            "production_model": self.settings_vars["production_model"].get(),
            "lora": self.settings_vars["lora"].get(),
            "lora_r": self.settings_vars["lora_r"].get(),
            "lora_alpha": self.settings_vars["lora_alpha"].get(),
            "lora_dropout": self.settings_vars["lora_dropout"].get(),
            "sim_type": self.settings_vars["sim_type"].get(),
            "add_token_ids": self.settings_vars["add_token_ids"].get(),
            "num_epochs": self.settings_vars["num_epochs"].get(),
            "probe_batch_size": self.settings_vars["probe_batch_size"].get(),
            "base_batch_size": self.settings_vars["base_batch_size"].get(),
            "probe_grad_accum": self.settings_vars["probe_grad_accum"].get(),
            "base_grad_accum": self.settings_vars["base_grad_accum"].get(),
            "lr": self.settings_vars["lr"].get(),
            "probe_lr": float(self.settings_vars["probe_lr"].get().strip()) if self.settings_vars["probe_lr"].get().strip() else None,
            "base_lr": float(self.settings_vars["base_lr"].get().strip()) if self.settings_vars["base_lr"].get().strip() else None,
            "lr_scheduler": self.settings_vars["lr_scheduler"].get(),
            "optimizer": self.settings_vars["optimizer"].get(),
            "weight_decay": self.settings_vars["weight_decay"].get(),
            "patience": self.settings_vars["patience"].get(),
            "seed": self.settings_vars["seed"].get(),
            "deterministic": self.settings_vars["deterministic"].get(),
            "full_finetuning": self.settings_vars["full_finetuning"].get(),
            "hybrid_probe": self.settings_vars["hybrid_probe"].get(),
            "num_runs": self.settings_vars["num_runs"].get(),
            "read_scaler": self.settings_vars["read_scaler"].get(),
            "dms_ids": dms_ids,
            "proteingym": run_proteingym,
            "mode": self.settings_vars["mode"].get(),
            "scoring_method": self.settings_vars["scoring_method"].get(),
            "scoring_window": self.settings_vars["scoring_window"].get(),
            "pg_batch_size": self.settings_vars["pg_batch_size"].get(),
            "compare_scoring_methods": self.settings_vars["compare_scoring_methods"].get(),
            "score_only": self.settings_vars["score_only"].get(),
            "use_wandb_hyperopt": self.settings_vars["use_wandb_hyperopt"].get(),
            "wandb_project": self.settings_vars["wandb_project"].get().strip() or "Protify",
            "wandb_entity": wandb_entity,
            "sweep_config_path": self.settings_vars["sweep_config_path"].get().strip() or "yamls/sweep.yaml",
            "sweep_count": self.settings_vars["sweep_count"].get(),
            "sweep_method": self.settings_vars["sweep_method"].get(),
            "sweep_metric_cls": self.settings_vars["sweep_metric_cls"].get().strip() or "eval_loss",
            "sweep_metric_reg": self.settings_vars["sweep_metric_reg"].get().strip() or "eval_loss",
            "sweep_goal": self.settings_vars["sweep_goal"].get(),
            "use_scikit": self.settings_vars["use_scikit"].get(),
            "scikit_n_iter": self.settings_vars["scikit_n_iter"].get(),
            "scikit_cv": self.settings_vars["scikit_cv"].get(),
            "scikit_random_state": self.settings_vars["scikit_random_state"].get(),
            "scikit_model_name": scikit_model_name,
            "n_jobs": self.settings_vars["n_jobs"].get(),
        }
        return config

    def _cloud_submit_run(self):
        print_message("Submitting remote cloud run...")

        def background_submit():
            backend = self._get_cloud_backend()
            config = self._collect_cloud_run_config()

            gpu_type = self.settings_vars["cloud_gpu_type"].get()
            timeout_seconds = self.settings_vars["cloud_timeout_seconds"].get()

            result = backend.submit_job(
                config=config,
                gpu_type=gpu_type,
                timeout_seconds=timeout_seconds,
            )
            assert isinstance(result, dict), "Cloud submit returned a non-dict response."
            assert "job_id" in result, "Cloud submit response missing job_id."

            job_id = result["job_id"]
            self.settings_vars["cloud_job_id"].set(job_id)

            print_message(f"Cloud job submitted.\nJob ID: {job_id}")
            if self.settings_vars["cloud_auto_poll"].get():
                self.cloud_polling_active = True
                self.master.after(0, self._cloud_auto_poll_loop)
            print_done()

        self.run_in_background(background_submit)

    def _cloud_start_auto_poll(self):
        if self.cloud_polling_active:
            print_message("Auto polling is already active.")
            return
        self.cloud_polling_active = True
        print_message("Started cloud auto polling.")
        self._cloud_auto_poll_loop()

    def _cloud_stop_auto_poll(self):
        self.cloud_polling_active = False
        print_message("Stopped cloud auto polling.")

    def _cloud_auto_poll_loop(self):
        if not self.cloud_polling_active:
            return
        if not self.settings_vars["cloud_auto_poll"].get():
            self.cloud_polling_active = False
            return

        job_id = self.settings_vars["cloud_job_id"].get().strip()
        if not job_id:
            self.cloud_polling_active = False
            return

        self._cloud_poll_status()
        poll_interval = self.settings_vars["cloud_poll_interval"].get()
        self.master.after(max(1, poll_interval) * 1000, self._cloud_auto_poll_loop)

    def _cloud_poll_status(self):
        job_id = self.settings_vars["cloud_job_id"].get().strip()
        if not job_id:
            print_message("No cloud job ID set. Submit a remote run first.")
            return
        print_message(f"Polling cloud status for job {job_id}...")

        def background_poll():
            backend = self._get_cloud_backend()
            status_payload = backend.get_job_status(job_id=job_id)

            assert isinstance(status_payload, dict), "get_job_status returned a non-dict response."

            status_value = status_payload.get("status", "UNKNOWN")
            phase_value = status_payload.get("phase", "N/A")
            error_value = status_payload.get("error")
            print_message(
                f"Cloud Status: {status_value}\n"
                f"Phase: {phase_value}"
            )
            if error_value:
                print_message(f"Error: {error_value}")

            try:
                log_payload = backend.get_job_logs(job_id=job_id, offset=0, max_chars=5000)
                content = log_payload.get("content", "")
                if content:
                    tail = content[-3000:] if len(content) > 3000 else content
                    print_message(f"Latest Logs:\n{tail}")
            except Exception:
                pass

            terminal_states = {"Complete", "Failed", "Cancelled", "SUCCESS", "FAILED", "TERMINATED", "TIMEOUT"}
            if status_value in terminal_states:
                self.cloud_polling_active = False
            print_done()

        self.run_in_background(background_poll)

    def _cloud_cancel_run(self):
        job_id = self.settings_vars["cloud_job_id"].get().strip()
        if not job_id:
            print_message("No cloud job ID set. Submit a run first.")
            return
        print_message(f"Cancelling cloud job {job_id}...")
        self.cloud_polling_active = False

        def background_cancel():
            backend = self._get_cloud_backend()
            result = backend.cancel_job(job_id=job_id)
            print_message(f"Cancel result: {result}")
            print_done()

        self.run_in_background(background_cancel)

    def _cloud_fetch_artifacts(self):
        job_id = self.settings_vars["cloud_job_id"].get().strip()
        if not job_id:
            print_message("No cloud job ID set. Submit a run first.")
            return
        print_message(f"Fetching cloud artifacts for job {job_id}...")

        def background_fetch():
            backend = self._get_cloud_backend()
            result_payload = backend.get_results(job_id=job_id)
            assert isinstance(result_payload, dict), "get_results returned a non-dict response."

            output_dir_raw = self.settings_vars["cloud_artifacts_dir"].get().strip() or "cloud_artifacts"
            home_dir = self.settings_vars["home_dir"].get().strip() or os.getcwd()
            if os.path.isabs(output_dir_raw):
                output_dir = output_dir_raw
            else:
                output_dir = os.path.abspath(os.path.join(home_dir, output_dir_raw))
            job_dir = os.path.join(output_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)

            file_count = 0

            results_tsv = result_payload.get("results_tsv")
            if results_tsv:
                tsv_path = os.path.join(job_dir, "results.tsv")
                with open(tsv_path, "w", encoding="utf-8") as f:
                    f.write(results_tsv)
                file_count += 1

            images = result_payload.get("images", [])
            if images:
                plots_dir = os.path.join(job_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                for img in images:
                    filename = img.get("filename", "plot.png")
                    data = img.get("data", "")
                    if data:
                        img_path = os.path.join(plots_dir, filename)
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(data))
                        file_count += 1

            metadata_path = os.path.join(job_dir, "cloud_fetch_summary.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, indent=2, default=str)

            hub_url = result_payload.get("hub_url")
            hub_msg = f"\nModel pushed to: {hub_url}" if hub_url else ""
            print_message(
                f"Saved cloud artifacts to {job_dir}\n"
                f"Files saved: {file_count}{hub_msg}"
            )
            print_done()

        self.run_in_background(background_fetch)

    def _session_start(self):
        print_message("Starting Protify session...")
        # Update session variables
        hf_token = self.settings_vars["huggingface_token"].get()
        synthyra_api_key = self.settings_vars["synthyra_api_key"].get()
        wandb_api_key = self.settings_vars["wandb_api_key"].get()
        def background_login():

            if hf_token:
                from huggingface_hub import login
                login(hf_token)
                print_message('Logged in to Hugging Face')
            if wandb_api_key:
                try:
                    import wandb
                    wandb.login(key=wandb_api_key)
                    print_message('Logged in to Weights & Biases')
                except Exception as error:
                    print_message(f'W&B login failed: {error}')
            if synthyra_api_key:
                print_message('Synthyra API key provided')
            
            self.full_args.hf_username = self.settings_vars["huggingface_username"].get()
            self.full_args.hf_token = hf_token
            self.full_args.synthyra_api_key = synthyra_api_key
            self.full_args.wandb_api_key = wandb_api_key
            self.full_args.home_dir = self.settings_vars["home_dir"].get()
            self.full_args.model_dtype = self._selected_model_dtype()
            self.full_args.use_xformers = self.settings_vars["use_xformers"].get()
            self.full_args.num_runs = self.settings_vars["num_runs"].get()
            self.full_args.use_wandb_hyperopt = self.settings_vars["use_wandb_hyperopt"].get()
            self.full_args.wandb_project = self.settings_vars["wandb_project"].get().strip() or "Protify"
            wandb_entity = self.settings_vars["wandb_entity"].get().strip()
            self.full_args.wandb_entity = wandb_entity if wandb_entity else None
            self.full_args.sweep_config_path = self.settings_vars["sweep_config_path"].get().strip() or "yamls/sweep.yaml"
            self.full_args.sweep_count = self.settings_vars["sweep_count"].get()
            self.full_args.sweep_method = self.settings_vars["sweep_method"].get()
            self.full_args.sweep_metric_cls = self.settings_vars["sweep_metric_cls"].get().strip() or "eval_loss"
            self.full_args.sweep_metric_reg = self.settings_vars["sweep_metric_reg"].get().strip() or "eval_loss"
            self.full_args.sweep_goal = self.settings_vars["sweep_goal"].get()
            self.full_args.score_only = self.settings_vars["score_only"].get()
            self.full_args.aa_to_dna = self.settings_vars["aa_to_dna"].get()
            self.full_args.aa_to_rna = self.settings_vars["aa_to_rna"].get()
            self.full_args.dna_to_aa = self.settings_vars["dna_to_aa"].get()
            self.full_args.rna_to_aa = self.settings_vars["rna_to_aa"].get()
            self.full_args.codon_to_aa = self.settings_vars["codon_to_aa"].get()
            self.full_args.aa_to_codon = self.settings_vars["aa_to_codon"].get()
            self.full_args.random_pair_flipping = self.settings_vars["random_pair_flipping"].get()
            self.full_args.data_dirs = []

            if self.full_args.use_xformers:
                os.environ["_USE_XFORMERS"] = "1"
            elif "_USE_XFORMERS" in os.environ:
                del os.environ["_USE_XFORMERS"]
            
            # Handle hf_home - convert empty string to None
            hf_home_value = self.settings_vars["hf_home"].get().strip()
            self.full_args.hf_home = hf_home_value if hf_home_value else None

            def _make_true_dir(path):
                true_path = os.path.join(self.full_args.home_dir, path)
                os.makedirs(true_path, exist_ok=True)
                return true_path

            self.full_args.log_dir = _make_true_dir(self.settings_vars["log_dir"].get())
            self.full_args.results_dir = _make_true_dir(self.settings_vars["results_dir"].get())
            self.full_args.model_save_dir = _make_true_dir(self.settings_vars["model_save_dir"].get())
            self.full_args.plots_dir = _make_true_dir(self.settings_vars["plots_dir"].get())
            self.full_args.embedding_save_dir = _make_true_dir(self.settings_vars["embedding_save_dir"].get())
            self.full_args.download_dir = _make_true_dir(self.settings_vars["download_dir"].get())

            self.full_args.replay_path = None
            self.logger_args = SimpleNamespace(**self.full_args.__dict__)
            self.start_log_gui()

            print_message(f"Session and logging started for id {self.random_id}")
            print_done()
        
        self.run_in_background(background_login)

    def _create_probe_args(self):
        print_message("Configuring probe...")
        
        # Gather settings from variables
        self.full_args.probe_type = self.settings_vars["probe_type"].get()
        self.full_args.probe_n_jobs = getattr(self.full_args, "probe_n_jobs", -1)
        self.full_args.tokenwise = self.settings_vars["tokenwise"].get()
        self.full_args.pre_ln = self.settings_vars["pre_ln"].get()
        self.full_args.n_layers = self.settings_vars["n_layers"].get()
        self.full_args.hidden_size = self.settings_vars["hidden_size"].get()
        self.full_args.dropout = self.settings_vars["dropout"].get()
        
        self.full_args.classifier_size = self.settings_vars["classifier_size"].get()
        self.full_args.classifier_dropout = self.settings_vars["classifier_dropout"].get()
        self.full_args.n_heads = self.settings_vars["n_heads"].get()
        self.full_args.rotary = self.settings_vars["rotary"].get()
        self.full_args.attention_backend = self.settings_vars["attention_backend"].get()
        
        pooling_str = self.settings_vars["probe_pooling_types"].get().strip()
        self.full_args.probe_pooling_types = [p.strip() for p in pooling_str.split(",") if p.strip()]
        
        self.full_args.transformer_dropout = self.settings_vars["transformer_dropout"].get()
        self.full_args.use_bias = self.settings_vars["use_bias"].get()
        self.full_args.add_token_ids = self.settings_vars["add_token_ids"].get()
        
        self.full_args.sim_type = self.settings_vars["sim_type"].get()
        self.full_args.save_model = self.settings_vars["save_model"].get()
        self.full_args.push_raw_probe = self.settings_vars["push_raw_probe"].get()
        self.full_args.production_model = self.settings_vars["production_model"].get()
        
        self.full_args.lora = self.settings_vars["lora"].get()
        self.full_args.lora_r = self.settings_vars["lora_r"].get()
        self.full_args.lora_alpha = self.settings_vars["lora_alpha"].get()
        self.full_args.lora_dropout = self.settings_vars["lora_dropout"].get()
        
        # Create ProbeArguments
        self.probe_args = ProbeArguments(**self.full_args.__dict__)
        
        # Update logger args
        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        self._write_args()
        
        print_message("Probe configuration saved")
        print_done()

    def _run_trainer(self):
        print_message("Starting training...")
        
        # Gather settings
        self.full_args.hybrid_probe = self.settings_vars["hybrid_probe"].get()
        self.full_args.full_finetuning = self.settings_vars["full_finetuning"].get()
        self.full_args.num_epochs = self.settings_vars["num_epochs"].get()
        self.full_args.probe_batch_size = self.settings_vars["probe_batch_size"].get()
        self.full_args.base_batch_size = self.settings_vars["base_batch_size"].get()
        self.full_args.probe_grad_accum = self.settings_vars["probe_grad_accum"].get()
        self.full_args.base_grad_accum = self.settings_vars["base_grad_accum"].get()
        self.full_args.lr = self.settings_vars["lr"].get()
        probe_lr = self.settings_vars["probe_lr"].get().strip()
        base_lr = self.settings_vars["base_lr"].get().strip()
        self.full_args.probe_lr = float(probe_lr) if probe_lr else None
        self.full_args.base_lr = float(base_lr) if base_lr else None
        self.full_args.lr_scheduler = self.settings_vars["lr_scheduler"].get()
        self.full_args.optimizer = self.settings_vars["optimizer"].get()
        self.full_args.weight_decay = self.settings_vars["weight_decay"].get()
        self.full_args.patience = self.settings_vars["patience"].get()
        self.full_args.seed = self.settings_vars["seed"].get()
        self.full_args.read_scaler = self.settings_vars["read_scaler"].get()
        self.full_args.deterministic = self.settings_vars["deterministic"].get()
        self.full_args.num_runs = self.settings_vars["num_runs"].get()
        self.full_args.use_wandb_hyperopt = self.settings_vars["use_wandb_hyperopt"].get()
        self.full_args.wandb_project = self.settings_vars["wandb_project"].get().strip() or "Protify"
        wandb_entity = self.settings_vars["wandb_entity"].get().strip()
        self.full_args.wandb_entity = wandb_entity if wandb_entity else None
        self.full_args.sweep_config_path = self.settings_vars["sweep_config_path"].get().strip() or "yamls/sweep.yaml"
        self.full_args.sweep_count = self.settings_vars["sweep_count"].get()
        self.full_args.sweep_method = self.settings_vars["sweep_method"].get()
        self.full_args.sweep_metric_cls = self.settings_vars["sweep_metric_cls"].get().strip() or "eval_loss"
        self.full_args.sweep_metric_reg = self.settings_vars["sweep_metric_reg"].get().strip() or "eval_loss"
        self.full_args.sweep_goal = self.settings_vars["sweep_goal"].get()
        self.full_args.use_xformers = self.settings_vars["use_xformers"].get()
        if self.full_args.use_xformers:
            os.environ["_USE_XFORMERS"] = "1"
        elif "_USE_XFORMERS" in os.environ:
            del os.environ["_USE_XFORMERS"]
        
        # Create TrainerArguments
        self.trainer_args = TrainerArguments(**self.full_args.__dict__)
        
        # Update logger args
        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        self._write_args()
        
        def background_train():
            if self.full_args.use_wandb_hyperopt:
                if not self.full_args.full_finetuning:
                    self.save_embeddings_to_disk()
                HyperoptModule.run_wandb_hyperopt(self)
            elif self.full_args.full_finetuning:
                self.run_full_finetuning()
            elif self.full_args.hybrid_probe:
                self.run_hybrid_probes()
            elif is_estimator_probe(self.full_args.probe_type):
                self.run_estimator_probes()
            else:
                self.run_nn_probes()
            print_done()
            
        self.run_in_background(background_train)

    def _run_proteingym(self):
        print_message("Starting ProteinGym...")
        
        # Gather settings
        self.full_args.proteingym = self.settings_vars["proteingym"].get()
        dms_ids_str = self.settings_vars["dms_ids"].get().strip()
        if dms_ids_str == "all":
            self.full_args.dms_ids = ["all"]
        else:
            self.full_args.dms_ids = dms_ids_str.split()
            
        self.full_args.mode = self.settings_vars["mode"].get()
        self.full_args.scoring_method = self.settings_vars["scoring_method"].get()
        self.full_args.scoring_window = self.settings_vars["scoring_window"].get()
        self.full_args.pg_batch_size = self.settings_vars["pg_batch_size"].get()
        self.full_args.compare_scoring_methods = self.settings_vars["compare_scoring_methods"].get()
        self.full_args.score_only = self.settings_vars["score_only"].get()
        
        # Update logger args
        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        self._write_args()
        
        def background_proteingym():
            if self.full_args.compare_scoring_methods and self.full_args.proteingym:
                print_message("Running scoring method comparison...")
                dms_ids = expand_dms_ids_all(self.full_args.dms_ids, mode=self.full_args.mode)
                model_names = self.full_args.model_names
                
                if len(model_names) == 0:
                    print_message("Error: No models selected for comparison")
                    return

                output_csv = os.path.join(self.full_args.results_dir, 'scoring_methods_comparison.csv')
                
                compare_scoring_methods(
                    model_names=model_names,
                    device=None,
                    methods=None,
                    dms_ids=dms_ids,
                    progress=True,
                    output_csv=output_csv
                )
                print_message(f"Scoring method comparison complete. Results saved to {output_csv}")
            
            elif self.full_args.proteingym:
                self.run_proteingym_zero_shot()
                
            print_done()
            
        self.run_in_background(background_proteingym)

    def _run_scikit(self):
        print_message("Starting Scikit-learn models...")
        assert "datasets" in self.__dict__, "Datasets are not loaded. Run the Data tab first."
        assert len(self.datasets) > 0, "No datasets are loaded. Run the Data tab first."
        assert "all_seqs" in self.__dict__, "Sequences are not loaded. Run the Data tab first."
        assert len(self.all_seqs) > 0, "No sequences are loaded. Run the Data tab first."
        
        # Gather model settings
        selected_indices = self.model_listbox.curselection()
        selected_models = [self.model_listbox.get(i) for i in selected_indices]
        if not selected_models:
            selected_models = standard_models
        self.full_args.model_names = selected_models
        self.full_args.model_paths = None
        self.full_args.model_types = None
        self.full_args.model_dtype = self._selected_model_dtype()
        self.full_args.use_xformers = self.settings_vars["use_xformers"].get()
        self.model_args = BaseModelArguments(**self.full_args.__dict__)

        # Gather embedding settings
        pooling_str = self.settings_vars["embedding_pooling_types"].get().strip()
        pooling_list = [p.strip() for p in pooling_str.split(",") if p.strip()]
        dtype_val = self._selected_embed_dtype()

        self.full_args.embedding_batch_size = self.settings_vars["batch_size"].get()
        self.full_args.embedding_num_workers = self.settings_vars["num_workers"].get()
        self.full_args.download_embeddings = self.settings_vars["download_embeddings"].get()
        self.full_args.matrix_embed = self.settings_vars["matrix_embed"].get()
        self.full_args.embedding_pooling_types = pooling_list
        self.full_args.embedding_hidden_state_index = self.settings_vars["embedding_hidden_state_index"].get()
        self.full_args.save_embeddings = True
        self.full_args.embed_dtype = dtype_val
        self.full_args.sql = self.settings_vars["sql"].get()
        self._sql = self.full_args.sql
        self._full = self.full_args.matrix_embed
        self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)

        # Gather scikit settings
        self.full_args.use_scikit = self.settings_vars["use_scikit"].get()
        self.full_args.scikit_n_iter = self.settings_vars["scikit_n_iter"].get()
        self.full_args.scikit_cv = self.settings_vars["scikit_cv"].get()
        self.full_args.scikit_random_state = self.settings_vars["scikit_random_state"].get()
        scikit_model_name = self.settings_vars["scikit_model_name"].get().strip()
        if scikit_model_name:
            self.full_args.scikit_model_name = scikit_model_name
        else:
            self.full_args.scikit_model_name = None
        self.full_args.n_jobs = self.settings_vars["n_jobs"].get()
        self.full_args.n_iter = self.full_args.scikit_n_iter
        self.full_args.cv = self.full_args.scikit_cv
        self.full_args.random_state = self.full_args.scikit_random_state
        self.full_args.model_name = self.full_args.scikit_model_name
        self.scikit_args = self._build_scikit_args()
        
        # Update logger args
        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        self._write_args()
        
        def background_scikit():
            self.save_embeddings_to_disk()
            self.run_scikit_scheme()
            print_done()
            
        self.run_in_background(background_scikit)

    def _select_models(self):
        print_message("Selecting models...")
        # Gather selected model names
        selected_indices = self.model_listbox.curselection()
        selected_models = [self.model_listbox.get(i) for i in selected_indices]

        # If no selection, default to the entire standard_benchmark
        if not selected_models:
            selected_models = standard_models

        # Update full_args with model settings
        self.full_args.model_names = selected_models
        self.full_args.model_paths = None
        self.full_args.model_types = None
        self.full_args.model_dtype = self._selected_model_dtype()
        self.full_args.use_xformers = self.settings_vars["use_xformers"].get()
        if self.full_args.use_xformers:
            os.environ["_USE_XFORMERS"] = "1"
        elif "_USE_XFORMERS" in os.environ:
            del os.environ["_USE_XFORMERS"]
        print_message(self.full_args.model_names)
        # Create model args from full args
        self.model_args = BaseModelArguments(**self.full_args.__dict__)

        print("Model Args:")
        for k, v in self.model_args.__dict__.items():
            if k != 'model_names':
                print(f"{k}:\n{v}")
        print("=========================\n")
        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        self._write_args()
        print_done()

    def _get_data(self):
        print_message("=== Getting Data ===")
        print_message("Loading and preparing datasets...")
        
        # Gather settings
        selected_indices = self.data_listbox.curselection()
        selected_datasets = [self.data_listbox.get(i) for i in selected_indices]
        data_dirs_str = self.settings_vars["data_dirs"].get().strip()
        data_dirs = [path.strip() for path in data_dirs_str.split(",") if path.strip()]
        
        if (not selected_datasets) and (len(data_dirs) == 0):
            selected_datasets = standard_data_benchmark
            
        def background_get_data():
            # Update full_args with data settings
            self.full_args.data_names = selected_datasets
            self.full_args.data_dirs = data_dirs
            self.full_args.max_length = self.settings_vars["max_length"].get()
            self.full_args.trim = self.settings_vars["trim"].get()
            self.full_args.delimiter = self.settings_vars["delimiter"].get()
            self.full_args.col_names = [name.strip() for name in self.settings_vars["col_names"].get().split(",") if name.strip()]
            self.full_args.aa_to_dna = self.settings_vars["aa_to_dna"].get()
            self.full_args.aa_to_rna = self.settings_vars["aa_to_rna"].get()
            self.full_args.dna_to_aa = self.settings_vars["dna_to_aa"].get()
            self.full_args.rna_to_aa = self.settings_vars["rna_to_aa"].get()
            self.full_args.codon_to_aa = self.settings_vars["codon_to_aa"].get()
            self.full_args.aa_to_codon = self.settings_vars["aa_to_codon"].get()
            self.full_args.random_pair_flipping = self.settings_vars["random_pair_flipping"].get()
            
            # Handle multi_column - convert space-separated string to list or None
            multi_column_str = self.settings_vars["multi_column"].get().strip()
            if multi_column_str:
                self.full_args.multi_column = multi_column_str.split()
            else:
                self.full_args.multi_column = None

            # Update mixin attributes
            self._max_length = self.full_args.max_length
            self._trim = self.full_args.trim
            self._delimiter = self.full_args.delimiter
            self._col_names = self.full_args.col_names
            self._multi_column = self.full_args.multi_column
            self._aa_to_dna = self.full_args.aa_to_dna
            self._aa_to_rna = self.full_args.aa_to_rna
            self._dna_to_aa = self.full_args.dna_to_aa
            self._rna_to_aa = self.full_args.rna_to_aa
            self._codon_to_aa = self.full_args.codon_to_aa
            self._aa_to_codon = self.full_args.aa_to_codon

            # Create data args and get datasets
            self.data_args = DataArguments(**self.full_args.__dict__)
            args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
            self.logger_args = SimpleNamespace(**args_dict)

            self._write_args()
            self.get_datasets()
            print_message("Data downloaded and stored")
            print_done()
            
        self.run_in_background(background_get_data)

    def _get_embeddings(self):
        if not self.all_seqs:
            print_message('Sequences are not loaded yet. Please run the data tab first.')
            return
            
        # Gather settings
        print_message("Computing embeddings...")
        pooling_str = self.settings_vars["embedding_pooling_types"].get().strip()
        pooling_list = [p.strip() for p in pooling_str.split(",") if p.strip()]
        dtype_val = self._selected_embed_dtype()
        
        def background_get_embeddings():
            # Update full args
            self.full_args.all_seqs = self.all_seqs
            self.full_args.model_dtype = self._selected_model_dtype()
            self.full_args.embedding_batch_size = self.settings_vars["batch_size"].get()
            self.full_args.embedding_num_workers = self.settings_vars["num_workers"].get()
            self.full_args.download_embeddings = self.settings_vars["download_embeddings"].get()
            self.full_args.matrix_embed = self.settings_vars["matrix_embed"].get()
            self.full_args.embedding_pooling_types = pooling_list
            self.full_args.embedding_hidden_state_index = self.settings_vars["embedding_hidden_state_index"].get()
            self.full_args.save_embeddings = True
            self.full_args.embed_dtype = dtype_val
            self.full_args.sql = self.settings_vars["sql"].get()
            self._sql = self.full_args.sql
            self._full = self.full_args.matrix_embed
            
            self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)
            args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
            self.logger_args = SimpleNamespace(**args_dict)
            self._write_args()
            
            print_message("Saving embeddings to disk")
            self.save_embeddings_to_disk()
            print_message("Embeddings saved to disk")
            print_done()
            
        self.run_in_background(background_get_embeddings)

    def _browse_replay_log(self):
        filename = filedialog.askopenfilename(
            title="Select Replay Log",
            filetypes=(("Txt files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            self.settings_vars["replay_path"].set(filename)

    def _start_replay(self):
        replay_path = self.settings_vars["replay_path"].get()
        if not replay_path:
            print_message("Please select a replay log file first")
            return
        
        print_message("Starting replay from log file...")
        
        def background_replay():
            from logger import LogReplayer
            replayer = LogReplayer(replay_path)
            replay_args = replayer.parse_log()
            replay_args.replay_path = replay_path
            
            # Create a new MainProcess instance with replay_args
            main = MainProcess(replay_args, GUI=False)
            for k, v in main.full_args.__dict__.items():
                print(f"{k}:\t{v}")
            
            # Run the replay on this MainProcess instance
            replayer.run_replay(main)
            print_done()
        
        self.run_in_background(background_replay)
        
    def _browse_results_file(self):
        filename = filedialog.askopenfilename(
            title="Select Results File",
            filetypes=(("TSV files", "*.tsv"), ("All files", "*.*"))
        )
        if filename:
            self.settings_vars["results_file"].set(filename)
            # Set use_current_run to False since we're selecting a specific file
            self.settings_vars["use_current_run"].set(False)
    
    def _generate_plots(self):
        print_message("Generating visualization plots...")
        
        # Determine which results file to use
        results_file = None
        
        if self.settings_vars["use_current_run"].get() and hasattr(self, 'random_id'):
            # Use the current run's random ID
            results_file = os.path.join(self.settings_vars["results_dir"].get(), f"{self.random_id}.tsv")
            print_message(f"Using current run results: {results_file}")
        elif self.settings_vars["results_file"].get():
            # Use explicitly selected file
            results_file = self.settings_vars["results_file"].get()
            print_message(f"Using selected results file: {results_file}")
        elif self.settings_vars["result_id"].get():
            # Use the specified result ID
            result_id = self.settings_vars["result_id"].get()
            results_file = os.path.join(self.settings_vars["results_dir"].get(), f"{result_id}.tsv")
            print_message(f"Using results file for ID {result_id}: {results_file}")
        else:
            print_message("No results file specified. Please enter a Result ID, browse for a file, or complete a run first.")
            return
        
        # Check if the results file exists
        if not os.path.exists(results_file):
            print_message(f"Results file not found: {results_file}")
            return
        
        # Get output directory
        output_dir = self.settings_vars["viz_output_dir"].get()
        def background_generate_plots():
            # Call the plot generation function
            print_message(f"Generating plots in {output_dir}...")
            create_plots(results_file, output_dir)
            print_message("Plots generated successfully!")
            print_done()
            
        self.run_in_background(background_generate_plots)


def main():
    root = tk.Tk()
    app = GUI(root)
    print_title("Protify")
    root.mainloop()


if __name__ == "__main__":
    main()
