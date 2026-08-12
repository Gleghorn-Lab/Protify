import os
import argparse
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dataclasses import dataclass, field
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE as SklearnTSNE
from typing import Optional, Union, List
from matplotlib.colors import LinearSegmentedColormap

try:
    from utils import torch_load, print_message
    from seed_utils import get_global_seed, set_global_seed, set_determinism
    from data.data_mixin import DataMixin, DataArguments
    from embedder import Embedder, EmbeddingArguments, get_embedding_filename
except ImportError:
    from ..utils import torch_load, print_message
    from ..seed_utils import get_global_seed, set_global_seed, set_determinism
    from ..data.data_mixin import DataMixin, DataArguments
    from ..embedder import Embedder, EmbeddingArguments, get_embedding_filename


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class VisualizationArguments:
    # Paths
    embedding_save_dir: str = "embeddings"
    fig_dir: str = "figures"
    
    # Model and embedding settings
    model_name: str = "ESM2-8"
    matrix_embed: bool = False
    sql: bool = False
    
    # Embedding arguments (defaults from main.py)
    embedding_batch_size: int = 16
    num_workers: int = 0
    download_embeddings: bool = False
    download_dir: str = "Synthyra/vector_embeddings"
    embedding_pooling_types: List[str] = field(default_factory=lambda: ["mean"])
    embedding_hidden_state_index: int = -1
    max_length: int = 2048
    save_embeddings: bool = False
    embed_dtype: str = "float32"  # Will be converted to torch dtype
    
    # Dimensionality reduction settings
    n_components: int = 2
    perplexity: float = 30.0  # for t-SNE
    n_neighbors: int = 15     # for UMAP
    min_dist: float = 0.1     # for UMAP
    
    # Visualization settings
    seed: Optional[int] = None  # If None, will use current time
    deterministic: bool = False
    fig_size: tuple = (10, 10)
    save_fig: bool = True
    task_type: str = "singlelabel"  # singlelabel, multilabel, regression


class DimensionalityReducer(DataMixin):
    """Base class for dimensionality reduction techniques"""
    def __init__(self, args: VisualizationArguments):
        # Initialize DataMixin without data_args since we're not loading datasets
        super().__init__(data_args=None)
        self.args = args
        self.embeddings = None
        self.labels = None
        # Set DataMixin instance variables based on args
        self._sql = args.sql
        self._full = args.matrix_embed
        
    def _check_and_embed(self, sequences: List[str]):
        """Check if embeddings exist, and embed sequences if they don't"""
        # Ensure embedding save directory exists
        os.makedirs(self.args.embedding_save_dir, exist_ok=True)
        
        # Check if we need to embed (similar to Embedder._read_embeddings_from_disk)
        pooling_types = self.args.embedding_pooling_types
        hidden_state_index = self.args.embedding_hidden_state_index
        filename_pth = get_embedding_filename(
            self.args.model_name,
            self.args.matrix_embed,
            pooling_types,
            'pth',
            hidden_state_index,
            max_length=self.args.max_length,
        )
        filename_db = get_embedding_filename(
            self.args.model_name,
            self.args.matrix_embed,
            pooling_types,
            'db',
            hidden_state_index,
            max_length=self.args.max_length,
        )
        save_path = os.path.join(self.args.embedding_save_dir, filename_pth)
        db_path = os.path.join(self.args.embedding_save_dir, filename_db)
        
        if self._sql:
            # Check SQL database
            import sqlite3
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
                c.execute("SELECT sequence FROM embeddings")
                already_embedded = set(row[0] for row in c.fetchall())
                conn.close()
                to_embed = [seq for seq in sequences if seq not in already_embedded]
            else:
                to_embed = sequences
        else:
            # Check PyTorch file
            if os.path.exists(save_path):
                emb_dict = torch_load(save_path)
                to_embed = [seq for seq in sequences if seq not in emb_dict]
            else:
                to_embed = sequences
        
        # If there are sequences to embed, do it
        if len(to_embed) > 0:
            print_message(f"Embedding {len(to_embed)} sequences that are not yet embedded")
            # Convert embed_dtype string to torch dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            embed_dtype = dtype_map.get(self.args.embed_dtype, torch.float32)
            
            # Create EmbeddingArguments matching VisualizationArguments
            embedding_args = EmbeddingArguments(
                embedding_batch_size=self.args.embedding_batch_size,
                embedding_num_workers=self.args.num_workers,
                download_embeddings=self.args.download_embeddings,
                download_dir=self.args.download_dir,
                matrix_embed=self.args.matrix_embed,
                embedding_pooling_types=self.args.embedding_pooling_types,
                save_embeddings=True,  # Always save embeddings when auto-embedding
                embed_dtype=embed_dtype,
                sql=self.args.sql,
                embedding_save_dir=self.args.embedding_save_dir,
                embedding_hidden_state_index=self.args.embedding_hidden_state_index,
                max_length=self.args.max_length,
            )
            # Initialize embedder with all sequences (it will only embed missing ones)
            embedder = Embedder(embedding_args, sequences)
            # Embed using the model name - embedder handles checking what needs embedding internally
            embedder(self.args.model_name)
            print_message(f"Finished embedding sequences")
        else:
            print_message(f"All {len(sequences)} sequences are already embedded")
    
    def load_embeddings(self, sequences: List[str], labels: Optional[List[Union[int, float, List[int]]]] = None):
        """Load embeddings from file using DataMixin functionality"""
        # First check if embeddings exist and embed if needed
        self._check_and_embed(sequences)
        
        embeddings = []
        
        pooling_types = self.args.embedding_pooling_types
        hidden_state_index = self.args.embedding_hidden_state_index
        if self._sql:
            import sqlite3
            filename = get_embedding_filename(
                self.args.model_name,
                self.args.matrix_embed,
                pooling_types,
                'db',
                hidden_state_index,
                max_length=self.args.max_length,
            )
            save_path = os.path.join(self.args.embedding_save_dir, filename)
            with sqlite3.connect(save_path) as conn:
                c = conn.cursor()
                for seq in sequences:
                    # Use DataMixin's _select_from_sql method
                    embedding = self._select_from_sql(c, seq, cast_to_torch=False)
                    # Reshape to 1D if needed (DataMixin returns shape (1, dim) or (seq_len, dim))
                    if len(embedding.shape) > 1:
                        if self._full:
                            # Average over sequence length
                            embedding = embedding.mean(axis=0)
                        else:
                            # Already averaged, just squeeze
                            embedding = embedding.squeeze(0)
                    embeddings.append(embedding)
        else:
            filename = get_embedding_filename(
                self.args.model_name,
                self.args.matrix_embed,
                pooling_types,
                'pth',
                hidden_state_index,
                max_length=self.args.max_length,
            )
            save_path = os.path.join(self.args.embedding_save_dir, filename)
            emb_dict = torch_load(save_path)
            for seq in sequences:
                # Use DataMixin's _select_from_pth method
                embedding = self._select_from_pth(emb_dict, seq, cast_to_np=True)
                # Reshape to 1D if needed
                if len(embedding.shape) > 1:
                    if self._full:
                        # Average over sequence length
                        embedding = embedding.mean(axis=0)
                    else:
                        # Already averaged, just squeeze
                        embedding = embedding.squeeze(0)
                embeddings.append(embedding)

        print_message(f"Loaded {len(embeddings)} embeddings")
        self.embeddings = np.stack(embeddings)
        if labels is not None:
            # Convert labels to a numpy array. For multi-label, this can be shape (num_samples, num_labels).
            self.labels = np.array(labels)
        else:
            self.labels = None
        
    def fit_transform(self):
        """Implement in child class"""
        raise NotImplementedError
        
    def plot(self, save_name: Optional[str] = None):
        """Plot the reduced dimensionality embeddings with appropriate coloring scheme"""
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")
            
        print_message("Fitting and transforming")
        reduced = self.fit_transform()
        print_message("Plotting")
        plt.figure(figsize=self.args.fig_size)
        
        if self.labels is None:
            # No labels - just a single color
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
            
        elif self.args.task_type == "singlelabel":
            unique_labels = np.unique(self.labels)
            # Handle binary or multiclass
            if len(unique_labels) == 2:  # Binary classification
                colors = ['#ff7f0e', '#1f77b4']  # Orange and Blue
                cmap = LinearSegmentedColormap.from_list('binary', colors)
                scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                      c=self.labels, cmap=cmap, alpha=0.6)
                plt.colorbar(scatter, ticks=[0, 1])
            else:  # Multiclass classification
                n_classes = len(unique_labels)
                if n_classes <= 10:
                    cmap = 'tab10'
                elif n_classes <= 20:
                    cmap = 'tab20'
                else:
                    # For many classes, create a custom colormap
                    colors = sns.color_palette('husl', n_colors=n_classes)
                    cmap = LinearSegmentedColormap.from_list('custom', colors)
                
                scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                      c=self.labels, cmap=cmap, alpha=0.6)
                plt.colorbar(scatter, ticks=unique_labels)
                
        elif self.args.task_type == "multilabel":
            # For multi-label, create spectrum from blue to red along the label axis
            # where more blue if the labels are closer to index 0 and more red if the labels are closer to index -1
            # If there are more than one postive (multi-hot), average their colors
            label_colors = np.zeros(len(self.labels))
            label_counts = np.sum(self.labels, axis=1)
            
            # For samples with positive labels, calculate the weighted average position
            for i, label_row in enumerate(self.labels):
                if label_counts[i] > 0:
                    # Calculate weighted position (0 = first label, 1 = last label)
                    positive_indices = np.where(label_row == 1)[0]
                    avg_position = np.mean(positive_indices) / (self.labels.shape[1] - 1)
                    label_colors[i] = avg_position
                    
            # Create a blue to red colormap
            blue_red_cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
            
            # Plot with both color dimensions: count and position
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                  c=label_colors, cmap=blue_red_cmap, 
                                  s=30 + 20 * label_counts, alpha=0.6)
            
            # Add two colorbars
            plt.colorbar(scatter, label='Label Position (blue=first, red=last)')
            
            # Add a size legend for count
            handles, labels = [], []
            for count in sorted(set(label_counts)):
                handles.append(plt.scatter([], [], s=30 + 20 * count, color='gray'))
                labels.append(f'{int(count)} labels')
            plt.legend(handles, labels, title='Label Count', loc='upper right')
            
        elif self.args.task_type == "regression":
            # For regression, use a sequential colormap
            vmin, vmax = np.percentile(self.labels, [2, 98])  # Robust scaling
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                  c=self.labels, cmap='viridis', 
                                  norm=norm, alpha=0.6)
            plt.colorbar(scatter, label='Value')
        
        plt.title(f'{self.__class__.__name__} visualization of {self.args.model_name} embeddings')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_name is not None and self.args.save_fig:
            os.makedirs(self.args.fig_dir, exist_ok=True)
            plt.savefig(os.path.join(self.args.fig_dir, save_name), 
                        dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


class PCA(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.pca = SklearnPCA(n_components=args.n_components, random_state=get_global_seed() or args.seed)
        
    def fit_transform(self):
        return self.pca.fit_transform(self.embeddings)


class TSNE(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.tsne = SklearnTSNE(
            n_components=self.args.n_components,
            perplexity=self.args.perplexity,
            random_state=get_global_seed() or self.args.seed
        )
        
    def fit_transform(self):
        return self.tsne.fit_transform(self.embeddings)


class UMAP(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.umap = umap.UMAP(
            n_components=self.args.n_components,
            n_neighbors=self.args.n_neighbors,
            min_dist=self.args.min_dist,
            random_state=get_global_seed() or self.args.seed
        )
        
    def fit_transform(self):
        return self.umap.fit_transform(self.embeddings)


def parse_arguments():
    """Parse command line arguments for visualization"""
    parser = argparse.ArgumentParser(description="Dimensionality reduction visualization for protein embeddings")
    
    # ----------------- Paths ----------------- #
    parser.add_argument("--embedding_save_dir", type=str, default="embeddings", 
                       help="Directory to save/load embeddings.")
    parser.add_argument("--fig_dir", type=str, default="figures", 
                       help="Directory to save figures.")
    
    # ----------------- Model and Embedding Settings ----------------- #
    parser.add_argument("--model_name", type=str, default="ESM2-8", 
                       help="Model name to use for embeddings.")
    parser.add_argument("--matrix_embed", action="store_true", default=False,
                       help="Use matrix embedding (per-residue embeddings).")
    parser.add_argument("--sql", action="store_true", default=False,
                       help="Use SQL storage for embeddings.")
    
    # ----------------- Embedding Arguments ----------------- #
    parser.add_argument("--embedding_batch_size", type=int, default=16,
                       help="Batch size for embedding generation.")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of worker processes for data loading.")
    parser.add_argument("--download_embeddings", action="store_true", default=False,
                       help="Download embeddings from HuggingFace hub.")
    parser.add_argument("--download_dir", type=str, default="Synthyra/vector_embeddings",
                       help="Directory to download embeddings from.")
    parser.add_argument("--embedding_pooling_types", nargs="+", default=["mean", "var"],
                       help="Pooling types for embeddings.")
    parser.add_argument("--embedding_hidden_state_index", type=int, default=-1,
                       help="Hidden-state tuple index to embed from. -1 uses the final hidden state.")
    parser.add_argument("--save_embeddings", action="store_true", default=False,
                       help="Save computed embeddings (auto-enabled when embedding).")
    parser.add_argument("--embed_dtype", type=str, default="float32", 
                       choices=["float32", "float16", "bfloat16"],
                       help="Data type for embeddings.")
    
    # ----------------- Data Arguments ----------------- #
    parser.add_argument("--data_names", nargs="+", default=["EC"],
                       help="List of dataset names to visualize.")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length.")
    parser.add_argument("--trim", action="store_true", default=False,
                       help="Trim sequences to max_length instead of removing them.")
    
    # ----------------- Dimensionality Reduction Settings ----------------- #
    parser.add_argument("--n_components", type=int, default=2,
                       help="Number of components for dimensionality reduction.")
    parser.add_argument("--perplexity", type=float, default=30.0,
                       help="Perplexity parameter for t-SNE.")
    parser.add_argument("--n_neighbors", type=int, default=15,
                       help="Number of neighbors for UMAP.")
    parser.add_argument("--min_dist", type=float, default=0.1,
                       help="Minimum distance for UMAP.")
    
    # ----------------- Visualization Settings ----------------- #
    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for reproducibility (if omitted, current time is used).")
    parser.add_argument("--deterministic", action="store_true", default=False,
                       help="Enable deterministic behavior (slower but reproducible).")
    parser.add_argument("--fig_size", nargs=2, type=int, default=[10, 10],
                       help="Figure size (width height).")
    parser.add_argument("--save_fig", action="store_true", default=True,
                       help="Save figures to disk.")
    parser.add_argument("--task_type", type=str, default=None,
                       choices=["singlelabel", "multilabel", "regression"],
                       help="Task type (auto-detected from dataset if not specified).")
    
    # ----------------- Reduction Methods ----------------- #
    parser.add_argument("--methods", nargs="+", 
                       choices=["PCA", "TSNE", "UMAP"], 
                       default=["PCA", "TSNE", "UMAP"],
                       help="Dimensionality reduction methods to use.")
    
    return parser.parse_args()


if __name__ == "__main__":
    # py -m visualization.reduce_dim
    # Parse arguments
    args = parse_arguments()
    
    # Set deterministic behavior if requested (must be before torch imports)
    if args.deterministic:
        set_determinism()
    
    # Set global seed before doing anything else
    chosen_seed = set_global_seed(args.seed)
    args.seed = chosen_seed
    print_message(f"Using seed: {chosen_seed}")
    
    # Get data using DataMixin
    data_args = DataArguments(
        data_names=args.data_names,
        max_length=args.max_length,
        trim=args.trim
    )
    data_mixin = DataMixin(data_args=data_args)
    datasets, all_seqs = data_mixin.get_data()
    
    # Get sequences and labels from first dataset
    dataset_name = list(datasets.keys())[0]
    train_set, valid_set, test_set, num_labels, label_type, ppi = datasets[dataset_name]
    
    # Determine task_type from label_type if not specified
    if args.task_type is None:
        if label_type == "multilabel":
            task_type = "multilabel"
        elif label_type in ["regression", "sigmoid_regression"]:
            task_type = "regression"
        else:
            task_type = "singlelabel"
    else:
        task_type = args.task_type
    
    sequences = list(train_set["seqs"])
    labels = list(train_set["labels"])
    
    # Create VisualizationArguments from parsed args
    vis_args = VisualizationArguments(
        embedding_save_dir=args.embedding_save_dir,
        fig_dir=args.fig_dir,
        model_name=args.model_name,
        matrix_embed=args.matrix_embed,
        sql=args.sql,
        embedding_batch_size=args.embedding_batch_size,
        num_workers=args.num_workers,
        download_embeddings=args.download_embeddings,
        download_dir=args.download_dir,
        embedding_pooling_types=args.embedding_pooling_types,
        embedding_hidden_state_index=args.embedding_hidden_state_index,
        save_embeddings=args.save_embeddings,
        embed_dtype=args.embed_dtype,
        n_components=args.n_components,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
        deterministic=args.deterministic,
        fig_size=tuple(args.fig_size),
        save_fig=args.save_fig,
        task_type=task_type
    )
    
    # Map method names to classes
    method_map = {
        "PCA": PCA,
        "TSNE": TSNE,
        "UMAP": UMAP
    }
    
    # Run specified reduction methods
    for method_name in args.methods:
        if method_name not in method_map:
            print_message(f"Unknown method: {method_name}, skipping")
            continue
        
        Reducer = method_map[method_name]
        print_message(f"Running {Reducer.__name__}")
        reducer = Reducer(vis_args)
        print_message("Loading embeddings")
        reducer.load_embeddings(sequences, labels)
        reducer.plot(f"{dataset_name}_{Reducer.__name__}.png")
