import copy
import os
import yaml
import json
import csv
from typing import Dict, Any, List, Tuple
from utils import torch_load, print_message
from embedder import Embedder, get_embedding_filename
from base_models.get_base_models import get_tokenizer


if os.environ.get('WANDB_AVAILABLE') == 'true':
    import wandb
else:
    pass


def _parse_pooling_types(value):
    """Accept a list, a single token ('mean'), or a hyphen/comma-joined string ('mean-cls', 'mean,var')."""
    if isinstance(value, (list, tuple)):
        return list(value)
    assert isinstance(value, str), f'pooling types must be str or list, got {type(value)}'
    for sep in ('-', ','):
        if sep in value:
            return [tok.strip() for tok in value.split(sep) if tok.strip()]
    return [value.strip()]


class HyperoptModule:
    def __init__(
        self, 
        main_process,
        model_name: str,
        data_name: str,
        dataset: Tuple,
        emb_dict: Any,
        sweep_config: Dict[str, Any],
        results_list: List[Dict[str, Any]],
        swept_param_keys: List[str] = None
    ):
        self.mp = main_process
        self.model_name = model_name
        self.data_name = data_name
        self.dataset = dataset
        self.emb_dict = emb_dict
        self.sweep_config = sweep_config
        self.results_list = results_list
        self.swept_param_keys = swept_param_keys or []
        
        self.base_probe_args = copy.deepcopy(self.mp.probe_args.__dict__)
        self.base_trainer_args = copy.deepcopy(self.mp.trainer_args.__dict__)
        self.base_embedding_args = copy.deepcopy(self.mp.embedding_args.__dict__)
        
        self.probe_keys = {
            'hidden_size','dropout','n_layers','pre_ln','classifier_size',
            'classifier_dropout','head_size','rotary','use_bias','probe_pooling_types',
            'lora','lora_r','lora_alpha','lora_dropout','probe_type','tokenwise', 'pooling_types',
            'add_token_ids','bom_k'
        }
        self.trainer_keys = {
            'lr','weight_decay','num_epochs','probe_batch_size',
            'base_batch_size','probe_grad_accum','base_grad_accum',
            'patience','seed'
        }
        self.full_args_keys = {
            'random_pair_flipping',
        }
        self.embedding_keys = {
            'embedding_pooling_types', 'embedding_hidden_state_index'
        }
        self.int_keys = {
            'hidden_size', 'n_layers', 'classifier_size', 'head_size',
            'lora_r', 'lora_alpha', 'num_epochs', 'probe_batch_size',
            'base_batch_size', 'probe_grad_accum', 'base_grad_accum',
            'patience', 'seed', 'embedding_hidden_state_index'
        }

    def apply_config(self, cfg: Dict[str, Any]):
        self.mp.probe_args.__dict__.update(copy.deepcopy(self.base_probe_args))
        self.mp.trainer_args.__dict__.update(copy.deepcopy(self.base_trainer_args))
        self.mp.embedding_args.__dict__.update(copy.deepcopy(self.base_embedding_args))

        if 'n_heads' in cfg and 'head_size' not in cfg:
            import warnings
            warnings.warn(
                "'n_heads' in sweep config is deprecated; use 'head_size' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hs = int(cfg['hidden_size']) if 'hidden_size' in cfg else self.mp.probe_args.hidden_size
            legacy = int(cfg.pop('n_heads'))
            assert hs % legacy == 0, f"hidden_size {hs} not divisible by legacy n_heads {legacy}"
            cfg['head_size'] = hs // legacy

        for key in self.int_keys:
            if key in cfg:
                cfg[key] = int(cfg[key])

        if 'hidden_size' in cfg:
            hs = int(cfg['hidden_size'])
            head_size = int(cfg['head_size']) if 'head_size' in cfg else self.mp.probe_args.head_size
            assert hs % head_size == 0, (
                f"sweep hidden_size {hs} not divisible by head_size {head_size}"
            )

        if 'dropout' in cfg:
            cfg['transformer_dropout'] = cfg['dropout']

        if 'probe_pooling_types' in cfg:
            cfg['probe_pooling_types'] = _parse_pooling_types(cfg['probe_pooling_types'])
            cfg['pooling_types'] = cfg['probe_pooling_types']
        if 'embedding_pooling_types' in cfg:
            cfg['embedding_pooling_types'] = _parse_pooling_types(cfg['embedding_pooling_types'])

        for k, v in cfg.items():
            if k in self.probe_keys and hasattr(self.mp.probe_args, k):
                setattr(self.mp.probe_args, k, v)
            if k in self.trainer_keys and hasattr(self.mp.trainer_args, k):
                setattr(self.mp.trainer_args, k, v)
            if k in self.full_args_keys and hasattr(self.mp.full_args, k):
                setattr(self.mp.full_args, k, v)
            # Handle embedding pooling types
            if k in self.embedding_keys:
                if k == 'embedding_pooling_types':
                    self.mp.embedding_args.pooling_types = v
                if k == 'embedding_hidden_state_index':
                    self.mp.embedding_args.hidden_state_index = v

    def reload_embeddings(self, ppi: bool) -> None:
        tokenizer = get_tokenizer(self.model_name)
        test_seq = self.mp.all_seqs[0]
        hidden_state_index = self.mp.embedding_args.hidden_state_index
        if self.mp._sql:
            filename = get_embedding_filename(
                self.model_name,
                self.mp._full,
                self.mp.embedding_args.pooling_types,
                'db',
                hidden_state_index,
            )
            save_path = os.path.join(self.mp.embedding_args.embedding_save_dir, filename)
            if not os.path.exists(save_path):
                embedder = Embedder(self.mp.embedding_args, self.mp.all_seqs)
                embedder(self.model_name)
            input_dim = self.mp.get_embedding_dim_sql(save_path, test_seq, tokenizer)
            self.emb_dict = None
        else:
            filename = get_embedding_filename(
                self.model_name,
                self.mp._full,
                self.mp.embedding_args.pooling_types,
                'pth',
                hidden_state_index,
            )
            save_path = os.path.join(self.mp.embedding_args.embedding_save_dir, filename)
            if not os.path.exists(save_path):
                previous_save_embeddings = self.mp.embedding_args.save_embeddings
                self.mp.embedding_args.save_embeddings = True
                embedder = Embedder(self.mp.embedding_args, self.mp.all_seqs)
                try:
                    embedder(self.model_name)
                finally:
                    self.mp.embedding_args.save_embeddings = previous_save_embeddings
            self.emb_dict = torch_load(save_path)
            input_dim = self.mp.get_embedding_dim_pth(self.emb_dict, test_seq, tokenizer)

        self.mp.probe_args.input_size = input_dim * 2 if (ppi and not self.mp._full) else input_dim
        self.mp.probe_args.max_seq_len = self.mp._max_length * 2 if (ppi and self.mp._full) else self.mp._max_length

    def train_model(self, sweep_mode=True):
        train_set, valid_set, test_set, _, _, ppi = self.dataset
        
        if self.mp.full_args.full_finetuning:
             model, valid_metrics, test_metrics = self.mp._run_full_finetuning(
                self.model_name, self.data_name, 
                train_set, valid_set, test_set, 
                ppi=ppi, sweep_mode=sweep_mode
            )
             return model, valid_metrics, test_metrics

        elif self.mp.full_args.hybrid_probe:
            tokenizer = get_tokenizer(self.model_name)
            model, valid_metrics, test_metrics = self.mp._run_hybrid_probe(
                self.model_name, self.data_name,
                train_set, valid_set, test_set,
                tokenizer,
                emb_dict=self.emb_dict,
                ppi=ppi,
                sweep_mode=sweep_mode
            )
            return model, valid_metrics, test_metrics

        else:
            tokenizer = get_tokenizer(self.model_name)
            probe, valid_metrics, test_metrics = self.mp._run_nn_probe(
                self.model_name, self.data_name,
                train_set, valid_set, test_set,
                tokenizer, 
                emb_dict=self.emb_dict, 
                ppi=ppi, 
                sweep_mode=sweep_mode
            )
            return probe, valid_metrics, test_metrics

    def select_metric(self, valid_metrics: Dict[str, Any], test_metrics: Dict[str, Any], sweep_metric: str) -> float:
        if valid_metrics and sweep_metric in valid_metrics:
            return float(valid_metrics[sweep_metric])
        elif test_metrics and sweep_metric in test_metrics:
            return float(test_metrics[sweep_metric])
        
        # Raise a helpful error if metric was not found
        available_keys = []
        if valid_metrics: available_keys.extend(valid_metrics.keys())
        if test_metrics: available_keys.extend(test_metrics.keys())
        raise KeyError(f"Metric '{sweep_metric}' not found in validation or test metrics. Available metrics: {available_keys}")

    def objective(self):
        run = wandb.init(
            project=self.mp.full_args.wandb_project,
            entity=self.mp.full_args.wandb_entity,
            config=self.sweep_config,
            reinit=True,
            tags=["sweep", f"model:{self.model_name}", f"data:{self.data_name}"],
        )
        run.name = f"sweep-{self.model_name}_{self.data_name}-{run.id[:6]}"
        
        # Store only the actual hyperparameters used for this run
        full_config = dict(wandb.config)
        self.apply_config(full_config)
        # Filter to only include the hyperparameters that were actually tuned
        applied_config = {k: v for k, v in full_config.items() if k in self.swept_param_keys}
        self.mp.trainer_args.make_plots = False
        
        embedding_config_changed = (
            'embedding_pooling_types' in full_config
            or 'embedding_hidden_state_index' in full_config
        )
        if embedding_config_changed and not self.mp.full_args.full_finetuning:
            _, _, _, _, _, ppi = self.dataset
            self.reload_embeddings(ppi)

        _, valid_metrics, test_metrics = self.train_model(sweep_mode=True)
        
        # Choose task-specific metric to optimize
        label_type = self.mp.probe_args.task_type
        metric_cls = getattr(self.mp.full_args, 'sweep_metric_cls', None)
        metric_reg = getattr(self.mp.full_args, 'sweep_metric_reg', None)
        dataset_metric = metric_cls if label_type in ["singlelabel", "multilabel"] else metric_reg

        all_metrics = {}
        if isinstance(valid_metrics, dict):
            for k, v in valid_metrics.items():
                all_metrics[f"{k}"] = v
        if isinstance(test_metrics, dict):
            for k, v in test_metrics.items():
                all_metrics[f"{k}"] = v
        wandb.log(all_metrics)
        
        metric_value = self.select_metric(valid_metrics, test_metrics, dataset_metric)
        
        self.results_list.append({
            "wandb_run_id": run.id,
            dataset_metric: metric_value,
            "config": applied_config,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        })
        
        run.finish()
        return float(metric_value)

    @classmethod
    def run_wandb_hyperopt(cls, mp):
        mp.logger.info("Called method: run_wandb_hyperopt")

        sweep_config = {}
        sweep_config_path = mp.full_args.sweep_config_path
            
        if os.path.exists(sweep_config_path):
            with open(sweep_config_path, 'r') as f:
                sweep_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Sweep config file not found: {sweep_config_path}")

        params_to_hyperopt = sweep_config.get("parameters", {})
        
        # Filter parameters based on probe type and LoRA settings
        probe_type = getattr(mp.probe_args, 'probe_type', 'mlp')
        use_lora = getattr(mp.probe_args, 'lora', False)
        
        # Define which parameters are relevant for each probe type
        mlp_probe_params = {'lr', 'weight_decay', 'hidden_size', 'n_layers', 'dropout', 'pre_ln', 'use_bias', 'probe_batch_size', 'embedding_hidden_state_index'}
        transformer_probe_params = {'lr', 'weight_decay', 'hidden_size', 'n_layers', 'transformer_dropout', 'pre_ln',
                                     'classifier_dropout', 'classifier_size', 'use_bias', 'probe_pooling_types', 'embedding_pooling_types', 'probe_batch_size',
                                     'embedding_hidden_state_index', 'bom_k', 'head_size', 'probe_grad_accum', 'add_token_ids', 'random_pair_flipping'}
        lora_params = {'lora_r', 'lora_alpha', 'lora_dropout'}
        
        # Determine which parameters to include
        if probe_type == 'mlp':
            relevant_params = mlp_probe_params
        elif probe_type == 'transformer':
            relevant_params = transformer_probe_params
        else:
            # For other probe types, include all common params
            relevant_params = mlp_probe_params | transformer_probe_params
        
        # Add LoRA parameters only if LoRA is enabled
        if use_lora:
            relevant_params = relevant_params | lora_params
        
        # Filter the parameters dictionary
        filtered_params = {k: v for k, v in params_to_hyperopt.items() if k in relevant_params}
        params_to_hyperopt = filtered_params
        
        # Log which parameters are being swept
        mp.logger.info(f"Probe type: {probe_type}, LoRA enabled: {use_lora}")
        mp.logger.info(f"Sweeping over {len(params_to_hyperopt)} parameters: {list(params_to_hyperopt.keys())}")

        method = mp.full_args.sweep_method
        early_term = sweep_config.get("early_terminate", None)

        total_combinations = len(mp.model_args.model_names) * len(mp.datasets)
        mp.logger.info(f"Hyperopt over {total_combinations} model/dataset combinations")
        for model_name in mp.model_args.model_names:
            tokenizer = get_tokenizer(model_name)
            test_seq = mp.all_seqs[0]

            if "random" in model_name.lower() or "onehot" in model_name.lower():
                print_message(f"Skipping hyperparameter optimization for {model_name}.")

                for data_name, dataset in mp.datasets.items():
                    train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                    mp.probe_args.num_labels = num_labels
                    mp.probe_args.task_type = label_type
                    mp.trainer_args.task_type = label_type
                    mp.trainer_args.make_plots = True

                    emb_dict = None
                    if not mp.full_args.full_finetuning:
                        if mp._sql:
                            filename = get_embedding_filename(
                                model_name,
                                mp._full,
                                mp.embedding_args.pooling_types,
                                'db',
                                mp.embedding_args.hidden_state_index,
                            )
                            save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                            input_dim = mp.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                        else:
                            filename = get_embedding_filename(
                                model_name,
                                mp._full,
                                mp.embedding_args.pooling_types,
                                'pth',
                                mp.embedding_args.hidden_state_index,
                            )
                            save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                            emb_dict = torch_load(save_path)
                            input_dim = mp.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)
                        mp.probe_args.input_size = input_dim * 2 if (ppi and not mp._full) else input_dim
                        mp.probe_args.max_seq_len = mp._max_length * 2 if (ppi and mp._full) else mp._max_length
                    if mp.full_args.full_finetuning:
                        _ = mp._run_full_finetuning(model_name, data_name, train_set, valid_set, test_set, ppi, sweep_mode=False)
                    elif mp.full_args.hybrid_probe:
                        _ = mp._run_hybrid_probe(model_name, data_name, train_set, valid_set, test_set, tokenizer, emb_dict=emb_dict, ppi=ppi, sweep_mode=False)
                    else:
                        _ = mp._run_nn_probe(model_name, data_name, train_set, valid_set, test_set, tokenizer, emb_dict=emb_dict, ppi=ppi, sweep_mode=False)
                continue

            for data_name, dataset in mp.datasets.items():
                mp.logger.info(f"Sweeping over {data_name} with {model_name}")
                train_set, _, _, num_labels, label_type, ppi = dataset
                mp.probe_args.num_labels = num_labels
                mp.probe_args.task_type = label_type
                mp.trainer_args.task_type = label_type

                emb_dict = None
                if not mp.full_args.full_finetuning:
                    if mp._sql:
                        filename = get_embedding_filename(
                            model_name,
                            mp._full,
                            mp.embedding_args.pooling_types,
                            'db',
                            mp.embedding_args.hidden_state_index,
                        )
                        save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                        input_dim = mp.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                    else:
                        filename = get_embedding_filename(
                            model_name,
                            mp._full,
                            mp.embedding_args.pooling_types,
                            'pth',
                            mp.embedding_args.hidden_state_index,
                        )
                        save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                        emb_dict = torch_load(save_path)
                        input_dim = mp.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)
                    mp.probe_args.input_size = input_dim * 2 if (ppi and not mp._full) else input_dim
                    mp.probe_args.max_seq_len = mp._max_length * 2 if (ppi and mp._full) else mp._max_length

                # Save base args for restoring after each trial
                base_probe = copy.deepcopy(mp.probe_args.__dict__)
                base_trainer = copy.deepcopy(mp.trainer_args.__dict__)

                results_list = []
                # Choose task-specific metric to optimize
                metric_cls = getattr(mp.full_args, 'sweep_metric_cls', None)
                metric_reg = getattr(mp.full_args, 'sweep_metric_reg', None)
                dataset_metric = metric_cls if label_type in ["singlelabel", "multilabel"] else metric_reg
                
                hyperopt_module = cls(
                    main_process=mp,
                    model_name=model_name,
                    data_name=data_name,
                    dataset=dataset,
                    emb_dict=emb_dict,
                    sweep_config=sweep_config,
                    results_list=results_list,
                    swept_param_keys=list(params_to_hyperopt.keys())
                )

                wb_sweep = {
                    "method": method,
                    "metric": {"name": dataset_metric, "goal": mp.full_args.sweep_goal},
                    "early_terminate": early_term,
                    "parameters": params_to_hyperopt,
                }
                cli_name = getattr(mp.full_args, 'sweep_name', None)
                if cli_name:
                    wb_sweep["name"] = cli_name
                elif "name" in sweep_config:
                    wb_sweep["name"] = sweep_config["name"]
                sweep_id = wandb.sweep(sweep=wb_sweep, project=mp.full_args.wandb_project, entity=mp.full_args.wandb_entity)
                wandb.agent(sweep_id, function=hyperopt_module.objective, count=mp.full_args.sweep_count)

                # Sort, write, and save sweep results
                reverse_flag = True if mp.full_args.sweep_goal == "maximize" else False
                results_list.sort(key=lambda x: x[dataset_metric], reverse=reverse_flag)
                sweep_log_path = os.path.join(mp.full_args.log_dir, f"{mp.random_id}_sweep_{data_name}_{model_name}.csv")
                with open(sweep_log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=',')
                    # Columns
                    columns = ["rank","wandb_run_id",dataset_metric,"config","valid_metrics","test_metrics"]
                    writer.writerow(columns)
                    for idx, res in enumerate(results_list, start=1):
                        writer.writerow([
                            idx,
                            res['wandb_run_id'],
                            res[dataset_metric],
                            json.dumps(res['config']),
                            json.dumps(res['valid_metrics']),
                            json.dumps(res['test_metrics']),
                        ])

                # Log best hyperparameters
                best = results_list[0] if results_list else None
                best_score = best[dataset_metric]
                best_config = best['config']
                print_message(f"Best sweep result - {dataset_metric}: {best_score}")
                print_message(f"Best hyperparameters: {json.dumps(best_config, indent=2)}")

                # Restore base args then apply best
                mp.probe_args.__dict__.update(copy.deepcopy(base_probe))
                mp.trainer_args.__dict__.update(copy.deepcopy(base_trainer))
                hyperopt_module.apply_config(best_config)
                mp.trainer_args.make_plots = True
                best_embedding_config_changed = (
                    'embedding_pooling_types' in best_config
                    or 'embedding_hidden_state_index' in best_config
                )
                if best_embedding_config_changed and not mp.full_args.full_finetuning:
                    hyperopt_module.reload_embeddings(ppi)
                
                final_config = {
                    **best_config,
                    'probe_batch_size': mp.trainer_args.probe_batch_size,
                    'seed': mp.trainer_args.seed,
                    'patience': mp.trainer_args.patience,
                    'num_epochs': mp.trainer_args.num_epochs,
                }
                print_message(f"Final training config: {json.dumps(final_config, indent=2)}")

                # Create a fresh wandb run for the final model to track it
                final_run = wandb.init(
                    project=mp.full_args.wandb_project,
                    entity=mp.full_args.wandb_entity,
                    config=final_config,
                    reinit=True,
                    tags=["final_model", f"model:{model_name}", f"data:{data_name}", f"best_sweep_score:{best_score}"],
                    name=f"final-{model_name}_{data_name}-best",
                )

                # Run best model with the best hyperparameters, log metrics, create plots
                _, valid_metrics, test_metrics = hyperopt_module.train_model(sweep_mode=False)
                
                # Log final model metrics to wandb
                all_final_metrics = {}
                if isinstance(valid_metrics, dict):
                    for k, v in valid_metrics.items():
                        all_final_metrics[f"final_{k}"] = v
                if isinstance(test_metrics, dict):
                    for k, v in test_metrics.items():
                        all_final_metrics[f"final_{k}"] = v
                wandb.log(all_final_metrics)
                
                final_run.finish()