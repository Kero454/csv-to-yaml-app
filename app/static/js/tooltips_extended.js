// Extended Tooltip Definitions for Model Configurations
const tooltipDefinitions = {
    // Dataset Configuration
    'dataset': 'Specifies basic information about the dataset. Name of the dataset, such as "weather". Choose a name that indicates the data source or domain.',
    'path': 'The file path where the dataset is stored. Ensure the path points to the folder containing the data.',
    'num_classes': 'Number of classes for classification tasks. Set to null if the task is not classification-related.',
    
    // Scheduler Configuration
    'scheduler_config': 'Controls the learning rate scheduling to improve model training.',
    'scheduler_type': 'Type of learning rate scheduler. Common options include "StepLR", "CosineAnnealingLR", and "ExponentialLR".',
    'gamma': 'Decay factor for learning rate at each step. For example, 0.75 decreases the learning rate by 25% with each step.',
    'step_size': 'Frequency, in steps, at which to adjust the learning rate. Used primarily with "StepLR".',
    
    // Optimizer Configuration
    'optim_config': 'Defines the optimizer\'s properties for adjusting model parameters during training.',
    'optimizer_type': 'Specifies the optimization algorithm, like "Adam", "SGD", or "AdamW".',
    'learning_rate': 'Starting learning rate for the optimizer. Smaller values slow down learning but may lead to more stable convergence.',
    'lr': 'Learning rate for the optimizer. Smaller values slow down learning but may lead to more stable convergence.',
    'weight_decay': 'Regularization parameter that prevents overfitting by adding a penalty for large weights. Often between 0.0001 and 0.01.',
    'momentum': 'Momentum factor, used with optimizers like SGD, which helps speed up convergence. Typical values are around 0.9.',
    'betas': 'Beta coefficients for optimizers like Adam. Controls the exponential decay rates for the moment estimates.',
    
    // Model Configuration
    'model': 'Specifies the overall model settings including architecture type and training mode.',
    'type': 'Specifies the model architecture or paradigm. Options include: tdlgm, rnn, linear, persistent, d3vae, diffusion, tft, informer, vva, vqvae, crossformer, autoformer, patchtst, dilated_conv_ed, tide, itransformer.',
    'retrain': 'Indicates whether to train from scratch (true) or use a pre-trained model (false).',
    
    // Time Series Configuration
    'ts': 'Contains parameters defining the dataset and how it should be enriched or pre-processed.',
    'name': 'Identifier or name of the dataset. Can include template variables like ${dataset.dataset}.',
    'version': 'Version number of the dataset configuration.',
    'enrich': 'List of additional features to include for temporal enrichment, such as ["hour"], ["hour", "month"], or [].',
    'use_covariates': 'Whether to use additional covariate (external) features.',
    'use_future_covariates': 'For some models (e.g., dilated_conv_ed), indicates if future covariates are used.',
    'silly': 'Special flag for diffusion models that may control non-standard behavior.',
    
    // Model Configs
    'model_configs': 'Contains the configuration for model input, output, and structure.',
    'past_steps': 'Number of past time steps (historical data points) used as model input.',
    'future_steps': 'Number of future time steps to predict.',
    'quantiles': 'List of quantiles for quantile regression, such as [0.1, 0.5, 0.9]. Set to [] if not using quantile loss.',
    'past_channels': 'Number of channels in past data; helps with multi-channel data (e.g., temperature, humidity).',
    'future_channels': 'Number of channels in predicted data for multivariate outputs.',
    'embs': 'Embedding size for categorical features, used in models that handle categorical data.',
    'out_channels': 'Number of output channels; useful if the output dimension differs from input.',
    'loss_type': 'Loss function type, e.g., "MSE" for mean squared error, "MAE" for mean absolute error, "QuantileLoss", "tdlgm_loss", "l1", "kl".',
    'persistence_weight': 'Weight given to a persistence model in ensembling. Controls the influence of persistence in the model.',
    
    // RNN/LSTM Parameters
    'hidden_RNN': 'Hidden dimension size for RNN-based layers.',
    'num_layers_RNN': 'Number of RNN layers.',
    'n_layers_RNN': 'Number of RNN layers in the model.',
    'bidirectional': 'For RNNs: whether to use a bidirectional architecture.',
    'lstm_type': 'Type or variant of LSTM to use (e.g., "sss" for xLSTM).',
    
    // Convolutional Parameters
    'kernel_size': 'Convolution kernel size (e.g., 3, 5).',
    'num_blocks': 'Number of blocks if the model is built with block modules.',
    
    // General Architecture Parameters
    'kind': 'Specifies a variant or subtype within a model family (e.g., "tdlgm", "dlinear", "lstm", "xlstm").',
    'sum_emb': 'Whether to sum embeddings (often to combine multiple categorical features).',
    'optim': 'PyTorch optimizer to use. Usually a reference to the class like torch.optim.Adam.',
    'activation': 'Activation function. Can be torch.nn.PReLU, torch.nn.LeakyReLU, torch.nn.GELU, torch.nn.ReLU.',
    'dropout_rate': 'Dropout probability for regularization (e.g., 0.1, 0.2, 0.5).',
    'remove_last': 'Whether to remove the last time step (or element) in sequences.',
    'use_bn': 'Whether to use Batch Normalization.',
    'hidden_size': 'Hidden dimension for linear or feedforward layers.',
    'simple': 'Used in linear models to indicate a simpler variant of the architecture.',
    
    // Transformer Parameters
    'd_model': 'Model dimension commonly used in transformer-like architectures.',
    'd_head': 'Dimension per attention head.',
    'n_head': 'Number of attention heads (e.g., 4, 8, 12, 16).',
    'n_heads': 'Number of attention heads in the model.',
    'n_layer': 'Number of layers in the model (e.g., in PatchTST).',
    'n_layer_encoder': 'Number of encoder layers in transformer models.',
    'n_layer_decoder': 'Number of decoder layers (e.g., in ITransformer).',
    'factor': 'Factor parameter used in models like Autoformer and CrossFormer.',
    
    // TIDE-specific
    'n_add_enc': 'Number of additional encoder layers (used in TIDE).',
    'n_add_dec': 'Number of additional decoder layers (used in TIDE).',
    
    // PatchTST Parameters
    'patch_len': 'Length of patches for patch-based models.',
    'stride': 'Stride size for patch extraction (often implies non-overlapping patches).',
    'decomposition': 'Whether to decompose the input (used in PatchTST).',
    
    // ITransformer Parameters
    'use_norm': 'Whether to use normalization in the decoder or output layer.',
    'class_strategy': 'Strategy for aggregating class/token information: "projection", "average", or "cls_token".',
    
    // Autoformer Parameters
    'label_len': 'Label length for Autoformer models.',
    
    // CrossFormer Parameters
    'win_size': 'Window size for CrossFormer.',
    'seg_len': 'Segment length for CrossFormer.',
    
    // VQ-VAE/VVA Parameters
    'hidden_channels': 'Number of hidden channels for VQVAE.',
    'commitment_cost': 'Cost factor for vector quantization commitment.',
    'decay': 'Decay factor for moving averages in VQVAE.',
    'max_voc_size': 'Maximum vocabulary size for token-based models (e.g., 64, 128).',
    'token_split': 'How many splits or tokens to create.',
    'epoch_vqvae': 'Number of epochs after which to switch training phases (VQVAE).',
    'lr_vqvae': 'Learning rate for the VQVAE part.',
    'lr_gpt': 'Learning rate for the GPT part.',
    'weight_decay_vqvae': 'Weight decay for the VQVAE part.',
    'weight_decay_gpt': 'Weight decay for the GPT part.',
    
    // D3VAE Parameters
    'embedding_dimension': 'Embedding dimension for the VAE latent space.',
    'scale': 'Scaling factor for the embeddings.',
    'num_layers': 'Number of layers in the model.',
    'diff_steps': 'Number of diffusion steps for diffusion models.',
    'beta_start': 'Start value for beta in diffusion scheduling.',
    'beta_end': 'End value for beta in diffusion scheduling.',
    'beta_schedule': 'Schedule type for beta (e.g., "linear").',
    'channel_mult': 'Multiplier for channels (D3VAE).',
    'mult': 'A multiplier factor (D3VAE).',
    
    // Diffusion Model Parameters
    'diffusion': 'Diffusion model that gradually denoises data over multiple steps to generate forecasts.',
    'learn_var': 'Whether the variance is learned in diffusion models.',
    'cosine_alpha': 'Whether to use a cosine schedule for alpha.',
    'diffusion_steps': 'Total number of diffusion steps (recommend >70 for good results).',
    'beta': 'Beta value when not learning variance (if learn_var is false, use beta >0.04).',
    'subnet': 'Sub-network configuration parameter (diffusion).',
    'perc_subnet_learning_for_step': 'Percentage of subnet learning allocated per step.',
    
    // Dilated Conv Parameters
    'dilated_conv': 'Uses dilated convolutions to expand receptive field exponentially.',
    'dilated_conv_ed': 'Encoder-decoder architecture with dilated convolutions.',
    'use_cumsum': 'Whether to use cumulative sum in preprocessing.',
    'use_bilinear': 'Whether to use a bilinear layer.',
    
    // Experiment Stack Configuration
    'stack': 'Contains the directory path for experiment outputs.',
    'dirpath': 'Path to save experiment data (e.g., checkpoints, logs). Set to a specific folder to store files.',
    
    // Data Splitting Parameters
    'split_params': 'Settings for dividing the dataset into training, validation, and testing.',
    'perc_train': 'Fraction of data for training, typically 0.8 for 80%.',
    'perc_valid': 'Fraction of data for validation, often 0.1 (10%).',
    'range_train': 'Date range for training data, e.g., ["2010-01-01", "2018-12-31"]. Null for default split.',
    'range_validation': 'Date range for validation data, or null for automatic selection.',
    'range_test': 'Date range for testing data, or null for default.',
    'shift': 'Amount of shift in the window, often 0 or 1 for time series.',
    'starting_point': 'Initial index for the time series. Null starts from the default first point.',
    'skip_step': 'Skip steps for the time series, commonly 1 for daily data or 7 for weekly.',
    'scaler': 'Data scaler, e.g., "StandardScaler()" or "MinMaxScaler()" for data normalization.',
    'test_size': 'Fraction of data reserved for testing.',
    'validation_size': 'Fraction of remaining data for validation.',
    'shuffle': 'Whether to shuffle data before splitting (only for non-time-dependent data).',
    'random_seed': 'Seed for random operations to ensure reproducibility.',
    'keep_entire_seq_while_shifting': 'If true, retains the entire sequence during the shift operation.',
    
    // Training Configuration
    'train_config': 'Parameters for model training.',
    'batch_size': 'Number of samples processed before updating model parameters.',
    'max_epochs': 'Maximum training epochs, where one epoch means a full pass through the dataset.',
    'gradient_clip_val': 'Gradient clipping value to prevent exploding gradients; null if not used.',
    'gradient_clip_algorithm': 'Type of gradient clipping, either "norm" (based on gradient norm) or "value" (by value).',
    'precision': 'Precision for training, such as 32, 16, "bf16" for mixed precision, or "auto" to select automatically.',
    'modifier': 'Optional modifier for the data loader, if special data handling is required (e.g., "ModifierVVA").',
    'modifier_params': 'Parameters for any specified modifier.',
    'num_workers': 'Number of CPU workers for loading data. Set to 0 for minimal setups.',
    'auto_lr_find': 'Automatically search for an optimal learning rate.',
    'devices': 'Specifies GPU/CPU devices for training. Use [0] for one GPU or "auto" to detect available devices.',
    'seed': 'Random seed to ensure reproducibility.',
    
    // Inference Configuration
    'inference': 'Parameters for running inference (predictions).',
    'output_path': 'Directory to save inference outputs.',
    'load_last': 'Whether to load the last trained model checkpoint for inference.',
    'set': 'Data subset to use for inference, options include "train", "validation", and "test".',
    'rescaling': 'Whether to rescale errors to the original data scale for error analysis.',
    
    // Hydra Configuration
    'defaults': 'Hydra settings for managing configurations and parallelization.',
    '_self_': 'Loads the current configuration file.',
    'architecture': 'Specifies the model architecture; typically set to null to allow runtime override.',
    'hydra': 'Contains configurations for Hydra\'s parallel execution and tuning features.',
    'launcher': 'Defines parameters for joblib\'s parallel processing.',
    'n_jobs': 'Number of parallel jobs in multirun mode.',
    'pre_dispatch': 'Number of batches to load in parallel.',
    'verbose': 'Verbosity level (1 for detailed output, 0 for silent).',
    '_target_': 'Specifies the target class for configuration.',
    'output_subdir': 'Set to null to prevent Hydra from saving an output subdirectory by default.',
    
    // Optuna/Sweeper Configuration
    'sweeper': 'Contains parameters for Optuna-based hyperparameter optimization.',
    'sampler': 'Configures the Optuna sampler for hyperparameter search.',
    'consider_prior': 'Advanced Optuna option for considering prior information.',
    'prior_weight': 'Weight given to prior information in Optuna sampling.',
    'consider_magic_clip': 'Advanced Optuna option for magic clipping.',
    'consider_endpoints': 'Whether to consider endpoints in Optuna sampling.',
    'n_startup_trials': 'Number of startup trials for Optuna.',
    'n_ei_candidates': 'Number of expected improvement candidates.',
    'multivariate': 'Whether to use multivariate sampling in Optuna.',
    'warn_independent_sampling': 'Whether to warn about independent sampling.',
    'direction': 'Direction of optimization, either "minimize" or "maximize".',
    'storage': 'Path to save Optuna trials in SQLite; null for in-memory.',
    'study_name': 'Name of the Optuna study, e.g., "tft".',
    'n_trials': 'Maximum number of optimization trials.',
    'params': 'Hyperparameter search space for tuning.',
    'cat_emb_dim': 'Embedding dimension for categorical features.',
    
    // Logging Configuration
    'logging': 'Settings for tracking model training and validation.',
    'log_dir': 'Directory for storing logs of training metrics.',
    
    // Model Family Descriptions
    'rnn': 'RNN Family: Process sequential data by maintaining hidden states that capture temporal dependencies. Includes GRU, LSTM, and xLSTM variants.',
    'gru': 'Gated Recurrent Unit: Uses reset and update gates for efficient learning of short-term dependencies. Fewer parameters than LSTM.',
    'lstm': 'Long Short-Term Memory: Adds memory cells and gates (input, output, forget) to handle long-term dependencies more effectively.',
    'xlstm': 'Extended LSTM: Extends LSTM with bidirectional processing and specialized configurations for enhanced sequence modeling.',
    
    'linear': 'Linear Family: Projects time series data directly using linear transformations, often incorporating decomposition or normalization.',
    'dlinear': 'Decomposes time series into trend and seasonal components, processes them separately with linear layers, then combines results.',
    'nlinear': 'Normalizes input by subtracting the last observed value to handle non-stationarity before applying linear layers.',
    'alinear': 'Attention-augmented linear transformations, focusing on dynamic feature integration.',
    
    'tft': 'Temporal Fusion Transformer: Combines transformers with RNNs and variable selection networks for interpretable multi-horizon forecasts.',
    'informer': 'Uses ProbSparse attention to reduce computational complexity for long sequences.',
    'autoformer': 'Integrates series decomposition (trend/seasonal separation) into the transformer.',
    'crossformer': 'Applies cross-correlation attention across time series segments.',
    'patchtst': 'Patch Time Series Transformer: Splits time series into patches for efficient local-global pattern capture.',
    'itransformer': 'Instance Transformer: Treats time points as features and uses instance-wise attention for multivariate forecasting.',
    
    'tdlgm': 'Temporal Deep Latent Generative Model: Uses bidirectional RNNs and latent variables for probabilistic time series modeling.',
    'tide': 'TIDE: Combines linear projections with additive encoder-decoder layers for dynamic feature integration.',
    'persistent': 'Persistent baseline: Predicts future values as the last observed value (no training required).',
    
    'd3vae': 'D3VAE: Combines VAE with diffusion processes for refined generation.',
    'vqvae': 'VQ-VAE: Uses vector quantization to discretize latent space for robust representations.',
    'vqvaea': 'VQ-VAE Autoencoder: Enhanced version with autoencoding capabilities.',
    'vva': 'VVA: Incorporates transformer-like tokenization and multi-head attention for feature extraction.'
};
