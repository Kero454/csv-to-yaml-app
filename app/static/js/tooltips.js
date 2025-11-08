// Comprehensive Tooltip System for Technical Terms and Model Configurations
const tooltipDefinitions = {
    // Dataset Configuration
    'dataset': 'Dataset name and basic information',
    'path': 'File path where dataset is stored',
    'num_classes': 'Number of classes for classification tasks',
    
    // Scheduler Configuration
    'scheduler_config': 'Learning rate scheduling configuration',
    'scheduler_type': 'LR scheduler type (StepLR, CosineAnnealingLR, etc.)',
    'gamma': 'LR decay factor (e.g., 0.75 = 25% decrease)',
    'step_size': 'Steps between LR updates',
    
    // Optimizer Configuration
    'optim_config': 'Optimizer configuration',
    'optimizer_type': 'Algorithm (Adam, SGD, AdamW)',
    'learning_rate': 'Initial learning rate',
    'lr': 'Learning rate',
    'weight_decay': 'L2 regularization (0.0001-0.01)',
    'momentum': 'SGD momentum (typically ~0.9)',
    'betas': 'Adam beta coefficients',
    
    // Model Configuration
    'model_configs': 'Model architecture settings',
    'past_steps': 'Historical time steps for input',
    'future_steps': 'Future time steps to predict',
    'quantiles': 'Quantiles for probabilistic forecasting',
    'past_channels': 'Input channels (features)',
    'future_channels': 'Output channels',
    'embs': 'Categorical embedding size',
    'out_channels': 'Output dimension',
    'loss_type': 'Loss function (MSE, MAE, L1, KL, etc.)',
    'persistence_weight': 'Persistence model weight',
    
    // Model and Time Series Configuration
    'model': 'Model settings',
    'type': 'Architecture (rnn, linear, transformer, etc.)',
    'retrain': 'Train from scratch (true/false)',
    'ts': 'Time series data configuration',
    'name': 'Dataset identifier',
    'version': 'Config version',
    'enrich': 'Temporal features (["hour"], ["month"], etc.)',
    'use_covariates': 'Use external features',
    'use_future_covariates': 'Use future covariates',
    'silly': 'Diffusion model flag',
    
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
    
    // Training Configuration
    'train_config': 'Training parameters',
    'batch_size': 'Samples per batch',
    'max_epochs': 'Maximum training epochs',
    'gradient_clip_val': 'Gradient clipping value',
    'gradient_clip_algorithm': 'Clipping type (norm/value)',
    'precision': 'Training precision (32/16/bf16)',
    'modifier': 'Data loader modifier',
    'modifier_params': 'Modifier parameters',
    'num_workers': 'CPU data workers',
    'auto_lr_find': 'Auto find learning rate',
    'devices': 'GPU/CPU devices',
    'seed': 'Random seed',
    
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
    'verbose': 'Verbosity level (1 for detailed output).',
    '_target_': 'Specifies the target class for configuration.',
    'output_subdir': 'Set to null to prevent Hydra from saving an output subdirectory by default.',
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
    'dropout_rate': 'Rate of dropout for regularization during training (e.g., 0.1, 0.2, 0.5).',
    
    // RNN/LSTM Specific Parameters
    'hidden_RNN': 'RNN hidden size',
    'num_layers_RNN': 'Number of RNN layers',
    'n_layers_RNN': 'RNN layer count',
    'bidirectional': 'Bidirectional RNN',
    'lstm_type': 'LSTM variant',
    'kind': 'Model subtype (lstm, gru, etc.)',
    
    // Convolutional and General Architecture Parameters
    'kernel_size': 'Convolution kernel size (e.g., 3, 5).',
    'num_blocks': 'Number of blocks if the model is built with block modules.',
    'sum_emb': 'Whether to sum embeddings (often to combine multiple categorical features).',
    'optim': 'PyTorch optimizer to use. Usually torch.optim.Adam or torch.optim.SGD.',
    'activation': 'Activation function: torch.nn.PReLU, torch.nn.LeakyReLU, torch.nn.GELU, torch.nn.ReLU.',
    'remove_last': 'Whether to remove the last time step (or element) in sequences.',
    'use_bn': 'Whether to use Batch Normalization.',
    'hidden_size': 'Hidden dimension for linear or feedforward layers.',
    'simple': 'Used in linear models to indicate a simpler variant of the architecture.',
    
    // Transformer-based Parameters
    'd_model': 'Model dimension',
    'd_head': 'Attention head dimension',
    'n_head': 'Number of attention heads',
    'n_heads': 'Attention head count',
    'n_layer': 'Layer count',
    'n_layer_encoder': 'Encoder layers',
    'n_layer_decoder': 'Decoder layers',
    'factor': 'Scaling factor',
    'label_len': 'Label sequence length',
    'win_size': 'Window size',
    'seg_len': 'Segment length',
    
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
    
    // Split Parameters
    'keep_entire_seq_while_shifting': 'If true, retains the entire sequence during the shift operation.',
    
    // Logging Configuration
    'logging': 'Settings for tracking model training and validation.',
    'log_dir': 'Directory for storing logs of training metrics.',
    
    // Model Family Descriptions - RNN Family
    'rnn': 'Recurrent Neural Network family',
    'gru': 'Gated Recurrent Unit - simpler than LSTM',
    'lstm': 'Long Short-Term Memory network',
    'xlstm': 'Extended LSTM with enhanced features',
    
    // Linear Family
    'linear': 'Linear projection models',
    'dlinear': 'Decomposition-based linear model',
    'nlinear': 'Normalized linear model',
    'alinear': 'Attention-augmented linear',
    
    // Transformer Family
    'transformer': 'Self-attention based models',
    'tft': 'Temporal Fusion Transformer',
    'informer': 'Efficient attention for long sequences',
    'autoformer': 'Auto-decomposition transformer',
    'crossformer': 'Cross-correlation transformer',
    'patchtst': 'Patch-based time series transformer',
    'itransformer': 'Instance-wise transformer',
    
    // VAE Family
    'vae': 'Variational Autoencoders learn latent representations of data for probabilistic forecasting.',
    'd3vae': 'D3VAE: Combines VAE with diffusion processes for refined generation.',
    'vqvae': 'VQ-VAE: Uses vector quantization to discretize latent space for robust representations.',
    'vqvaea': 'VQ-VAE Autoencoder: Enhanced version with autoencoding capabilities.',
    'vva': 'VVA: Incorporates transformer-like tokenization and multi-head attention for feature extraction.',
    
    // Other Model Families
    'tdlgm': 'Temporal Deep Latent Generative Model: Uses bidirectional RNNs and latent variables for probabilistic time series modeling.',
    'tide': 'TIDE: Combines linear projections with additive encoder-decoder layers for dynamic feature integration.',
    'persistent': 'Persistent baseline: Predicts future values as the last observed value (no training required).',
    
    // Additional Architecture-specific Fields
    'num_preprocess_blocks': 'Number of preprocessing blocks in the architecture.',
    'num_preprocess_cells': 'Number of preprocessing cells per block.',
    'num_channels_enc': 'Number of channels in the encoder.',
    'arch_instance': 'Architecture instance type (e.g., "res_mbconv" for residual mobile convolution).',
    'num_latent_per_group': 'Number of latent variables per group.',
    'num_channels_dec': 'Number of channels in the decoder.',
    'groups_per_scale': 'Number of groups per scale in the architecture.',
    'num_postprocess_blocks': 'Number of postprocessing blocks.',
    'num_postprocess_cells': 'Number of postprocessing cells per block.'
};

function initializeTooltips() {
    // Track which terms have already been processed to avoid duplicates
    const processedTerms = new Set();
    
    // Find all elements that might contain technical terms
    // Exclude headers and buttons
    const textElements = document.querySelectorAll('label, span:not(.btn), td, th, p, li:not(.nav-item)');
    
    textElements.forEach(element => {
        // Skip if element already has tooltips
        if (element.querySelector('.tooltip-term')) return;
        
        // Skip elements that contain buttons, inputs, or form elements
        if (element.querySelector('button, input, select, textarea, a.btn')) return;
        
        // Skip elements that are part of file paths or contain URLs
        const isFilePath = element.textContent.includes('/') || element.textContent.includes('\\') || 
                          element.textContent.includes('.yaml') || element.textContent.includes('.yml') ||
                          element.textContent.includes('.csv') || element.textContent.includes('.py') ||
                          element.textContent.includes('.config') || element.textContent.includes('Config_main');
        if (isFilePath) return;
        
        // Skip code blocks and pre elements
        if (element.closest('code, pre')) return;
        
        // Only process text nodes to avoid breaking HTML structure
        processTextNodes(element, processedTerms);
    });
    
    setupTooltipPositioning();
}

function processTextNodes(element, processedTerms) {
    const walker = document.createTreeWalker(
        element,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );
    
    const textNodes = [];
    let node;
    while (node = walker.nextNode()) {
        if (node.nodeValue.trim()) {
            textNodes.push(node);
        }
    }
    
    textNodes.forEach(textNode => {
        let hasMatch = false;
        const content = textNode.nodeValue;
        
        // Check if this text contains any terms we want to add tooltips to
        for (const term in tooltipDefinitions) {
            // Skip if we've already added a tooltip for this term in the document
            if (processedTerms && processedTerms.has(term.toLowerCase())) continue;
            
            const regex = new RegExp(`\\b${term}\\b`, 'gi');
            if (regex.test(content)) {
                hasMatch = true;
                break;
            }
        }
        
        if (hasMatch) {
            // Create a wrapper span
            const wrapper = document.createElement('span');
            wrapper.innerHTML = content;
            
            // Replace each term with a tooltip span (only first occurrence)
            Object.keys(tooltipDefinitions).forEach(term => {
                // Only add tooltip if we haven't processed this term yet
                if (processedTerms && !processedTerms.has(term.toLowerCase())) {
                    const regex = new RegExp(`\\b(${term})\\b`, 'i');
                    if (regex.test(wrapper.innerHTML)) {
                        const replacement = `<span class="tooltip-term" data-tooltip="${tooltipDefinitions[term]}">$1<span class="tooltip-content">${tooltipDefinitions[term]}</span></span>`;
                        wrapper.innerHTML = wrapper.innerHTML.replace(regex, replacement);
                        processedTerms.add(term.toLowerCase());
                    }
                }
            });
            
            // Replace the text node with the new wrapper
            textNode.parentNode.replaceChild(wrapper, textNode);
        }
    });
}

function setupTooltipPositioning() {
    // Add dynamic positioning for tooltips near screen edges
    document.querySelectorAll('.tooltip-term').forEach(tooltip => {
        tooltip.addEventListener('mouseenter', function() {
            const rect = this.getBoundingClientRect();
            const tooltipContent = this.querySelector('.tooltip-content');
            
            // Reset classes
            this.classList.remove('tooltip-left', 'tooltip-right', 'tooltip-top');
            
            // Check if tooltip would go off screen
            if (rect.left < 200) {
                this.classList.add('tooltip-left');
            } else if (rect.right > window.innerWidth - 200) {
                this.classList.add('tooltip-right');
            }
            
            if (rect.top < 150) {
                this.classList.add('tooltip-top');
            }
        });
    });
}

// Initialize tooltips when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Delay initialization to ensure all other scripts have loaded
    setTimeout(initializeTooltips, 100);
});

// Re-initialize tooltips when new content is loaded dynamically
function refreshTooltips() {
    // Use setTimeout to avoid interfering with other scripts
    setTimeout(initializeTooltips, 100);
}

// Export for use in other scripts
window.refreshTooltips = refreshTooltips;

// Re-attach event listeners after tooltip initialization
function preserveEventListeners() {
    // Preserve form submission handlers
    document.querySelectorAll('form').forEach(form => {
        const clonedForm = form.cloneNode(false);
        while (form.firstChild) {
            clonedForm.appendChild(form.firstChild);
        }
        form.parentNode.replaceChild(clonedForm, form);
    });
}

// Ensure tooltips don't break existing functionality
window.safeTooltipInit = function() {
    // Save references to all forms and their submit handlers
    const forms = document.querySelectorAll('form');
    const formHandlers = new Map();
    
    forms.forEach(form => {
        if (form.onsubmit) {
            formHandlers.set(form, form.onsubmit);
        }
    });
    
    // Initialize tooltips
    initializeTooltips();
    
    // Restore form handlers
    forms.forEach(form => {
        if (formHandlers.has(form)) {
            form.onsubmit = formHandlers.get(form);
        }
    });
};
