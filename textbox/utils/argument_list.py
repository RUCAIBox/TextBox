general_arguments = [
    'gpu_id', 'use_gpu', 'seed', 'reproducibility', 'state', 'data_path', 'checkpoint_dir', 'generated_text_dir'
]

training_arguments = [
    'epochs', 'train_batch_size', 'optimizer', 'learning_rate', 'eval_step', 'stopping_step', 'grad_clip',
    'g_pretraining_epochs', 'd_pretraining_epochs', 'd_sample_num', 'd_sample_training_epochs',
    'adversarail_training_epochs', 'adversarail_g_epochs', 'adversarail_d_epochs', 'scheduler', 'init_lr',
    'warmup_steps', 'max_steps', 'weight_decay', 'epsilon', 'adam_beta1', 'adam_beta2'
]

evaluation_arguments = ['beam_size', 'decoding_strategy', 'metrics', 'n_grams', 'eval_batch_size']
