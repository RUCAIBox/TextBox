# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

general_arguments = [
    'gpu_id', 'use_gpu', 'seed', 'reproducibility', 'state', 'data_path', 'checkpoint_dir', 'generated_text_dir'
]

training_arguments = [
    'epochs', 'train_batch_size', 'learner', 'learning_rate', 'eval_step', 'stopping_step', 'grad_clip',
    'g_pretraining_epochs', 'd_pretraining_epochs', 'd_sample_num', 'd_sample_training_epochs',
    'adversarail_training_epochs', 'adversarail_g_epochs', 'adversarail_d_epochs'
]

evaluation_arguments = ['beam_size', 'decoding_strategy', 'metrics', 'n_grams', 'eval_batch_size', 'eval_generate_num']

dataset_arguments = [
    'max_vocab_size', 'source_max_vocab_size', 'target_max_vocab_size', 'source_max_seq_length',
    'target_max_seq_length', 'source_language', 'target_language', 'source_suffix', 'target_suffix', 'split_strategy',
    'split_ratio', 'share_vocab', 'task_type'
]
