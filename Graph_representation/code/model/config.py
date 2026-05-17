# 模型配置
MODEL_CONFIG = {
    'herb_num': 563,
    'gene_num': 2106,
    'factor_num': 64,
    'dropout_rate': 0.1,
    'num_heads': 8,
    'num_gcn_layers': 4,
    'attention_layers': [1, 3],  # 在哪些层应用attention
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 2048,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'max_epochs': 500,
    'patience': 30,
    'eval_interval': 5,
    'save_interval': 50,
    'grad_clip': 1.0,
}

# 排序优化配置
RANKING_CONFIG = {
    'top_k_list': [1, 5, 10, 20],
    'margin': 1.0,
    'temperature': 0.1,
    'hard_negative_ratio': 0.3,
    'random_negative_ratio': 0.7,
}

# 损失权重配置
LOSS_WEIGHTS = {
    'ranking': 1.0,
    'l2': 0.001,
    'diversity': 0.001,
}

# 数据配置
DATA_CONFIG = {
    'num_negative_samples': 5,
    'num_workers': 4,
    'pin_memory': True,
}