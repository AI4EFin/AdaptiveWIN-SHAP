"""
Benchmarking utilities for comparing SHAP explanation methods.
"""

from .metrics import (
    xai_eval_fnc,
    compute_faithfulness_metrics,
    compute_point_faithfulness,
    evaluate_explanation_quality
)

from .baselines import (
    GlobalSHAP,
    RollingWindowSHAP,
    TimeShapWrapper,
    LSTMModel,
    create_sequences,
    train_lstm_model
)

__all__ = [
    'xai_eval_fnc',
    'compute_faithfulness_metrics',
    'compute_point_faithfulness',
    'evaluate_explanation_quality',
    'GlobalSHAP',
    'RollingWindowSHAP',
    'TimeShapWrapper',
    'LSTMModel',
    'create_sequences',
    'train_lstm_model'
]
