from dataclasses import dataclass
from typing import Optional, Type
import os
import numpy as np
import pandas as pd

@dataclass
class RunConfig:
    # Mirrors the arguments used in the original main
    seq_len: int = 3
    n_0: int = 50
    jump: int = 1
    search_step: int = 10
    alpha: float = 0.95
    num_bootstrap: int = 1
    epochs: int = 100

class AdaptiveWINSHAP:
    """
    Thin orchestrator around the unchanged notebook code. Optionally accepts a
    custom model class by overriding the symbol "LSTMRegressor" inside the legacy
    module at runtime (no changes to original logic).
    """

    def __init__(self, model_class: Optional[Type] = None):
        if model_class is not None:
            # Monkey-patch the symbol used by the legacy functions
            legacy.LSTMRegressor = model_class

    def run(self,
            data_1d: np.ndarray,
            output_csv_path: str,
            config: Optional[RunConfig] = None) -> pd.DataFrame:
        if config is None:
            config = RunConfig()

        df_out, _ = legacy.detect_changes_with_lstm(
            data_1d,
            seq_len=config.seq_len,
            n_0=config.n_0,
            jump=config.jump,
            search_step=config.search_step,
            alpha=config.alpha,
            num_bootstrap=config.num_bootstrap,
            epochs=config.epochs,
        )

        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
        df_out.to_csv(output_csv_path, index=False)
        return df_out
