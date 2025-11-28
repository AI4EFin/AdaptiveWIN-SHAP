"""
Evaluation metrics for time-series importance scores.

Implements perturbation analysis and sequence analysis metrics
for evaluating the quality of SHAP explanations on time series models.
"""

import math
import numpy as np
from copy import deepcopy


def xai_eval_fnc(model, relevance, input_x, model_type='lstm', percentile=90,
                 eval_type='prtb', seq_len=10, by='all'):
    """
    Evaluates the quality metrics of time-series importance scores using various evaluation methods.

    Parameters
    ----------
    model : prediction model that is explained
    relevance : A 3D array of importance scores for each time step of the time-series data
    input_x : input data of the prediction model. If the input data consists of different modalities,
              the first module should be a 3D time series data
    model_type : type of model, either 'lstm' or 'lstm_plus'. Use 'lstm' when the time series data
                 is the only modality of the input, otherwise use 'lstm_plus'
    percentile : percentile of top time steps that are going to be perturbed
    eval_type : evaluation method, either 'prtb' for the perturbation analysis metric or
                'sqnc' for sequence analysis metric
    seq_len : sequence length for 'sqnc' method
    by : whether to evaluate each temporal feature separately or all time steps together,
         either 'time' or 'all'

    Returns
    -------
    predictions : prediction of the modified input time-series data using the input model
    """

    input_new = deepcopy(input_x)
    relevance = np.absolute(relevance)

    # Handle different model input types
    if model_type == 'lstm_plus':
        input_ts = input_x[0]
        input_new_ts = input_new[0]
    elif model_type == 'lstm':
        input_ts = input_x
        input_new_ts = input_new

    assert len(input_ts.shape) == 3, "The time series data needs to be 3-dimensional [N, T, F]"
    num_instance = input_ts.shape[0]
    num_time_step = input_ts.shape[1]
    num_feature = input_ts.shape[2]

    # Find top important time steps
    if by == 'time':
        # Find top steps for each feature separately
        top_steps = math.ceil((1 - percentile/100) * num_time_step)
        top_indices = np.argsort(relevance, axis=1)[:, -top_steps:, :]
        # Convert to flattened indices
        for j in range(num_feature):
            top_indices[:, :, j] = top_indices[:, :, j] * num_feature + j
        top_indices = top_indices.flatten()
    elif by == 'all':
        # Find top steps across all features
        top_steps = math.ceil((1 - percentile/100) * num_time_step * num_feature)
        top_indices = np.argsort(relevance, axis=None)[-top_steps:]

    # Create a masking matrix for top time steps
    top_indices_mask = np.zeros(input_ts.size)
    top_indices_mask[top_indices] = 1
    top_indices_mask = top_indices_mask.reshape(input_ts.shape)

    # Apply perturbation based on evaluation type
    for p in range(num_instance):
        for v in range(num_feature):
            for t in range(num_time_step):
                if top_indices_mask[p, t, v]:
                    if eval_type == 'prtb':
                        # Perturbation: flip the value
                        input_new_ts[p, t, v] = np.max(input_ts[p, :, v]) - input_ts[p, t, v]
                    elif eval_type == 'sqnc':
                        # Sequence: zero out sequence starting at this point
                        input_new_ts[p, t:t + seq_len, v] = 0

    return model.predict(input_new)


def compute_faithfulness_metrics(original_preds, perturbed_preds):
    """
    Compute faithfulness metrics comparing original and perturbed predictions.

    The idea is that perturbing the most important features should cause
    the largest change in predictions. A higher change indicates better faithfulness.

    Parameters
    ----------
    original_preds : array-like
        Original model predictions
    perturbed_preds : array-like
        Predictions after perturbing important features

    Returns
    -------
    dict : Dictionary containing various faithfulness metrics
    """
    original_preds = np.asarray(original_preds).flatten()
    perturbed_preds = np.asarray(perturbed_preds).flatten()

    # Mean absolute change
    mae = np.mean(np.abs(original_preds - perturbed_preds))

    # Mean squared change
    mse = np.mean((original_preds - perturbed_preds) ** 2)

    # Relative change (normalized by original prediction magnitude)
    relative_change = np.mean(np.abs(original_preds - perturbed_preds) /
                              (np.abs(original_preds) + 1e-10))

    # Correlation (lower correlation = better, since perturbation should change predictions)
    correlation = np.corrcoef(original_preds, perturbed_preds)[0, 1]

    return {
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'relative_change': relative_change,
        'correlation': correlation,
        'faithfulness_score': mae  # Higher is better - more change means more faithful
    }


def compute_point_faithfulness(model, shap_values, input_sequence,
                                percentiles=[90, 70, 50],
                                eval_types=['prtb', 'sqnc'],
                                seq_len=10):
    """
    Compute faithfulness metrics for a single prediction point.

    This function evaluates a single explanation by perturbing the input
    based on SHAP importance and measuring prediction change.

    Parameters
    ----------
    model : prediction model
        The model that produced the SHAP values (must have predict method)
    shap_values : array-like
        SHAP values for this point, shape [seq_len] or [seq_len, features]
    input_sequence : array-like
        Input sequence for this point, shape [seq_len, features] or [1, seq_len, features]
    percentiles : list of int
        Percentiles of top features to perturb
    eval_types : list of str
        Evaluation types: 'prtb' (perturbation) or 'sqnc' (sequence)
    seq_len : int
        Sequence length for 'sqnc' method

    Returns
    -------
    dict : Dictionary with keys like 'faithfulness_prtb_p90', 'faithfulness_sqnc_p50', etc.
    """
    # Ensure correct shapes
    shap_values = np.asarray(shap_values)
    input_sequence = np.asarray(input_sequence)

    # Handle different input shapes
    if input_sequence.ndim == 2:
        # [seq_len, features] -> [1, seq_len, features]
        input_sequence = input_sequence[None, :, :]
    elif input_sequence.ndim == 3 and input_sequence.shape[0] != 1:
        # Take first instance if batch
        input_sequence = input_sequence[0:1, :, :]

    # Flatten SHAP values if needed [seq_len, features] -> [seq_len * features]
    # Ensure it's always at least 1D (handles scalar case for seq_length=1)
    shap_flat = np.abs(np.atleast_1d(shap_values)).flatten()

    # Get original prediction
    original_pred = model.predict(input_sequence)

    results = {}

    for percentile in percentiles:
        for eval_type in eval_types:
            # Create perturbed version
            input_perturbed = deepcopy(input_sequence)

            # Determine number of top features to perturb
            num_features = len(shap_flat)
            top_k = max(1, int(np.ceil((1 - percentile/100) * num_features)))

            # Get indices of top important features
            top_indices = np.argsort(shap_flat)[-top_k:]

            # Apply perturbation
            num_time_steps = input_sequence.shape[1]
            num_feature_dims = input_sequence.shape[2]

            for flat_idx in top_indices:
                # Convert flat index back to (time, feature) indices
                t_idx = flat_idx // num_feature_dims
                f_idx = flat_idx % num_feature_dims

                if t_idx < num_time_steps:
                    if eval_type == 'prtb':
                        # Perturbation: flip value
                        max_val = np.max(input_sequence[0, :, f_idx])
                        input_perturbed[0, t_idx, f_idx] = max_val - input_sequence[0, t_idx, f_idx]
                    elif eval_type == 'sqnc':
                        # Sequence: zero out from this point forward
                        end_idx = min(t_idx + seq_len, num_time_steps)
                        input_perturbed[0, t_idx:end_idx, f_idx] = 0

            # Get perturbed prediction
            perturbed_pred = model.predict(input_perturbed)

            # Compute metrics
            mae = np.abs(original_pred - perturbed_pred).mean()
            key = f'faithfulness_{eval_type}_p{percentile}'
            results[key] = float(mae)

    return results


def compute_point_ablation(model, shap_values, input_sequence,
                           percentiles=[90, 70, 50],
                           ablation_types=['mif', 'lif']):
    """
    Compute ablation metrics for a single prediction point.

    Ablation systematically removes features in order of importance and measures
    the cumulative effect on predictions. This is considered one of the most
    rigorous XAI evaluation metrics.

    Parameters
    ----------
    model : prediction model
        The model that produced the SHAP values (must have predict method)
    shap_values : array-like
        SHAP values for this point, shape [seq_len] or [seq_len, features]
    input_sequence : array-like
        Input sequence for this point, shape [seq_len, features] or [1, seq_len, features]
    percentiles : list of int
        Percentiles of features to remove (e.g., 90 = remove top 10%)
    ablation_types : list of str
        Ablation strategies:
        - 'mif': Most Important First (remove most important features first)
        - 'lif': Least Important First (remove least important features first)

    Returns
    -------
    dict : Dictionary with keys like 'ablation_mif_p90', 'ablation_lif_p50', etc.
           Values represent the average prediction change as features are removed.
    """
    # Ensure correct shapes
    shap_values = np.asarray(shap_values)
    input_sequence = np.asarray(input_sequence)

    # Handle different input shapes
    if input_sequence.ndim == 2:
        # [seq_len, features] -> [1, seq_len, features]
        input_sequence = input_sequence[None, :, :]
    elif input_sequence.ndim == 3 and input_sequence.shape[0] != 1:
        # Take first instance if batch
        input_sequence = input_sequence[0:1, :, :]

    # Flatten SHAP values if needed [seq_len, features] -> [seq_len * features]
    # Ensure it's always at least 1D (handles scalar case for seq_length=1)
    shap_flat = np.abs(np.atleast_1d(shap_values)).flatten()

    # Get original prediction
    original_pred = model.predict(input_sequence)

    results = {}
    num_time_steps = input_sequence.shape[1]
    num_feature_dims = input_sequence.shape[2]

    for percentile in percentiles:
        # Determine number of features to ablate
        num_features = len(shap_flat)
        top_k = max(1, int(np.ceil((1 - percentile/100) * num_features)))

        for ablation_type in ablation_types:
            # Get indices of features to remove
            if ablation_type == 'mif':
                # Most important first: sort descending
                sorted_indices = np.argsort(shap_flat)[::-1][:top_k]
            elif ablation_type == 'lif':
                # Least important first: sort ascending
                sorted_indices = np.argsort(shap_flat)[:top_k]
            else:
                raise ValueError(f"Unknown ablation_type: {ablation_type}")

            # Iteratively remove features and measure cumulative effect
            cumulative_changes = []
            input_ablated = deepcopy(input_sequence)

            for i, flat_idx in enumerate(sorted_indices):
                # Convert flat index back to (time, feature) indices
                t_idx = flat_idx // num_feature_dims
                f_idx = flat_idx % num_feature_dims

                if t_idx < num_time_steps:
                    # Remove this feature by setting to 0 (standard ablation)
                    input_ablated[0, t_idx, f_idx] = 0

                    # Get prediction with this feature removed
                    ablated_pred = model.predict(input_ablated)

                    # Measure absolute change from original
                    change = np.abs(original_pred - ablated_pred).mean()
                    cumulative_changes.append(change)

            # Compute average degradation (area under curve normalized)
            if len(cumulative_changes) > 0:
                avg_degradation = np.mean(cumulative_changes)
            else:
                avg_degradation = 0.0

            key = f'ablation_{ablation_type}_p{percentile}'
            results[key] = float(avg_degradation)

    return results


def evaluate_explanation_quality(model, shap_values, input_sequences,
                                  percentiles=[90, 70, 50],
                                  eval_types=['prtb', 'sqnc'],
                                  seq_len=10):
    """
    Comprehensive evaluation of explanation quality using multiple metrics.

    Parameters
    ----------
    model : prediction model
    shap_values : array of shape [N, T, F] - SHAP values for each instance
    input_sequences : array of shape [N, T, F] - original input sequences
    percentiles : list of percentiles to test
    eval_types : list of evaluation types ('prtb', 'sqnc')
    seq_len : sequence length for 'sqnc' method

    Returns
    -------
    results : dict of results for each combination of percentile and eval_type
    """
    # Get original predictions
    original_preds = model.predict(input_sequences)

    results = {}

    for percentile in percentiles:
        for eval_type in eval_types:
            key = f"{eval_type}_p{percentile}"

            # Get perturbed predictions
            perturbed_preds = xai_eval_fnc(
                model=model,
                relevance=shap_values,
                input_x=input_sequences,
                model_type='lstm',
                percentile=percentile,
                eval_type=eval_type,
                seq_len=seq_len,
                by='all'
            )

            # Compute metrics
            metrics = compute_faithfulness_metrics(original_preds, perturbed_preds)
            results[key] = metrics

    return results
