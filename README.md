# Adaptive WIN-SHAP

This module encapsulates the implementation of the Adaptive WIN-SHAP algorithm.

To run locally
```bash
 pip install -e .
```

### Running the examples
```bash
python examples/lstm_simulation.py
```

### Reverse video with ffmpeg
```bash
ffmpeg -i run_0.mp4 -vf reverse run_0_rv.mp4
```

### Visualizing the robustness results
```bash
python examples/robustness/visualize_robustness_benchmark.py --growth geometric --dataset piecewise_ar3 --test-type lpa_sensitivity
```