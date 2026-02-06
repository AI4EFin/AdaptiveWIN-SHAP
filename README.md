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

### Combine the frames with ffmpeg
MacOS
```bash
ffmpeg -i run_0_%04d.png -vf reverse -c:v hevc_videotoolbox -alpha_quality 0.75 -tag:v hvc1 run_0.mov
```

Other (untested)
```bash
ffmpeg -i run_0_%04d.png -vf "reverse,scale=iw*0.7:ih*0.7" -r 15 -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le run_0.mov
```

### Visualizing the robustness results
```bash
python examples/robustness/visualize_robustness_benchmark.py --growth geometric --dataset piecewise_ar3 --test-type lpa_sensitivity
```

Usage Examples

  # Run window detection with specific lambda
  python examples/lstm_simulation.py --dataset piecewise_ar3 --n0 50 --mc-reps 100 --penalty-factor 0.05

  # Run benchmark with matching lambda
  python examples/benchmark.py --dataset piecewise_ar3 --n0 50 --penalty-factor 0.05

  # Run robustness sensitivity analysis (tests multiple lambda values)
  python examples/robustness/01_lpa_sensitivity.py --datasets piecewise_ar3

  # Visualize results
  python examples/robustness/visualize_robustness_benchmark.py --test-type lpa_sensitivity --dataset
  piecewise_ar3 --growth geometric