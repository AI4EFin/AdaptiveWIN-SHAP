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
frames % ffmpeg -i run_0_%04d.png -vf reverse -c:v hevc_videotoolbox -alpha_quality 0.75 -tag:v hvc1 run_0.mov
```

Other (untested)
```bash
ffmpeg -i run_0_%04d.png -vf "reverse,scale=iw*0.7:ih*0.7" -r 15 -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le run_0.mov
```

### Visualizing the robustness results
```bash
python examples/robustness/visualize_robustness_benchmark.py --growth geometric --dataset piecewise_ar3 --test-type lpa_sensitivity
```