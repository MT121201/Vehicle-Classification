### Installation
```bash
pip uninstall onnxruntime
python -m pip install onnxruntime-gpu==1.15.1
```
### Test install
```bash
python
import onnxruntime as rt
rt.get_device()
#>> "GPU"

# If this raise ERROR uninstall onnxruntime-gpu and install again
pip uninstall onnxruntime-gpu
python -m pip install onnxruntime-gpu==1.15.1
```
### Inference
- model path : /checkpoints/vehicle_cls/vehicle_cls.onnx
```bash
python tools/inference/onnx_native.py
--img image path to inference model
```
