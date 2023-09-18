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
- check the config in file "configs/inference/onnx_infer.py"
```bash
python tools/inference/onnx_native.py
    --config path to config file 
    --img image path to inference model
```
