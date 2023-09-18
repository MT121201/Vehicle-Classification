## Vehicle Classification
Introducing a GitHub repository focused on vehicle classification, showcasing a nimble, real-time model.
This repository not only facilitates the creation of pseudo-datasets through image retrieval but also offers detailed guidance on labeling using CVAT.
It demonstrates how to leverage the pseudo-labeling tool for smooth model training and deployment.
The repository is specifically designed for classifying Vietnamese vehicles at toll booths, with images collected from a top-down perspective.


### Installation
- Refer to [docs/installation.md](docs/installation.md) for installing necessary libs.

### New Data 
- Refer to [docs/data.md](docs/data.md) for to know how to label and create new dataset.

### Usage
- Refer to [docs/train_test.md](docs/train_test.md) for to know how train and test the model.

### Deployment
- Refer to [docs/deployment.md](docs/deployment.md) for to know how convert model to ONNX.

### Inference 
- Refer to [docs/inference.md](docs/inference.md) for inference the model on video.


### Visualize trained dataset
```bash
python tools/visualizer/test_dataset.py \
    --config_file configs/tnivc/repvgg_vc.py \
    --save_img_dir ./cache/debugdata
```

