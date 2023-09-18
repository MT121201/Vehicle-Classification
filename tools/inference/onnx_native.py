import onnxruntime as rt
import numpy as np
from PIL import Image
import argparse
import os
import importlib.util

def load_img(img_path, mean, std, resize):    
    img = Image.open(img_path)
    # Resize to (224, 224)
    img = img.resize(resize, Image.BILINEAR)
    # Normalize to [0.0, 1.0]
    img = np.array(img)
    # Normalize
    img = (img - mean) / std
    # HWC to CHW format:
    img = np.transpose(img, [2, 0, 1]).astype(np.float32)
    # NCHW format:
    img = np.expand_dims(img, axis=0)
    return img


def load_model(onnx_file):
    provider = 'CUDAExecutionProvider' if rt.get_device(
    ) == 'GPU' else 'CPUExecutionProvider'
    session = rt.InferenceSession(onnx_file, providers=[provider])
    return session


def get_top_prediction(result):
    # Get top-1 prediction
    prediction = np.argmax(result[0], axis=1)
    score = result[0][0][prediction]
    return prediction, score


def main(onnx_path, img_path, class_names, mean, std, resize):
    # Load image
    img = load_img(img_path, mean, std, resize)
    # Load model
    session = load_model(onnx_path)
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img})
    # Get top prediction
    prediction, score = get_top_prediction(result)

    # Print top prediction
    print('Prediction: {}, score: {:.2f}%'.format(
        class_names[prediction[0]], score[0] * 100))


def arg_parse():
    parser = argparse.ArgumentParser(description="Inference onnx model")
    parser.add_argument("--config", help="Path to the config file")
    parser.add_argument("--img", help="Path to the input image")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    config_spec = importlib.util.spec_from_file_location('config', args.config)
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    onnx_path = config_module.model
    class_names = config_module.class_names
    mean = np.array(config_module.mean)
    std = np.array(config_module.std)
    resize = config_module.size
    img_path = args.img

    main(onnx_path, img_path, class_names, mean, std, resize)
