import onnxruntime as rt
import numpy as np
from PIL import Image
import argparse
import os
import importlib.util
import cv2

class Vehicle_classifier:
    def __init__(self,
                 model_path,
                 device,
                 threshold,
                 mean,
                    std,
                 ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.mean = np.array(mean)
        self.std = np.array(std)
        # Build model with the specified device (CPU or GPU)
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device == 'gpu':
            providers = ['CUDAExecutionProvider']
        else:
            raise ValueError("Invalid device choice. Use 'cpu' or 'gpu'.")
        self.sess = rt.InferenceSession(self.model_path, providers=providers)
    
    def crop_detect_images(self, img, result):
        # crop image
        x1, y1, x2, y2 = result[:4]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img.shape[1], int(x2))
        y2 = min(img.shape[0], int(y2))
        crop_img = img[y1:y2, x1:x2]
        return crop_img

    def preprocess(self, img):
        # When reading image from cv2, it is BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        np_img = np.array(im_pil)
        # normalize image as training
        np_img = (np_img-self.mean)/self.std
        # convert to float32
        np_img = np_img.transpose(2, 0, 1).astype(np.float32)
        input_batch = np.expand_dims(np_img, 0)
        return input_batch
    
    def predict(self, img):
        # preprocess image
        input_batch = self.preprocess(img)
        # run inference
        result = self.sess.run(None, {self.sess.get_inputs()[0].name: input_batch})
        # top 1 prediction
        prediction = np.argmax(result[0], axis=1)
        score = result[0][0][prediction]
        return prediction, score


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
    classifier = Vehicle_classifier(onnx_path, 'gpu', 0.8, mean, std)
    img = cv2.imread(img_path)
    prediction, score = classifier.predict(img)
    print(class_names[prediction[0]], score[0])
