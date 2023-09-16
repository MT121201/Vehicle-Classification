import onnxruntime as rt
import numpy as np
from PIL import Image
import argparse


def load_img(img_path):
    img = Image.open(img_path)
    # Resize to (224, 224)
    img = img.resize((224, 224), Image.BILINEAR)
    # Normalize to [0.0, 1.0]
    img = np.array(img).astype(np.float32) / 255.0
    # HWC to CHW format:
    img = np.transpose(img, [2, 0, 1])
    # NCHW format:
    img = np.expand_dims(img, axis=0)
    return img

def load_model(onnx_file):
    provider = 'CUDAExecutionProvider' if rt.get_device() == 'GPU' else 'CPUExecutionProvider'
    session = rt.InferenceSession(onnx_file, providers=[provider])
    return session

def get_top_prediction(result):
    # Get top-1 prediction
    prediction = np.argmax(result[0], axis=1)
    score = result[0][0][prediction]
    return prediction, score

def main(onnx_path, img_path, class_names):
    # Load image
    img = load_img(img_path)
    # Load model
    session = load_model(onnx_path)
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img})
    # Get top prediction
    prediction, score = get_top_prediction(result)
    
    # Print top prediction
    print('Prediction: {}, score: {:.2f}%'.format(class_names[prediction[0]], score[0] * 100))
    
def arg_parse():
    parser = argparse.ArgumentParser(description="Inference onnx model")
    parser.add_argument("--img", help="Path to the input image")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    onnx_path = "/checkpoints/vehicle_cls/vehicle_cls.onnx"
    img_path = args.img
    class_names = ['xe sedan', 'xe SUV', 'xe ban tai', 'xe ba gac', 'xe tai nho', 
                 'xe tai lon', 'xe container', 'xe may xuc, may cau, xe lu, xe cho be tong', 
                 'xe 16 cho', 'xe 29-32 cho', 'xe 52 cho']
    main(onnx_path, img_path, class_names)