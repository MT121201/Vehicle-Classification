import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import os

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine, batch_size=-1):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype(np.float32) /255.0
    # HWC to NCHW format:
    img = np.transpose(img, [2, 0, 1])
    # NCHW format:
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output):
    return np.argmax(output)

def main(engine_file, image_path, class_names):
    engine = load_engine(engine_file)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    img = preprocess(image_path)
    context.set_binding_shape(0, img.shape)
    np.copyto(inputs[0]['host'], img.ravel())

    # Do inference
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # Postprocess
    output = outputs[0]['host']

    # Print top prediction, score
    print('Prediction: {}, score: {:.2f}%'.format(class_names[postprocess(output)], output[postprocess(output)] * 100))

if __name__ == '__main__':
    engine_file = 'workdir/tnivc/rt/end2end.engine'
    image_path = 'workdir/tnivc/20230721113738260_113_2.jpg'
    class_names = ['xe sedan', 'xe SUV', 'xe ban tai', 'xe ba gac', 'xe tai nho', 
                 'xe tai lon', 'xe container', 'xe may xuc, may cau, xe lu, xe cho be tong', 
                 'xe 16 cho', 'xe 29-32 cho', 'xe 52 cho']
    main(engine_file, image_path, class_names)
