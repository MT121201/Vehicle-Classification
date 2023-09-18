# path to onnx model
model = "/checkpoints/vehicle_cls/pretrain_repvgg.onnx"
class_names = ['xe sedan', 'xe SUV', 'xe ban tai', 'xe ba gac', 'xe tai nho',
                'xe tai lon', 'xe container', 'xe may xuc, may cau, xe lu, xe cho be tong',
                'xe 16 cho', 'xe 29-32 cho', 'xe 52 cho']
# mean and std
mean = [0, 0, 0]
std=[255, 255, 255]
# input size of model
size = (224, 224)