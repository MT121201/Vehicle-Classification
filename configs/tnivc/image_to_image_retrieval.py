from mmpretrain import ImageRetrievalInferencer
import os


input_img  = '/home/tni/Workspace/triet/Vehicle-Classification/test_imgs/query/20230714081536846_113_0.jpg'
gallery_path  = '/home/tni/Workspace/triet/Vehicle-Classification/test_imgs/gallery'

# take image list from gallery_path
list_gallery = []
for image in os.listdir(gallery_path):
    if image.endswith('.jpg'):
        list_gallery.append(os.path.join(gallery_path, image))
    if image.endswith('.png'):
        list_gallery.append(os.path.join(gallery_path, image))

inferencer = ImageRetrievalInferencer('resnet50-arcface_inshop', prototype=list_gallery)
predict = inferencer(input_img)[0] 
print(predict)