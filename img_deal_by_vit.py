# @Author  : tiancn
# @Time    : 2022/10/16 18:48
from transformers import ViTImageProcessor, ViTConfig, ViTModel
import torch
import torchvision.transforms as transforms
import cv2 as cv
from tqdm import tqdm
import os
import pickle
from PIL import Image

deviceName = 'cuda:0'
device = torch.device(deviceName if torch.cuda.is_available() else 'cpu')

feature_extractor = ViTImageProcessor.from_pretrained("./data/HF/preprocessor_config.json")
#feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
config = ViTConfig.from_pretrained("./data/HF/config.json")

model = ViTModel.from_pretrained("./data/HF",
                                 config=config,
                                 local_files_only=True,
                                 use_safetensors=False  # 如果是 .bin 文件，设为 False
                                 )
model = model.to(device)



def imgdealByVit(img_path):
    img = cv.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None  # 或者跳过这张图
    transf = transforms.ToTensor()
    img_tensor = transf(img)
    inputs = feature_extractor(images=img_tensor,return_tensors = 'pt')
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.cpu()
    return last_hidden_states.squeeze()

if __name__ == '__main__':
    img_dir_arr = ['data/twitter2015_images','data/twitter2017_images']
    for img_dir in img_dir_arr:
        for filepath, dirnames, filenames in os.walk(img_dir):
            datas = {}
            count = 0
            for filename in tqdm(filenames, desc="deal data :" + img_dir):
                img_path = os.path.join(img_dir, filename)
                feature = imgdealByVit(img_path)
                datas[filename] = feature
            saveFileName = img_dir.split('/')[-1]
            with open('data/imgDealFile/'+ saveFileName + '.pkl', 'wb') as f:
                pickle.dump(datas, f)
    