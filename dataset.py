from os import listdir
from os.path import join
import random
from PIL import Image ,ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file


def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray



def transf(im):
  im = im.resize((256, 256), Image.BICUBIC)

  # im= re_size(im)
  im=transforms.ToTensor()(im)
  return im  
def re_size(img):
  w,h =img.size
  if h%8!=0:
    h1=h-h%8
  else:
    h1=h

  if w%8!=0:
    w1=w- w%8
  else: 
    w1=w

  return img.resize((w1,h1))

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = image_dir#join(image_dir)
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
     


        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)



        a = a.resize((512,512), Image.BICUBIC)
        # a=re_size(a)
        # tar=re_size(tar)
        a = transform(a)

        
        return a, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)