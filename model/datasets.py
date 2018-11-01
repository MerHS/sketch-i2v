import cv2
from random import uniform
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from sketchify.sketchify import blend_xdog_and_sketch, add_intensity
def pseudorand_by_id(id):


class SketchDataset(Dataset):
    def __init__(self, image_dir_path, file_id_list, tag_class_list, transform=None, is_train=True):
        self.image_dir_path = image_dir_path
        self.file_id_list = file_id_list
        self.tag_class_list = tag_class_list
        self.transform = transform
        self.data_len = len(file_id_list)
        self.is_train = is_train

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        tag_class = self.tag_class_list[index]
        illust_path = self.image_dir_path / f"{file_id}.png"
        sketch_path = self.image_dir_path / f"{file_id}_sk.png"

        blend = uniform(-0.75, 0.25)
        sketch = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
        blend_img = None
        if blend > 0:
            illust = cv2.imread(str(illust_path))
            intensity = uniform(1., 2.)
            degamma = uniform(1./intensity, 1.)
            k = uniform(2.0, 3.0)
            sigma = uniform(0.35, 0.45)
            blend_img = blend_xdog_and_sketch(illust, sketch, intensity=intensity, degamma=degamma, k=k, sigma=sigma)
        else:
            intensity = uniform(1., 1.3)
            blend_img = add_intensity(sketch, intensity)

        sketch_img = Image.fromarray(blend_img)

        if self.transform is not None:
            sketch_img = self.transform(sketch_img)

        return (sketch_img, tag_class)

    def __len__(self):
        return self.data_len

class ColorAndSketchDataset(SketchDataset):
    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        tag_class = self.tag_class_list[index]
        image_path = self.image_dir_path / f"{file_id}.png"
        sketch_path = self.image_dir_path / f"{file_id}_sk.png"

        color_img = Image.open(image_path).convert('RGB')
        sketch_img = Image.open(sketch_path).convert('L') # to [1, H, W]

        if self.transform is not None:
            color_img = self.transform(color_img)
            sketch_img = self.transform(sketch_img)

        return (color_img, sketch_img, tag_class)