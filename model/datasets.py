import cv2
from random import uniform
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from sketchify.xdog_blend import blend_xdog_and_sketch, add_intensity

def pseudo_uniform(id, a, b):
    return (((id * 1.253 + a * 324.2351 + b * 534.342) * 20147.2312369804) + 0.12949) % (b - a) + a
def real_uniform(id, a, b):
    return uniform(a, b)

class SketchDataset(Dataset):
    def __init__(self, image_dir_path, file_id_list, iv_tag_list, 
            override_len=None, transform=None, is_train=True, **kwargs):
        self.image_dir_path = image_dir_path
        self.file_id_list = file_id_list
        self.iv_tag_list = iv_tag_list
        self.transform = transform
        self.data_len = len(file_id_list)
        self.is_train = is_train
        self.rand_gen = real_uniform if is_train else pseudo_uniform
        self.override_len = override_len

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        iv_tag_class = self.iv_tag_list[index]
        illust_path = self.image_dir_path / f"{file_id}.png"
        sketch_path = self.image_dir_path / f"{file_id}_sk.png"

        blend_img = self._sketch_blend(file_id, illust_path, sketch_path)
        sketch_img = Image.fromarray(blend_img)

        if self.transform is not None:
            sketch_img = self.transform(sketch_img)

        return (sketch_img, iv_tag_class)

    def __len__(self):
        if self.override_len > 0 and self.data_len > self.override_len:
            return self.override_len
        return self.data_len

    def _sketch_blend(self, file_id, illust_path, sketch_path):
        blend = self.rand_gen(file_id, -0.5, 0.25)
        rand_gen = self.rand_gen
        sketch = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)

        if blend > 0:
            illust = cv2.imread(str(illust_path))
            intensity = rand_gen(file_id, 1., 2.)
            degamma = rand_gen(file_id, 1./intensity, 1.)
            k = rand_gen(file_id, 2.0, 3.0)
            sigma = rand_gen(file_id, 0.35, 0.45)
            blend_img = blend_xdog_and_sketch(illust, sketch, intensity=intensity, degamma=degamma, k=k, sigma=sigma)
        else:
            intensity = rand_gen(file_id, 1., 1.3)
            blend_img = add_intensity(sketch, intensity)

        return blend_img


class ColorDataset(Dataset):
    def __init__(self, image_dir_path, file_id_list, cv_tag_list,
            override_len=None, transform=None, is_train=True, **kwargs):
        self.image_dir_path = image_dir_path
        self.file_id_list = file_id_list
        self.cv_tag_list = cv_tag_list
        self.transform = transform
        self.data_len = len(file_id_list)
        self.is_train = is_train
        self.rand_gen = real_uniform if is_train else pseudo_uniform
        self.override_len = override_len

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        cv_tag_class = self.cv_tag_list[index]
        illust_path = self.image_dir_path / f"{file_id}.png"

        color_img = Image.open(illust_path).convert('RGB')

        if self.transform is not None:
            color_img = self.transform(color_img)

        return (color_img, cv_tag_class)

    def __len__(self):
        if self.override_len > 0 and self.data_len > self.override_len:
            return self.override_len
        return self.data_len
