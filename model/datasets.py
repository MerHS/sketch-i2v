import cv2
from random import uniform, randint, choice
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from sketchify.xdog_blend import blend_xdog_and_sketch, add_intensity, get_xdog_image

def pseudo_uniform(id, a, b):
    return (((id * 1.253 + a * 324.2351 + b * 534.342) * 20147.2312369804) + 0.12949) % (b - a) + a
def real_uniform(id, a, b):
    return uniform(a, b)

class MultiImageDataset(Dataset):
    def __init__(self, image_dir_path_list, file_id_list, tag_list, 
            override_len=0, transform=None, is_color=False, **kwargs):
        self.image_dir_path_list = image_dir_path_list
        self.file_id_list = file_id_list
        self.tag_list = tag_list
        self.transform = transform
        self.data_len = len(file_id_list)
        self.override_len = override_len
        self.conv_arg = 'RGB' if is_color else 'L'

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        tag_class = self.tag_list[index]
        img_dir_path = choice(self.image_dir_path_list)

        img_path = img_dir_path / f"{file_id}.png"
        img = Image.open(str(img_path)).convert(self.conv_arg)
        if self.transform is not None:
            img = self.transform(img)

        return (img, tag_class)

    def __len__(self):
        if self.override_len > 0 and self.data_len > self.override_len:
            return self.override_len
        return self.data_len

class RawSketchDataset(Dataset):
    def __init__(self, image_dir_path, file_id_list, iv_tag_list, 
            override_len=0, transform=None, **kwargs):
        self.image_dir_path = image_dir_path
        self.file_id_list = file_id_list
        self.iv_tag_list = iv_tag_list
        self.transform = transform
        self.data_len = len(file_id_list)
        self.override_len = override_len

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        iv_tag_class = self.iv_tag_list[index]
        sketch_path = self.image_dir_path / f"{file_id}.png"
        sketch_img = Image.open(str(sketch_path)).convert('L')
        if self.transform is not None:
            sketch_img = self.transform(sketch_img)

        return (sketch_img, iv_tag_class)

    def __len__(self):
        if self.override_len > 0 and self.data_len > self.override_len:
            return self.override_len
        return self.data_len

class SketchRawXDogDataset(Dataset):
    def __init__(self, image_dir_path, file_id_list, iv_tag_list, 
            override_len=None, transform=None, is_train=True, **kwargs):
        self.image_dir_path = image_dir_path
        self.file_id_list = file_id_list
        self.iv_tag_list = iv_tag_list
        self.transform = transform
        self.data_len = len(file_id_list)
        self.is_train = is_train
        self.override_len = override_len

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        iv_tag_class = self.iv_tag_list[index]
        sketch_path = self.image_dir_path / f"{file_id}_sk.png"

        # if self.is_train:
        #     rgb_path = self.image_dir_path / f"{file_id}.png"
        #     if rgb_path.exists():
        #         sketch_img = self._get_rand_xdog(sketch_path, rgb_path)
        #     else:
        #         sketch_img = Image.open(str(sketch_path)).convert('L')
        # else:
        #     sketch_img = Image.open(str(sketch_path)).convert('L')
        sketch_img = Image.open(str(sketch_path)).convert('L')
        if self.transform is not None:
            sketch_img = self.transform(sketch_img)

        return (sketch_img, iv_tag_class)

    def __len__(self):
        if self.override_len > 0 and self.data_len > self.override_len:
            return self.override_len
        return self.data_len
    
    def _get_rand_xdog(self, sketch_path, rgb_path):
        r = randint(0, 5)
        if r >= 5:
            grey_img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
            k = uniform(2.0, 3.0)
            sigma = uniform(0.35, 0.45)
            result = get_xdog_image(grey_img, k=k, sigma=sigma, gamma=0.98)

            # if r == 4:
            #     blend = uniform(0.2, 0.8)
            #     liners = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
            #     if liners.shape != result.shape:
            #         result = cv2.resize(result, liners.shape, interpolation=cv2.INTER_AREA)
            #     result = cv2.addWeighted(result, blend, liners, 1-blend, 0)

        if r <= 4:
            result = Image.open(str(sketch_path)).convert('L')
        else:
            result = Image.fromarray(result)

        return result

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
