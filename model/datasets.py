from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class SketchDataset(Dataset):
    def __init__(self, image_dir_path, file_id_list, tag_class_list, transform=None):
        self.image_dir_path = image_dir_path
        self.file_id_list = file_id_list
        self.tag_class_list = tag_class_list
        self.transform = transform
        self.data_len = len(file_id_list)

    def __getitem__(self, index):
        file_id = self.file_id_list[index]
        tag_class = self.tag_class_list[index]
        sketch_path = self.image_dir_path / f"{file_id}_sk.png"

        sketch_img = Image.open(sketch_path).convert('L') # to [1, H, W]

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