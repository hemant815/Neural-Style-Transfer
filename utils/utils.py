import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_PIXELS = 100_000_000  # 100 million pixels

ImageFile.LOAD_TRUNCATED_IMAGES = True
class ImageFolderDataset(Dataset):
    def __init__(self,root,transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [
            p for p in self.files
            if p.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.files)
    
    



    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])

        try:
            image = Image.open(image_path)

            # Skip huge images
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(
                    f"Image too large: {image.width}x{image.height}"
                )

            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading {image_path}: {e}")

            return self.__getitem__((idx + 1) % len(self.files))


def Adain(content_feature, style_feature):
    #[batch size, channel, h , w]
    size = content_feature.size()
    style_mean, style_std = calc_mean_std(style_feature)
    content_mean, content_std = calc_mean_std(content_feature)

    norm_content_feats = (content_feature - content_mean.expand(size))/content_std.expand(size)

    return norm_content_feats * style_std.expand(size) + style_mean.expand(size)


    pass

def calc_mean_std(feats, eps=1e-5):
    #[batch size, channel, h , w]
    
    size = feats.size()
    assert (len(size) ==4)

    batch_size, channel=size[:2]
    feats_mean =feats.view(batch_size,channel,-1).mean(dim=2).view(batch_size,channel,1,1)
    feats_var = feats.view(batch_size,channel,-1).var(dim=2, unbiased=False) + eps
    feats_std = feats_var.sqrt().view(batch_size, channel,1,1)
    return feats_mean, feats_std
