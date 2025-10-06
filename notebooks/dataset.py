import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224

EMOTION_TO_ID = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}


class FER2013Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pix = np.array(row['pixels'].split(), dtype=np.uint8)
        img = Image.fromarray(pix.reshape(48, 48)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = EMOTION_TO_ID[row['emotion']]
        return img, label