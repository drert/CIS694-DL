from torch.utils.data import Dataset, DataLoader

'''
The input to the model is expected to be a list of tensors, each of shape [C, H, W], 
        one for each image, and should be in 0-1 range. Different images can have different sizes.
The behavior of the model changes depending on if it is in training or evaluation mode.

During training, the model expects both the input tensors and targets (list of dictionary), containing:
        --- boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        --- labels (Int64Tensor[N]): the class label for each ground-truth box
        --- masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
'''

img_dict = {"img" : [], "img_name" : [], "img_pred" : [], "img_gt" : []} # dictionary for a single image


class MPII_Dataset(Dataset):
    def __init__(self, img_folder, annot_file, sample_size = 24987) : # max size is the 24987 total images
        
        self.imgs = random.sample(os.listdir(path),sample_size)
        self.size = sample_size
        self.path = path

    def __len__(self):
        return(len(self.imgs))

    def __getitem__(self, idx):
        if torch.is_tensor(idx) :
            idx = idx.tolist()

        if self.transform:
            gray = self.transform(gray)
            color = self.transform(color)
        sample = {'train':gray, 'truth':color, 'name':self.imgs[idx]}

        return sample