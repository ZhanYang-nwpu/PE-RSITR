import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import clip
import numpy as np

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.loc = opt.data_path
        self.img_path = opt.image_path
        self.datatype = opt.datatype
        self.preprocess = opt.preprocess

        # Captions
        self.captions = []
        self.maxlength = 0

        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1


        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.RandomCrop(224),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                ])
    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index / self.im_div)
        caption_out = self.captions[index]
        caption_out = str(caption_out)[2:-1]
        image = Image.open(self.img_path + str(self.images[img_id])[2:-1])

        # Convert caption (string) to word ids.
        caption = clip.tokenize(caption_out).squeeze(0)
        image = self.preprocess(image)

        word_id, word_mask, labelID = 0., 0., 0.
        return image, caption, caption_out, index, img_id, word_id, word_mask, labelID

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    # data.sort(key=lambda x: sum(x[-1]), reverse=True)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, caption_out, ids, img_ids, word_id, word_mask, labelID = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l != 0 else 1 for l in lengths]
    ids = np.array(ids)

    return images, targets, lengths, caption_out, ids, img_ids, word_mask, labelID

def get_precomp_loader(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader

def get_loaders(vocab, opt):
    train_loader = get_precomp_loader( 'train', vocab,
                                      opt.batch_size, True, opt.workers, opt=opt)
    val_loader = get_precomp_loader( 'val', vocab,
                                    opt.batch_size_val, True, opt.workers, opt=opt)
    return train_loader, val_loader


def get_test_loader(vocab, opt):
    test_loader = get_precomp_loader( 'test', vocab,
                                      opt.batch_size_val, False, opt.workers, opt=opt)
    return test_loader

