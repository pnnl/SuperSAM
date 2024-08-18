import numpy as np
import torch
from PIL import Image
import torch
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils
import random
from PIL import Image
from pycocotools.coco import COCO
from tqdm import trange


class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, processor, do_crop=False, encoder=None, label='first'):
        self.datadir = dataset_directory
        self.processor = processor
        self.imgs, self.labels = SA1BDataset.get_image_json_pairs(self.datadir)
        self.do_crop = do_crop
        self.encoder = encoder
        self.label = label
        

    def __len__(self):
        return len(self.imgs)
    
    def loader(self,file_path):
        image = Image.open(file_path)
        image = np.array(image)
        image = np.moveaxis(image, -1, 0)
        return image

    @staticmethod
    #Crop region around masked object
    def cropper(image, ground_truth_map,padding=50):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # Pad the crop
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - padding)
        x_max = min(W, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(H, y_max + padding)

        #For color image
        if len(image.shape) > 2:
            cropped_image = image[:, y_min:y_max, x_min:x_max]
        #For grayscale mask
        else:
            cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image
    
    @staticmethod
    #Get bounding boxes from mask.
    def get_bounding_box(ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    
    @staticmethod
    def get_random_prompt(ground_truth_map,bbox):
        x_min, y_min, x_max, y_max = bbox
        while True:
            # Generate random point within the bounding box
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            # Check if the point lies inside the mask
            if ground_truth_map[y, x] == 1:
                return [x, y]
    
    @staticmethod
    def get_image_json_pairs(directory):
        jpg_files = []
        json_files = []

        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                jpg_files.append(filename)
                json_file = filename[:-4] + '.json'
                json_files.append(json_file)

        return jpg_files, json_files
    
    @staticmethod
    def get_embedding(filepath):
        return torch.load(filepath)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path =  os.path.join(self.datadir, self.imgs[index]) # discard automatic subfolder labels 
        label_path = os.path.join(self.datadir, self.labels[index])
        image = self.loader(img_path)
        masks = json.load(open(label_path))['annotations'] # load json masks

        #Just pick the first mask
        if self.label == 'first':
            mask = masks[0]
            bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
        elif self.label == 'rand':
            idx = random.randint(0, len(masks) - 1)
            mask = masks[idx]
            bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
        elif self.label == 'to_embedding':
            inputs = self.processor(image, return_tensors="pt")
            embed_file = self.imgs[index][:-4] + '.pt'
            embed_path = os.path.join(self.datadir,embed_file)
            return inputs, embed_path
        elif self.label == 'from_embedding':
            inputs = self.processor(image, return_tensors="pt")
            embed_file = self.imgs[index][:-4] + '.pt'
            embed_path = os.path.join(self.datadir,embed_file)
            embedding = SA1BDataset.get_embedding(embed_path)

            return inputs, embedding

        elif self.label == 'all_test':
            points, boxes = [], []
            for mask in masks:
                bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
                bbox_prompt = SA1BDataset.get_bounding_box(bin_ground_truth_mask)
                point_prompt = SA1BDataset.get_random_prompt(bin_ground_truth_mask,bbox_prompt)
                points.append(point_prompt)
                boxes.append(bbox_prompt)
        
            inputs = self.processor(image, input_points=points, input_boxes=[[boxes]], return_tensors="pt")
            return inputs, image, masks, boxes,  points
                
        elif self.label == 'all_train':
            points, boxes = [], []
            for mask in masks:
                bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
                bbox_prompt = SA1BDataset.get_bounding_box(bin_ground_truth_mask)
                point_prompt = SA1BDataset.get_random_prompt(bin_ground_truth_mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
                
            inputs = self.processor(image, input_points=[points], input_boxes=[[boxes]], return_tensors="pt")
            return inputs, image, masks, boxes,  points

    
class MitoDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
  
    @staticmethod
    #Get bounding boxes from mask.
    def get_bounding_box(ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    
    @staticmethod
    def get_random_prompt(ground_truth_map,bbox):
        x_min, y_min, x_max, y_max = bbox
        while True:
            # Generate random point within the bounding box
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            # Check if the point lies inside the mask
            if ground_truth_map[y, x] == 1:
                return x, y

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        ground_truth_mask = torch.tensor(ground_truth_mask)

        # get bounding box prompt
        bbox_prompt = MitoDataset.get_bounding_box(ground_truth_mask)
        point_prompt = MitoDataset.get_random_prompt(ground_truth_mask,bbox_prompt)


        # Convert the image to grayscale (if it's not already in grayscale)
        image = image.convert("L")

        # Add a channel dimension
        image = image.convert("RGB")

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[bbox_prompt]],input_points=[[point_prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        return inputs, ground_truth_mask


class COCOSegmentation(torch.utils.data.Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 base_dir='datasets/coco',
                 split='train',
                 year='2017',
                 labels='one',
                 processor=None):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, '{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask_utils
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args
        self.processor = processor
        self.labels = labels
        self.mask_idx = 0 if split == 'val' else 1
    
    @staticmethod
    def get_bounding_box(ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox

    @staticmethod
    def get_random_prompt(ground_truth_map,bbox):
        x_min, y_min, x_max, y_max = bbox
        while True:
            # Generate random point within the bounding box
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            # Check if the point lies inside the mask
            if ground_truth_map[y, x] == 1:
                return x, y

    def __getitem__(self, index):
        _img, _target = self._make_img_gts_pair(index)
        
        image, masks = np.array(_img), [np.array(t) for t in _target]
        image = np.moveaxis(image, -1, 0)


        masks = [torch.tensor(mask) for mask in masks if np.any(mask)]
        #binarize mask
        masks = [(mask > 0).float() for mask in masks]

        if self.split == 'val':
            points, boxes = [], []
            for mask in masks:
                bbox_prompt = COCOSegmentation.get_bounding_box(mask)
                point_prompt = COCOSegmentation.get_random_prompt(mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
        
            inputs = self.processor(image, input_points=points, input_boxes=[[boxes]], return_tensors="pt")
            return inputs, image, masks, boxes,  points
                
        elif self.split == 'train':
            points, boxes = [], []
            for mask in masks:
                bbox_prompt = COCOSegmentation.get_bounding_box(mask)
                point_prompt = COCOSegmentation.get_random_prompt(mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
                
            inputs = self.processor(image, input_points=[points], input_boxes=[[boxes]], return_tensors="pt")
            return inputs, image, masks, boxes,  points


    def _make_img_gt_point_pair(self, index, mask_idx=0):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        annids = coco.getAnnIds(imgIds=img_id)
        if mask_idx == 0:
            annid = annids[0] 
        else:
            idx = random.randint(0, len(annids) - 1)
            annid = annids[idx] 
        cocotarget = coco.loadAnns(annid)
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _make_img_gts_pair(self,index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        annids = coco.getAnnIds(imgIds=img_id)
        _targets = []
        for annid in annids:
            cocotarget = coco.loadAnns(annid)
            _targets.append(Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width'])))
        
        return _img, _targets

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.ids)