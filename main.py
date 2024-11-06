import argparse
import random
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

class CocoDataset(Dataset):
    def __init__(self, json_file: str, image_dir: str, transforms: Optional[object] = None):
        """
        Initialize the COCO format dataset.
        
        Args:
            json_file (str): Path to the COCO format annotation file
            image_dir (str): Directory containing the images
            transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.transforms = transforms
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Annotation file not found: {json_file}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        # Load COCO annotations
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.image_dir = image_dir
        
        # Create image ID mapping
        self.image_dict = {img['id']: img for img in self.coco_data['images']}
        
        # Group annotations by image_id
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)
            
        self.image_ids = list(self.annotations.keys())
        print(f"Dataset loaded successfully: {len(self.image_ids)} images")
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        image_info = self.image_dict[image_id]
        
        img_path = os.path.join(self.image_dir, image_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = Image.open(img_path).convert("RGB")
        img = torchvision.transforms.ToTensor()(img)
        
        anns = self.annotations[image_id]
        boxes = []
        labels = []
        
        image_width = image_info['width']
        image_height = image_info['height']
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            x = max(0, min(x, image_width))
            y = max(0, min(y, image_height))
            w = min(w, image_width - x)
            h = min(h, image_height - y)
            
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
        
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor([ann["area"] for ann in anns]),
            "iscrowd": torch.tensor([ann["iscrowd"] for ann in anns])
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self) -> int:
        return len(self.image_ids)

class Compose:
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            if len(bbox):
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target

def get_transform(train: bool) -> Compose:
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   data_loader: DataLoader, 
                   device: torch.device) -> float:
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def main(args):
    json_file = args.json_file
    image_dir = args.image_dir
    num_classes = args.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    
    try:
        dataset_train = CocoDataset(
            json_file=json_file,
            image_dir=image_dir,
            transforms=get_transform(train=True)
        )
        
        dataset_test = CocoDataset(
            json_file=json_file,
            image_dir=image_dir,
            transforms=get_transform(train=False)
        )
        
        data_loader_train = DataLoader(
            dataset_train, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, 
            collate_fn=collate_fn  
        )
        
        data_loader_test = DataLoader(
            dataset_test, 
            batch_size=1, 
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn  
        )
        
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        model.to(device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        best_loss = float('inf')
        for epoch in range(num_epochs):
            try:
                avg_loss = train_one_epoch(model, optimizer, data_loader_train, device)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                
                lr_scheduler.step(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = os.path.join(args.output_dir, 'frcnn_captcha_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, best_model_path)
                    print(f"New best model saved: {best_model_path}")
                
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = os.path.join(args.output_dir, f'frcnn_captcha_epoch_{epoch+1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
                    
            except KeyboardInterrupt:
                print("Training interrupted. Saving current model...")
                interrupted_path = os.path.join(args.output_dir, 'frcnn_captcha_interrupted.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, interrupted_path)
                print(f"Interrupted model saved: {interrupted_path}")
                break
        
        final_path = os.path.join(args.output_dir, 'frcnn_captcha_final.pth')
        torch.save(model.state_dict(), final_path)
        print(f"Final model saved: {final_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on custom COCO dataset.")
    parser.add_argument('--json_file', type=str, required=True, help="Path to the COCO format annotation file")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing the images")
    parser.add_argument('--output_dir', type=str, default='.', help="Directory to save model checkpoints")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes including background")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training")
    
    args = parser.parse_args()
    main(args)
