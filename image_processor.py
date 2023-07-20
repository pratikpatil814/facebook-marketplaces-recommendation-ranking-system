from torchvision import transforms
from PIL import Image
def image_processor(image_path):
    image = Image.open(image_path)
    transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]) 
            ])
    processed_image = transforms(image)
    processed_image = processed_image.unsqueeze(0)
    return processed_image

    