from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((256)),
        # transforms.RandomRotation(15),
        # transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])