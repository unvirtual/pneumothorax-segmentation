import albumentations as album

def get_augmentations():
    transforms = [
        album.HorizontalFlip(p=0.5),
    
        album.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0.2, shift_limit=0.1, p=1, border_mode=0),
    
        album.IAAAdditiveGaussianNoise(p=0.2),
        album.IAAPerspective(p=0.2),
    
        album.OneOf(
            [
                album.CLAHE(p=1),
                album.RandomBrightness(p=1),
                album.RandomGamma(p=1)
            ],
            p=0.5
        ),
        album.OneOf(
            [
                album.RandomContrast(p=1)
    #            albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        )
    ]
    return album.Compose(transforms)
