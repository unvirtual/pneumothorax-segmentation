import albumentations as album

def get_augmentations():
    transforms = [
        album.HorizontalFlip(p=0.5),
    
        album.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),
    
#        album.IAAAdditiveGaussianNoise(p=0.2),
#        album.IAAPerspective(p=0.2),
    
        album.OneOf(
            [
#                album.CLAHE(p=1),
                album.RandomContrast(),
                album.RandomBrightness(),
                album.RandomGamma()
            ],
            p=0.3
        ),
        album.OneOf(
            [
                album.ElasticTransform(alpha=120, sigma=120*0.05,alpha_affine=120*0.05),
                album.GridDistortion(),
                album.OpticalDistortion(distort_limit=2, shift_limit=0.5)
            ],
            p=0.3,
        )
    ]
    return album.Compose(transforms)
