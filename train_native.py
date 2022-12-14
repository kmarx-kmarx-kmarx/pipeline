import os
from statistics import mode
from interactive_m2unet import M2UnetInteractiveModel

import numpy as np
import imageio
import albumentations as A
from skimage.filters import threshold_otsu
from skimage.measure import label   
from tqdm import tqdm
# specify the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

def main():
    # setting up
    data_dir = '/media/prakashlab/T7/subsets'#'./Cellpose Exports for 30deg 4ul 15mmps' # data should contain a train and a test folder
    model_root = "./models4"
    epochs = 8000
    steps = 1
    resume = True
    corrid = "200"
    pretrained_model = None # os.path.join(model_root, str(corrid), "model.h5")

    # define the transforms
    transform = A.Compose(
        [
            A.RandomCrop(1500, 1500),
            A.Rotate(limit=(-10, 10), p=1),
            A.HorizontalFlip(p=0.5),
            A.CenterCrop(1024, 1024),  
        ]
    )
    # unet model hyperparamer can be found here: https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=f899f7a8a9144b3f946c4a1362f7e38ae0c00c59&device=unknown_device&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f79696e676b61697368612f6b657261732d756e65742d636f6c6c656374696f6e2f663839396637613861393134346233663934366334613133363266376533386165306330306335392f6578616d706c65732f757365725f67756964655f6d6f64656c732e6970796e62&logged_in=true&nwo=yingkaisha%2Fkeras-unet-collection&path=examples%2Fuser_guide_models.ipynb&platform=mac&repository_id=323426984&repository_type=Repository&version=95#Swin-UNET
    model_config = {
        "type": "m2unet",
        "activation": "sigmoid",
        "output_channels": 1,
        "loss": {"name": "BCELoss", "kwargs": {}},
        "optimizer": {"name": "RMSprop", "kwargs": {"lr": 1e-2, "weight_decay": 1e-8, "momentum": 0.9}},
        "augmentation": A.to_dict(transform),
    }

    #perform_training(data_dir, model_root, epochs, steps, resume, corrid, pretrained_model, transform, model_config)
    perform_testing(data_dir, model_config, model_root, resume, pretrained_model, corrid)


def perform_training(data_dir, model_root, epochs, steps, resume, corrid, pretrained_model, transform, model_config):
    os.makedirs(os.path.join(model_root, str(corrid)), exist_ok=True)
    A.save(transform, model_root + "/transform.json")

    # check if GPU is available
    print(f'GPU: {torch.cuda.is_available()}')

    model = M2UnetInteractiveModel(
        model_config=model_config,
        model_dir=model_root,
        resume=resume,
        pretrained_model=pretrained_model,
        default_save_path=os.path.join(model_root, str(corrid), "model.pth"),
    )
    # load samples
    train_samples = load_samples(data_dir + '/train')
    # train the model 
    iterations = 0
    for epoch in tqdm(range(epochs)):
        losses = []
        # image shape: 512, 512, 3
        # labels shape: 512, 512, 1
        for (image, labels) in train_samples:
            mask = model.transform_labels(labels)
            x = np.expand_dims(image, axis=0)
            x = (x - np.mean(x)) /np.std(x)
            y = np.expand_dims(mask, axis=0)
            losses = []
            for _ in range(steps):
                # x and y will be augmented for each step
                loss = model.train_on_batch(x, y)
                losses.append(loss)
                iterations += 1
                #print(f"iteration: {iterations}, loss: {loss}")
    model.save()

# test
def perform_testing(data_dir, model_config, model_root, resume, pretrained_model, corrid):
    test_samples = load_samples(data_dir + '/test')
    model = M2UnetInteractiveModel(
        model_config=model_config,
        model_dir=model_root,
        resume=resume,
        pretrained_model=pretrained_model,
        default_save_path=os.path.join(model_root, str(corrid), "model.pth"),
    )
    for i, sample in enumerate(test_samples):
        inputs = sample[0].astype("float32")[None, :1024, :1024, :]
        imageio.imwrite(f"m2unet/octopi-inputs_{i}.png", inputs[0].astype('uint8'))
        inputs = (inputs - np.mean(inputs)) /np.std(inputs)
        labels = sample[1].astype("float32")[None, :1024, :1024, :] * 255
        imageio.imwrite(f"m2unet/octopi-labels_{i}.png", labels[0].astype('uint8'))
        results = model.predict(inputs)
        output = np.clip(results[0] * 255, 0, 255)[:, :, 0].astype('uint8')
        imageio.imwrite(f"m2unet/octopi-pred-prob_{i}.png", output)
        threshold = threshold_otsu(output)
        mask = ((output > threshold) * 255).astype('uint8')
        predict_labels = label(mask)
        imageio.imwrite(f"m2unet/octopi-pred-labels_{i}.png", predict_labels)

    print("all done")

# a function for loading cellpose output (image, mask and outline)
def load_samples(train_dir):
    npy_files = [os.path.join(train_dir, s) for s in os.listdir(train_dir) if s.endswith('.npy')]
    samples = []
    for file in npy_files:
        items = np.load(file, allow_pickle=True).item()
        mask = (items['masks'][:, :, None]  > 0) * 1.0
        outline = (items['outlines'][:, :, None]  > 0) * 1.0
        mask = mask * (1.0 - outline)
        sample = (np.stack([items['img'],]*3, axis=2), mask)
        samples.append(sample)
    return samples

if __name__ == "__main__":
    main()