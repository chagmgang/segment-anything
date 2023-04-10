import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import argparse
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    # ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def main(args):

    image = Image.open(args.image_path)
    image = np.array(image)
    sam_checkpoint = args.checkpoint
    model_type = 'vit_h'
    device = 'cuda'

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)
    input_boxs = np.array([
        [76, 103, 105, 138],
        [246, 259, 285, 281],
        [495, 392, 553, 426],
    ])

    plt.imshow(image)
    for input_box in input_boxs:

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
    plt.savefig('test.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()
    main(args)
