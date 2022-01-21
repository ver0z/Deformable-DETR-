import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T
import requests
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util import box_ops
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import time
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='ICDAR2013')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--imgs_dir', type=str, help='input images folder for inference')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_names = ['N/A', 'table']
colors = ['red', 'blue', 'green', 'yellow', 'black']

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    
    t0 = time.time()
    #img_path = os.path.join(args.imgs_dir, img_file)
    out_imgName = 'test_bem_sucedido.png'
    url = 'data:image/jpeg;base64,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'
    im = Image.open(requests.get(url, stream=True).raw)
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    img=img.cuda()
    # propagate through the model
    outputs = model(img)
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = out_logits.softmax()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    keep = scores[0] > 0.2
    boxes = boxes[0, keep]
    labels = labels[0, keep]

    # and from relative [0, 1] to absolute [0, height] coordinates
    im_h,im_w = im.size
    #print('im_h,im_w',im_h,im_w)
    target_sizes =torch.tensor([[im_w,im_h]])
    target_sizes =target_sizes.cuda()
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    print(time.time()-t0)
    #plot_results
    source_img = Image.open(requests.get(url, stream=True).raw).convert("RGBA")
        
    #fnt = ImageFont.truetype("/content/content/Deformable-DETR/font/Aaargh.ttf", 18)
    draw = ImageDraw.Draw(source_img)
    #print ('label' , labels.tolist())
    label_list =  labels.tolist()
    #print("Boxes",boxes,boxes.tolist())
    i=0
    for xmin, ymin, xmax, ymax in boxes[0].tolist():
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline =colors[label_list[i]-1])
        # print('--------')
        # print('i= ', i)
        # print('label is = ', label_list[i]-1)
        # print(label_names[label_list[i]-1])
        if ymin-18 >=0 :
          ymin = ymin-18
        draw.text((xmin, ymin), label_names[label_list[i]-1], anchor = 'md', fill=colors[label_list[i]-1])
        i+=1
    

    source_img.save(out_imgName, "png")
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    print("Outputs",results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
