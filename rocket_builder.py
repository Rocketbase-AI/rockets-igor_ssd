import torch
import torch.nn as nn
from PIL import Image
import types
import numpy as np
from PIL import ImageDraw

from .data import VOC_CLASSES as labels

from .ssd import build_ssd

def build() -> nn.Module:
    """Builds a pytorch compatible deep learning model

    The model can be used as any other pytorch model. Additional methods
    for `preprocessing`, `postprocessing`, `label_to_class` have been added to ease handling of the model
    and simplify interchangeability of different models.
    """
    # Build SSD3000
    model = build_ssd('test', 300, 21)    # initialize SSD
    model.load_weights('./rockets/SSD/ssd300_mAP_77.43_v2.pth')
   
    # classes = load_classes(os.path.join(os.path.realpath(os.path.dirname(__file__)), "coco.data"))

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    # model.label_to_class = types.MethodType(label_to_class, model)
    # model.get_loss = types.MethodType(get_loss, model)
    # setattr(model, 'classes', classes)

    return model

def preprocess(self, img: Image) -> torch.Tensor:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.

    Args:
        x (list or PIL.Image): input image or list of images.
    """
    # Resize the image
    img = img.resize((300, 300), Image.ANTIALIAS)
    # Convert it to numpy array
    input_img = np.array(img).astype(np.float32)
    # Normalize the color values
    input_img -= (104.0, 117.0, 123.0)
    # Cast to Float32
    input_img = input_img.astype(np.float32)

    input_img = input_img[:, :, ::-1].copy()
    # Convert it to a tensor and permute the channels
    input_img = torch.from_numpy(input_img).permute(2, 0, 1)

    return input_img.unsqueeze(0)

def postprocess(self, detections: torch.Tensor, input_img: Image, visualize: bool=False):
    """Converts pytorch tensor into interpretable format

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.
    This model supports either outputting a list of bounding boxes of the format
    (x0, y0, w, h) or outputting a `PIL.Image` with the bounding boxes
    and (class name, class confidence, object confidence) indicated.

    Args:
        detections (Tensor): Output Tensor to postprocess
        input_img (PIL.Image): Original input image which has not been preprocessed yet
        visualize (bool): If True outputs image with annotations else a list of bounding boxes
    """
    img = np.array(input_img)
    detections = detections.data

    list_detections = []
    # scale each detection back up to the image
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.3:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            list_detections.append([pt[0], pt[1], pt[2]-pt[0]+1, pt[3]-pt[1]+1, label_name, score])
            j+=1

    if visualize:
        img_out = input_img
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        for bbox in list_detections:
            x1, y1, w, h, label_name, score = bbox
            ctx.rectangle([(x1, y1), (x1 + w, y1 + h)], outline=(255, 0, 0, 255), width=2)
            ctx.text((x1+5, y1+10), text="{}, {:.2f}".format(label_name, score))
        del ctx
        return img_out

    return list_detections



