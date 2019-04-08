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

def clamp(n, minn, maxn):
    """Make sure n is between minn and maxn

    Args:
        n (number): Number to clamp
        minn (number): minimum number allowed
        maxn (number): maximum number allowed
    """
    return max(min(maxn, n), minn)

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
    img_height, img_width, _ =  img.shape
    detections = detections.data
    score_threshold = 0.3

    list_detections = []
    # scale each detection back up to the image
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    # Go throught each class
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= score_threshold:
            # detections = [image_id, class_id, anchor, [score, anchor_x0, anchor_y0, anchor_x1, anchor_y1]]
            score = clamp(detections[0,i,j,0], 0, 1).cpu().item()
            label_name = str(labels[i-1])

            pt = (detections[0,i,j,1:]*scale).cpu().numpy()

            topLeft_x = int(clamp(round(pt[0]), 0, img_width))
            topLeft_y = int(clamp(round(pt[1]), 0, img_height))

            bottomRight_x = int(clamp(round(pt[2]), 0, img_width))
            bottomRight_y = int(clamp(round(pt[3]), 0, img_width))
            
            width = abs(bottomRight_x - topLeft_x) + 1
            height = abs(bottomRight_y - topLeft_y) + 1
            
            list_detections.append({
                'topLeft_x': topLeft_x,
                'topLeft_y': topLeft_y,
                'width': width,
                'height': height,
                'bbox_confidence': score, # No bbox_confidence then it is equal to class_confidence
                'class_name': label_name,
                'class_confidence': score})
            
            j+=1

    if visualize:
        line_width = 2
        img_out = input_img
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        for detection in list_detections:
            # Extract information from the detection
            topLeft = (detection['topLeft_x'], detection['topLeft_y'])
            bottomRight = (detection['topLeft_x'] + detection['width'] - line_width, detection['topLeft_y'] + detection['height']- line_width)
            class_name = detection['class_name']
            bbox_confidence = detection['bbox_confidence']
            class_confidence = detection['class_confidence']

            # Draw the bounding boxes and the information related to it
            ctx.rectangle([topLeft, bottomRight], outline=(255, 0, 0, 255), width=line_width)
            ctx.text((topLeft[0] + 5, topLeft[1] + 10), text="{}, {:.2f}, {:.2f}".format(class_name, bbox_confidence, class_confidence))

        del ctx
        return img_out

    return list_detections