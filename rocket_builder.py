import torch
import torch.nn as nn
from PIL import Image
import types
import numpy as np
from PIL import ImageDraw
from torchvision import transforms

from .data import VOC_CLASSES as labels

from .ssd import build_ssd

def build() -> nn.Module:
    """Builds a pytorch compatible deep learning model

    The model can be used as any other pytorch model. Additional methods
    for `preprocessing`, `postprocessing`, `label_to_class` have been added to ease handling of the model
    and simplify interchangeability of different models.
    """
    # Build SSD3000
    model = build_ssd(size=300, num_classes=21)    # initialize SSD
    model.load_weights('./rockets/SSD/ssd300_mAP_77.43_v2.pth')
   
    # classes = load_classes(os.path.join(os.path.realpath(os.path.dirname(__file__)), "coco.data"))

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    model.train_forward = types.MethodType(train_forward, model)

    # model.label_to_class = types.MethodType(label_to_class, model)
    # model.get_loss = types.MethodType(get_loss, model)
    # setattr(model, 'classes', classes)

    return model


def preprocess(self, img: Image, labels: list = None) -> torch.Tensor:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """

    # todo: support batch size bigger than 1 for training and inference
    # todo: replace this hacky solution and work directly with tensors
    if type(img) == Image.Image:
        # PIL.Image
        pass
    elif type(img) == torch.Tensor:
        # list of tensors
        img = img[0].cpu()
        img = transforms.ToPILImage()(img)
    elif "PIL" in str(type(img)):  # type if file just has been opened
        img = img.convert("RGB")
    else:
        raise TypeError("wrong input type: got {} but expected list of PIL.Image, "
                        "single PIL.Image or torch.Tensor".format(type(img)))

    in_width, in_height = img.size

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
    out_tensor = input_img.unsqueeze(0)

    if labels is None:
        return out_tensor

    max_objects = 50
    filled_labels = np.zeros((max_objects, 5))  # max objects in an image for training=50, 5=(x1,y1,x2,y2,category_id)
    if labels is not None:
        for idx, label in enumerate(labels):

            padded_w = in_width
            padded_h = in_height

            # resize coordinates to match Yolov3 input size
            scale_x = 300.0 / padded_w
            scale_y = 300.0 / padded_h

            label[0] *= scale_x
            label[1] *= scale_y
            label[2] *= scale_x
            label[3] *= scale_y

            x1 = label[0] / 300.0
            y1 = label[1] / 300.0
            x2 = label[2] / 300.0
            y2 = label[3] / 300.0

            filled_labels[idx] = np.asarray([x1, y1, x2, y2, label[4]])
            if idx >= max_objects - 1:
                break

    # remove zero rows
    # filled_labels = filled_labels[~np.all(filled_labels == 0, axis=1)]
    filled_labels = torch.from_numpy(filled_labels)


    return out_tensor, filled_labels.unsqueeze(0)


def train_forward(self, x: torch.Tensor, targets: torch.Tensor):
    """Performs forward pass and returns loss of the model

    The loss can be directly fed into an optimizer.
    """
    loss = self.forward(x, targets.float())
    return loss


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