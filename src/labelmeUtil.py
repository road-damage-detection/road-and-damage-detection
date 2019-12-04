import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import cv2
import os
from glob import glob
import json
import os.path as osp
import sys
import math
import PIL.Image
import PIL.ImageDraw
import io


def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]

    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        # assert len(xy) > 2, 'Polygon must have points more than 2'
        if (len(xy) > 2):
            draw.polygon(xy=xy, fill=1, outline=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.int32)
        instance_names = ['_background_']
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)

            ins_id = instance_names.index(label)
        cls_id = label_name_to_value[cls_name]
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        if type == 'instance':
            ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls


def generateMaskImage(input_dir, output_dir):
    listOfFiles = os.listdir(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_names = []
    class_name_to_id = {}
    labels = ["__ignore__", "_background_", "rough road", "pothole", "waterlog", "wet road", "muddy road",
              "obstruction", "bump", "side road"]

    for i, line in enumerate(labels):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)

    class_names = tuple(class_names)

    for l in listOfFiles:
        subFolder = input_dir + "/" + l
        out_subFolder = output_dir + "/" + l
        if (os.path.isdir(subFolder)):
            for label_file in glob(osp.join(subFolder, '*.json')):
                with open(label_file) as f:

                    if not os.path.exists(out_subFolder):
                        os.makedirs(out_subFolder)

                    print("Processing File Name : ", label_file)
                    base = osp.splitext(osp.basename(label_file))[0]
                    out_png_file = osp.join(out_subFolder, base + '.png')
                    data = json.load(f)
                    img_file = osp.join(osp.dirname(label_file), data['imagePath'])
                    img = np.asarray(PIL.Image.open(img_file))

                    lbl = shapes_to_label(
                        img_shape=img.shape,
                        shapes=data['shapes'],
                        label_name_to_value=class_name_to_id,
                    )

                    lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
                    lbl_pil.save(out_png_file)
