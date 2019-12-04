import numpy as np
import os
from glob import glob
import json
import os.path as osp
import sys
from PIL import Image

class roadDamageDataset():

    def __init__(self, imagesFolderPath, labelsFolderPath):
        self.images_root = imagesFolderPath
        self.labels_root = labelsFolderPath
        
    def load_data(self):
        img=[]
        lbl=[]
        labelsFolder=self.labels_root
        imagesFolder = self.images_root
        listOfFiles = os.listdir(labelsFolder)
        
        for l in listOfFiles:
            outputFolder = labelsFolder + "/" + l
            inputFolder = imagesFolder + "/" + l
            if (os.path.isdir(outputFolder)):
                for label_file in glob(osp.join(outputFolder, '*.png')):
                    with open(label_file) as f:
                        base = osp.splitext(osp.basename(label_file))[0]
                        img_file = osp.join(inputFolder, base + '.png')
                        if(os.path.isfile(img_file)):
                            lbl.append(np.asarray(Image.open(f.name)))
                            img.append(np.asarray(Image.open(img_file)))
                            
        return np.asarray(img),np.asarray(lbl)
  