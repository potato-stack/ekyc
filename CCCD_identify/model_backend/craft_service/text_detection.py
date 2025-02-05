import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from collections import OrderedDict

from ..craft_service.craft import CRAFT 
from ..craft_service.refinenet import RefineNet
from ..craft_service import craft_utils, file_utils, imgproc



class TextDetection(object):
    def __init__(
            self, 
            path_to_trained_model, 
            path_to_refines_model, 
            path_to_save_file = './result/',
            low_text = 0.7, 
            link_threshold = 0.2,
            text_threshold = 0.7,
            canvas_size = 1280,
            mag_ratio = 2.5,
            cuda = False,
            poly = False,
            refine = True
            ):
        # Model path 
        self.path_to_trained_model = path_to_trained_model
        self.path_to_refines_model = path_to_refines_model
        self.path_to_save_file = path_to_save_file
        # Configuration threshold 
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.text_threshold = text_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        # Configuration flag
        self.cuda = cuda
        self.poly = poly
        self.refine = refine
        # Predfined model holder
        self.trained_model = None
        self.refined_model = None
        self.load_model()
    
    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    
    def load_model(self):
        """
        The text area detection model will include two seperate wieght file in the pytorch format.
        One will use to detect and one is used to refine, make sure to download both
        """

        # Load the NET model
        self.trained_model = CRAFT()
        print('Loading weights from checkpoint (' + self.path_to_trained_model + ')')
        if self.cuda:
            self.trained_model.load_state_dict(self.copyStateDict(torch.load(self.path_to_trained_model)))
        else:
            self.trained_model.load_state_dict(self.copyStateDict(torch.load(self.path_to_trained_model, map_location='cpu', weights_only=False)))

        if self.cuda:
            self.trained_model = self.trained_model.cuda()
            self.trained_model = torch.nn.DataParallel(self.trained_model)
            cudnn.benchmark = False

        self.trained_model.eval()

        #Load LinkRefiner model
        if self.refine:
            self.refined_model = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.path_to_refines_model + ')')
            if self.cuda:
                self.refined_model.load_state_dict(self.copyStateDict(torch.load(self.path_to_refines_model)))
                self.refined_model = self.refined_model.cuda()
                self.refined_model = torch.nn.DataParallel(self.refined_model)
            else:
                self.refined_model.load_state_dict(self.copyStateDict(torch.load(self.path_to_refines_model, map_location='cpu', weights_only=False)))

            self.refined_model.eval()
            self.poly = True
        
    def predict(self, img, save = False, verbose = False):
        if verbose: t0  = time.time()
        """
            Resize image to the appropriate size land ajust minor format
        """
        # Resize image 
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        """
            Preprocessing input for model
        """
        # Preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        """
            Inference
        """
        # forward pass
        with torch.no_grad():
            y, feature = self.trained_model(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if self.refined_model is not None:
            with torch.no_grad():
                y_refiner = self.refined_model(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        """
            Post processing output model
        """
        # Take out the detection box of text area
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)
        # Scale the box back to img original size

        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        
        if verbose: 
            t0 = time.time() - t0
            t1 = time.time()
            print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
        ret_score_text = None
        if save:
            img_name = "res_img"
            render_img = score_text.copy()
            render_img = np.hstack((render_img, score_link))
            ret_score_text = imgproc.cvt2HeatmapImg(render_img)
            if os.path.isdir(self.path_to_save_file):
                # If the directory exists, count the number of files
                files = os.listdir(self.path_to_save_file)
                # Optionally filter out directories if only files are needed
                img_name += "_"
                img_name += str(len([f for f in files if os.path.isfile(os.path.join(self.path_to_save_file, f))]))
            img_name += ".jpg"
            file_utils.saveResult(img_name, img[:,:,::-1], polys, dirname=self.path_to_save_file)
        
        return boxes, polys, ret_score_text