import numpy as np
import os
import pickle
from scipy import sparse as sp
import tensorflow as tf


class MultimodalRec(object):
	"""DocString"""
    def __init__(self,
    			audio_encoding='MFCC', # Type of Encodings for Audio Data (STRING)
    			visual_encoding='PreTrained_CNN', # Type of Encodings for Visual Data (STRING)
    			viewership_encoding='Node2Vec', # Type of Encodings for Viewership Data (STRING)
                n_visual=100, # Dimension of the Visual Representation Vector (INT)
                n_audial=100, # Dimension of the Visual Representation Vector (INT)
                video_processor=AudioVisualEncoder(), # Encoder to Extract AudioVisual Representations of Given Movie Trailers (OBJ)
                user_item_network=BipartiteNetwork(), # U-I Adjacency Graph to compute User and Item Representation (OBJ)
                ):

    	# ADD check version

    	# ADD check args*

    	self.audio_encoding = audio_encoding
    	self.visual_encoding = visual_encoding
    	self.viewership_encoding = viewership_encoding
    	self.n_visual = n_visual
        self.n_audial = n_audial
        self.video_processor=video_processor
        self.user_item_network = user_item_network
      	
        # ADD Tensorflow Graph Naming

