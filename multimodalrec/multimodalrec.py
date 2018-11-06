# Packages
import os, pickle, sys
import numpy as np
import tensorflow as tf

# In-built packages
from .trailer import AudioVisualEncoder
from .viewership import BipartiteNetwork, CollaborativeFiltering
# from .basicGraph import BasicGraph
sys.path.append("/Users/salihgundogdu/Desktop/gits/multimodalrec/data/")
from data import data_creation as dc



def SVM_model(user_factors, movie_factors, trailer_sequence_representation, output):
    pass



class MultimodalRec(object):
    """DocString"""
    def __init__(self,
                audio_encoding='MFCC', # Type of Encodings for Audio Data (STRING)
                visual_encoding='PreTrained_CNN', # Type of Encodings for Visual Data (STRING)
                viewership_encoding='Node2Vec', # Type of Encodings for Viewership Data (STRING) 'Node2Vec' or 'LatentFactor'
                n_visual=128, # Dimension of the Visual Representation Vector (INT)
                n_audial=128, # Dimension of the Visual Representation Vector (INT)
                video_processor=None,#AudioVisualEncoder(), # Encoder to Extract AudioVisual Representations of Given Movie Trailers (OBJ)
                system='SVM' # 'Basic' To compare different architectures

                #user_item_network=BipartiteNetwork(), # U-I Adjacency Graph to compute User and Item Representation (OBJ)

                ):

        # ADD check version

        # ADD check args*

        self.audio_encoding = audio_encoding
        self.visual_encoding = visual_encoding
        self.viewership_encoding = viewership_encoding
        self.n_visual = n_visual
        self.n_audial = n_audial
        self.video_processor=video_processor
        self.system = system
        if viewership_encoding == 'Node2Vec':
            self.user_item_network = None#BipartiteNetwork()
        elif viewership_encoding == 'LatentFactor':
            self.user_item_network = None#CollaborativeFiltering()


           # self.data_generator = DataGenerator()
        self.user_latent=None
        self.movie_latent=None
        self.visual_features=None

        # ADD Tensorflow Graph Naming

    def organize_multimodal_data(self, load=False):
        directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-1m/ratings.dat'
        all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
        pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/pickles/'

        data = dc.get_movielens_1M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)

        # Get Latent Factors
        print('User-Movie Latent Factors are extracting...')
        self.user_item_network = CollaborativeFiltering(data['training']) # 0 Threshold trim
        user_latent, movie_latent, sigma = self.user_item_network.compute_latent_factors(algorithm='SVD', k=256)
        self.user_latent = user_latent
        self.movie_latent = movie_latent
        print('Done.')

        # Get Representations of Trailer Frames
        print('Visual Representations are extracting...')
        self.video_processor = AudioVisualEncoder()
        sequences = self.video_processor.extract_visual_features(_dir_='data/',load=load)
        self.visual_features = sequences
        print('Done.')




        #yield (X, y)

    def train(self, system):
        if system == 'Basic':
            model = BasicGraph()
        elif system == 'SVM':
            SVM_model(self.user_factors, self.movie_factors, self.trailer_sequence_representation, self.output)


