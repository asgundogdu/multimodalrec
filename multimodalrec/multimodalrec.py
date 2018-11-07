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


def create_train_val_test(ratings_df_training, user_latent_traninig, movie_factors_training,
           					ratings_df_test, user_latent_test, movie_factors_test, visual_features, output):
    # TRAIN
    # Create X_train y_train (POSITIVE)
	pos_ratings_df = ratings_df_training[ratings_df_training.Rating>4]
	pos_ratings_df = pos_ratings_df.assign(Likes= lambda x: 1)

	# Create X_train y_train (NEGATIVE)
	neg_ratings_df = ratings_df_training[ratings_df_training.Rating<3]
	neg_ratings_df = neg_ratings_df.assign(Likes= lambda x: 0)

	training_df = pd.concat([pos_ratings_df,neg_ratings_df],axis=0, sort=False)
	training_df = training_df.drop(['Timestamp','Rating'], axis=1)

	train = training_df.sample(frac=0.8,random_state=200)
	val = training_df.drop(train.index)

	# TEST
	# Create X_train y_train (POSITIVE)
	pos_ratings_df = ratings_df_test[ratings_df_test.Rating>4]
	pos_ratings_df = pos_ratings_df.assign(Likes= lambda x: 1)

	# Create X_train y_train (NEGATIVE)
	neg_ratings_df = ratings_df_test[ratings_df_test.Rating<3]
	neg_ratings_df = neg_ratings_df.assign(Likes= lambda x: 0)

	test = pd.concat([pos_ratings_df,neg_ratings_df],axis=0, sort=False)
	test = test.drop(['Timestamp','Rating'], axis=1)

	X_train,y_train = normalize_concat_inputs(user_latent_traninig, visual_features, train)
	X_val,y_val = normalize_concat_inputs(user_latent_traninig, visual_features, val)
	X_test,y_test = normalize_concat_inputs(user_latent_test, visual_features, test)

	return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_concat_inputs(user_latent, visual_features, data)
	X,y = [],[]
	for index, row in data.iterrows(): 
		fusion_input = np.array(user_latent[row['User']]) / np.linalg.norm(np.array(user_latent[row['User']]))
		
		vis_frames = visual_features[row['Movie']]
		lstm_input = []
		for frame in vis_frames:
			normed_frame_features = frame / np.linalg.norm(frame)
			lstm_input.append(normed_frame_features)
		lstm_input = np.array(lstm_input)
		X.append((lstm_input, fusion_input))
		y.append(row['Likes'])
		return X,y


def Model1(ratings_df_training, user_latent_traninig, movie_factors_training,
           ratings_df_test, user_latent_test, movie_factors_test, visual_features, output): # Inputs are dictionary
	X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test(ratings_df_training, user_latent_traninig, movie_factors_training,
           																	ratings_df_test, user_latent_test, movie_factors_test, visual_features, output)

	

	def train()




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
            self.user_item_network_training = None#BipartiteNetwork()
            self.user_item_network_test = None
        elif viewership_encoding == 'LatentFactor':
            self.user_item_network_training = None#CollaborativeFiltering()
            self.user_item_network_test = None

           # self.data_generator = DataGenerator()
        self.user_latent_traninig=None
        self.movie_latent_traninig=None

        self.user_latent_test=None
        self.movie_latent_test=None

        self.visual_features=None

        # ADD Tensorflow Graph Naming

    def organize_multimodal_data(self, load=False):
        directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-1m/ratings.dat'
        all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
        pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/pickles/'

        data = dc.get_movielens_1M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)

        # Get Latent Factors Training
        print('Training User-Movie Latent Factors are extracting...')
        self.user_item_network_training = CollaborativeFiltering(data['training']) # 0 Threshold trim
        user_latent_traninig, movie_latent_traninig, sigma = self.user_item_network.compute_latent_factors(algorithm='SVD', k=256)
        self.user_latent_traninig = user_latent_traninig
        self.movie_latent_traninig = movie_latent_traninig
        print('Done.')

        # Get Latent Factors Test
        print('Test User-Movie Latent Factors are extracting...')
        self.user_item_network = CollaborativeFiltering(data['test']) # 0 Threshold trim
        user_latent_test, movie_latent_test, sigma = self.user_item_network.compute_latent_factors(algorithm='SVD', k=256)
        self.user_latent_test = user_latent_test
        self.movie_latent_test = movie_latent_test
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
            model = Model1(self.user_item_network_training.CF_data, self.user_latent_traninig, self.movie_factors_training,
            			   self.user_item_network_test.CF_data, self.user_latent_test, self.movie_factors_test, self.visual_features, self.output)
        elif system == 'SVM':
            SVM_model(self.user_factors, self.movie_factors, self.trailer_sequence_representation, self.output)


