import os, sys, pickle
import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx
from networkx.algorithms import bipartite
from scipy.sparse.csgraph import minimum_spanning_tree as mst_nsim
from scipy.sparse.linalg import svds
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

from node2vec import Node2Vec

# from .data import data_creation as create

def _svd_compute(Ratings_demeaned, k):
	U, sigma, Vt = svds(Ratings_demeaned, k)
	return (U, sigma, Vt)


def _compute_Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=6):
		node_vects = Node2Vec(G, dimensions=64, walk_length=32, num_walks=200, workers=8)
		model = node_vects.fit()

		vectors = model.wv.vectors.tolist()
		node_indices = model.wv.index2word

		#node_vectors_dict = {'vector':vectors,'node_index':node_indices}
		representations = {}
		for index, vector in zip(model.wv.index2word,model.wv.vectors.tolist()):
		    representations[index] = vector

		return representations


def _trim_graph(network, trim_type, value=64):
	# Getting Adjacency Matrix of the Network
	A_ij = nx.adjacency_matrix(network)
	A_ij = A_ij.todense()
	# print(np.count_nonzero(A_movies >= 1))
	if trim_type==1:
		A_ij_trimed = _knn_mst(A_ij, k=value)
	# print(np.count_nonzero(A_trimed >= 1))
	elif trim_type==0:
		A_ij_trimed = A_ij
		A_ij_trimed[A_ij_trimed < value] = 0
		print("Number of remaining edge in trimmed graph: {}".format(np.count_nonzero(A_ij_trimed >= 1)))

	G_movies_trimmed = nx.from_numpy_matrix(A_ij_trimed)

	return G_movies_trimmed


def _knn_mst(D, k=13): # BUG WORKS WITH DISTANCE MATRIX (+++++++++++++++++)
    n = D.shape[0]
    assert (D.shape[0] == D.shape[1])

    np.fill_diagonal(D, 0)
    A = np.zeros((n, n))
    for i in range(n):
        ix = np.argsort(D[i, :])
        A[i, ix[0,1]] = 1  # Connect to the nearest node after itself
        A[ix[0,1], i] = 1  # The same on other direction

        for j in range(k - 1):
            ij = j + 1
            #             if D[i, ix[ij]] < theta:
            A[i, ix[0,ij]] = 1
            A[ix[0,ij], i] = 1

    mst_remained_edges = _mst_sym(D, False)
    remained_edges = np.maximum(A, mst_remained_edges)

    return remained_edges


def _mst_sym(A, return_LongestLinks=True):
    """scipy mst (kruskal) return triangular matrix as mst"""
    dim = A.shape[0]
    mst = mst_nsim(A).todense()
    mst[mst > 0] = 1
    remained_edges = np.maximum(mst, mst.T)

    if False:#return_LongestLinks: # make it false
        LongestLinks = findMlink(np.multiply(A, remained_edges))
        return remained_edges, LongestLinks
    else:
        return remained_edges


class BipartiteNetwork(object):
	"""
	Bipartite Network Node2Vec Encoding
	
	Bipartite_data: (pandas dataframe)
	trim_type: (int) 0 for threshold 1 for knn_mst
	trim_value_user: (int) either threshold or number k used in knn_mst
	trim_value_movie: (int) either threshold or number k used in knn_mst

	"""
	def __init__(self, Bipartite_data, trim_type, trim_value_user, trim_value_movie): 
		super(BipartiteNetwork, self).__init__()
		self.Bipartite_data = Bipartite_data
		self.trim_type = trim_type
		self.trim_value_user = trim_value_user
		self.trim_value_movie = trim_value_movie
		self.Bipartite_graph = None
		self.G_users = None
		self.G_movies = None
		self.User_representations = None
		self.Movie_representations = None


	def compute_projection(self, node_type):
		"""Bipartite Graph Projection to a particular node type: 
		   0 for user 1 for movies"""
		if self.Bipartite_graph == None:
			B = nx.Graph()
			B.add_nodes_from(['u-'+str(each) for each in self.Bipartite_data.User.tolist()], bipartite=0) 
			B.add_nodes_from(self.Bipartite_data.Movie.tolist(), bipartite=1) 

			# All nodes are added correctly
			assert len(B.nodes()) == len(set(self.Bipartite_data.User.unique().tolist())) + len(set(self.Bipartite_data.Movie.unique().tolist()))

			edges = [(i,j,k) for i,j,k in zip(['u-'+str(each) for each in self.Bipartite_data.User.tolist()],
		                                  self.Bipartite_data.Movie.tolist(),
		                                  [int(float(each)*2) for each in self.Bipartite_data.Rating.tolist()])]

			B.add_weighted_edges_from([e for e in edges])

			self.Bipartite_graph = B

		_nodes_ = {n for n, d in B.nodes(data=True) if d['bipartite']==node_type}
		# movie_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==1}

		G_projection = bipartite.weighted_projected_graph(B, _nodes_) # weighted_projected_graph: edges weight is the number of shared neighbors
		# G_movies = bipartite.projected_graph(B, movie_nodes)

		if node_type:
			self.G_movies = G_projection
		else:
			self.G_users = G_projection

		node_type_dict = {0:'User', 1:'Movie'}

		print("Projected {} Graph density: {}".format(node_type_dict[node_type],nx.density(G_projection)))
		# return G_projection

	def get_User_representations(self):
		trimmed_user_graph = _trim_graph(self.G_users, self.trim_type, self.trim_value_user)
		self.User_representations = _compute_Node2Vec(trimmed_user_graph)
		return self.User_representations

	def get_Movie_representations(self):
		trimmed_movie_graph = _trim_graph(self.G_movies, self.trim_type, self.trim_value_movie)
		self.Movie_representations = _compute_Node2Vec(trimmed_movie_graph)
		return self.Movie_representations


class CollaborativeFiltering(object):
	"""
	Collaborative Filtering Encoding
	
	CF_data: (pandas dataframe)

	"""
	def __init__(self, CF_data): 
		super(CollaborativeFiltering, self).__init__()
		self.CF_data = CF_data
		self.Ratings = self.CF_data.pivot(index = 'User', columns ='Movie', values = 'Rating').fillna(0)
		self.R = self.Ratings.values
		self.user_ratings_mean = np.mean(self.R, axis = 1)
		self.Ratings_demeaned = self.R - self.user_ratings_mean.reshape(-1, 1)
		self.n_users = self.CF_data.User.unique().shape[0]
		self.n_movies = self.CF_data.Movie.unique().shape[0]
		self.sparsity = round(1.0 - len(self.CF_data) / float(self.n_users * self.n_movies), 3)
		self.algorithm = None
		self.User_representations = None
		self.Movie_representations = None
		self.sigma = None

		print('The sparsity level of training dataset is ' +  str(self.sparsity * 100) + '%')


	def compute_latent_factors(self, algorithm='SVD', k=64):
		(U, sigma, Vt) = (None, None, None)
		self.algorithm = algorithm
		if algorithm == 'SVD':
			(U, sigma, Vt) = _svd_compute(self.Ratings_demeaned, k)
			self.User_representations = U
			self.Movie_representations = Vt.T
			self.sigma = np.diag(sigma)
		return self.User_representations, self.Movie_representations, self.sigma









