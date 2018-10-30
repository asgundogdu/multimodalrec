import os, sys, pickle
import pandas as pd
import numpy as np
from collections import Counter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

from node2vec import Node2Vec

from .data import data_creation as create


class BipartiteNetwork(object):
	"""docstring for BipartiteNetwork"""
	def __init__(self, arg):
		super(BipartiteNetwork, self).__init__()
		self.arg = arg
