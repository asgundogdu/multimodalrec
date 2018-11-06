# from .video import VideoEncoder
from .image import extract_features
# from .audio import AudioEncoder


# LSTM input generation

class AudioVisualEncoder(object):
	"""docstring for AudioVisualEncoder"""
	def __init__(self, Audial_data=None, Visual_data=None, extraction_type='Pretrained', pretrained_model='ImageNet'):
		super(AudioVisualEncoder, self).__init__()
		self.Audial_data = Audial_data
		self.Visual_data = Visual_data
		self.extraction_type = extraction_type
		self.pretrained_model = pretrained_model
		
		# self.graph = None
		self.movie_vis_features = None
		self.movie_aud_features = None


	def extract_audial_features(self):
		"""Generates representation vector per second"""
		self.movie_aud_features = AudioEncoder(Audial_data)
		return self.movie_aud_features


	def extract_visual_features(self, _dir_='data/', load=False):
		"""Generates representation vector per second"""
		self.movie_vis_features = extract_features(_dir_=_dir_, load=load)#(Visual_data, extraction_type, pretrained_model)
		return self.movie_vis_features


	def process_data(self):
		pass


		