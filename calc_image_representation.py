from multimodalrec.multimodalrec import MultimodalRec
from multimodalrec.multimodalrec import data_pipeline
from multimodalrec.model import model

def main():
	recmodel = MultimodalRec()
	recmodel.organize_multimodal_data(load=False, dataset=10, 
		trailer_directory='/home/ubuntu/gits/multimodalrec/croped_frames/', 
		sequence_dir = '/home/ubuntu/gits/multimodalrec/')

if __name__ == '__main__':
	main()