"""Run using command: (input file "ml-youtube.csv" from movieLens dataset)
python3 trailer_scrapper.py <link-to-the-trailer-id-list-file> <output-directory-for-trailers-t0-be-downloaded> <error-file-to-be-created>(OPTIONAL)"""

from pytube import YouTube
import pandas as pd
import time, os, re
import numpy as np


def _print_some_info(output_path):
	filmlist = os.listdir(output_path)
	# Number of films in 2000s
	pattern = re.compile(r"\((20\d\d)\)")
	years = Counter([item for sublist in [pattern.findall(each) for each in filmlist] for item in sublist])
	print("Number of trailers produced in year {d}".format(sum([pair[1] for pair in sorted(years.items(), key=lambda pair: pair[1], reverse=True)])))

	# Average screen sizes NEEDS cv2
	# import cv2
	# width_list = []
	# height_list = []
	# for enum, film in enumerate(filmlist):
	#     file_path = output_path+film
	#     vid = cv2.VideoCapture(file_path)
	#     height_list.append(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	#     width_list.append(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	#     if enum%1000==0: print(enum)
	# print("Average screen width".format(np.mean(width_list)))
	# print("Average screeb height".format(np.std(width_list)))

def _yt_downloader(link, title, movieId, output_path):
    title = title.replace(' ','_')
    try:
        yt = YouTube(link)
        yt.streams.filter(progressive=True, file_extension='mp4').order_by(
            'resolution').desc().first().download(
            output_path=output_path, filename=title+'_'+str(movieId)) # Download the highest resolution available in .mp4 format
    except:
        print('{:05d} : {:s} NOT valid'.format(movieId, link))
        return True


def _download_trailers(in_dir, out_path, error_dir):
	ml_youtube = pd.read_csv(in_dir)
	if ml_youtube.shape[1] == 3 and ml_youtube.shape[0]>0:
		start_time = time.time()
		errors = {} # To track depriciated/invalid/unavailable links

		for enum, each_row in enumerate(ml_youtube.iterrows()):
		    link, movieId, title = 'https://www.youtube.com/watch?v='+each_row[1][0], each_row[1][1], each_row[1][2]
		    error = _yt_downloader(link, title, movieId, out_path)
		    if error: errors[movieId] = (link, title)

		    ############################################################
		    if each_row[0]%500==0:
		        print(enum)
		        print(each_row[0])
		        print("--- %s seconds ---" % (time.time() - start_time))
		        start_time = time.time()
	        ############################################################

	    with open(error_dir,'w') as data:
		    data.write(str(errors))

		filmlist = os.listdir(out_path)
		print("Number of downloaded trailers: {d}".format(len(filmlist)))
		_print_some_info(out_path)
	else:
		print("Trailer list can not be found!")



def main():
	if len(sys.argv) > 1:
	    links_dir = sys.argv[1]
	    output_path = sys.argv[2]
	    error_dir = 'errors.txt'
	    if sys.argv[3] != None:
	    	error_dir = sys.argv[3]
	    _download_trailers(links_dir, output_path, error_dir)

	else:
	    print('Please mention the trailer directory!')


if __name__ == '__main__':
  main()