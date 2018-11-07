# MultimodalRec
Deep Multi-Modal Movie Trailer Recommendation System built in Python

## What does MultimodalRec do?

MultimodalRec is a Python movie recommendation system that allows you to solve cold start problem for predicting users behaviour on brand new movies.  

## Data Sources

MultimodalRec takes two sources of data: `video_trailers` and `user_viewership_history`. By learning users' audio-visual feature preferences along with their rating history, the modal predicts the likelihood of users to watch a brand new movie trailer. This can be used for marketing purposes...

For detail info and its usage, please see the [cyclopedia.](https://github.com/asgundogdu/multimodalrec)

### The dimensions of data 

#### LSTM Cell

Frame representations are input into neural networks a 2048 lenght vector. This vector is encoded in such a way as to capture some aspect of the sentiment of the frame. So, each frame input into the LSTM network below will be a 2048 lenght vector. Afterwards, as we will be inputting a sequence of frames into LSTM, for each input row we will be inputting 30 of these frame vectors. So the input for each row will be (30, 2048) in size. Eventually, with Tensorflow, we can process batches of data using multidimensional tensors. If we have a batch size of 64, out *training* input data for LSTM will be (64 x 30 x 2048), where the batch size is the first dim of the tensor. 

Note the small batch size is to allow a more stochastic gradient descent which will prevent to be stucking in local mimina during training for many iterations.  







