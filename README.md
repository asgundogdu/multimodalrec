# MultimodalRec
Deep Multi-Modal Movie Trailer Recommendation System built in Python

## What does MultimodalRec do?

MultimodalRec is a Python movie recommendation system that allows you to solve cold start problem for predicting users behaviour on brand new movies.  

## Data Sources

MultimodalRec takes two sources of data: `video_trailers` and `user_viewership_history`. By learning users' audio-visual feature preferences along with their rating history, the modal predicts the likelihood of users to watch a brand new movie trailer. This can be used for marketing purposes...

For detail info and its usage, please see the [cyclopedia.](https://github.com/asgundogdu/multimodalrec/tree/master/cyclopedia/RelatedWorks)

### The dimensions of data 

#### 1-dimensional Convolutions

Frame representations are input into neural networks a 2048 lenght vector. This vector is encoded in such a way as to capture some aspect of the sentiment of the frame. So, each frame input into the CNN below will be a 2048 lenght vector. Afterwards, as we will be inputting a sequence of frames into network, for each input row we will be inputting 60 of these frame vectors. So the input for each row will be (60, 2048) in size. Eventually, with Tensorflow, we can process batches of data using multidimensional tensors. If we have a batch size of 64, out *training* input data for visual modality input will be (64 x 60 x 2048), where the batch size is the first dim of the tensor. 

To represent each audio feature of the trailer input we use 100-dimensional [MFCC features](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), we aggregate each feature in 1 second level resulting (60, 100) in size. If we have a batch size of 64, out *training* input data for audio modality input will be (64 x 60 x 100).

For each trailer our framework colvolve over time channel using different filter sizes (determined by CV).

Note the small batch size is to allow a more stochastic gradient descent which will prevent to be stucking in local mimina during training for many iterations.  

#### User Latent Factors

We use Colloborative Filtering model's user latent factors to represent each user as 100 dimensional vectors, the movie features are extracted by SVD model and weights (user features) are infered using ALS.

### Multimodal Fusion

We use [Gated Multimodal Unit](https://openreview.net/pdf?id=Hy-2G6ile) to capture an intermediate representation based on a combination of data from different modalities being (audio, image, and user representations). 

## Evaluation Metrics

Comparing performance of proposed model vs. existing approaches. The evaluations are performed by applying: 

1- RMSE, distance between predicted preferences and true preferences over items.
2- Recall, the portion of favored items that were suggested

### Simulation for Online Experimental Setup

Hiding partial historical data using timestamps of ratings.

The goal is to simulate sets of past user selections that are represenatative of what the system will face when deployed. For example, sample a set of test users, then sample a single test time, and hide all items after the sampled test time for each test user. This simulates a situation where the RS is trained as of the test time, and then makes recommendations without taking into account any new data that arrives after test time. (sample a test time for each test user Leave one out approach)

### Significance Testing on Model Development

In order to perform a significance test that algorithm A is indeed better than algorithm B, we require the results of several independent experiments comparing A and B. The protocol we have chosen in generating our test data ensures that we will have this set of results.

Given paired per-user performance measures for algorithms A and B the simplest test of significance is the sign test (Demsar, 2006). 

In this test, we count the number of users for whom algorithm A outperforms algorithm B ($n_A$) and the number of users for whom algorithm B outperforms algorithm A ($n_B$). The probability that A is not truly better than B is estimated as the probability of at least $n_A$ out of $n_A + n_B$ 0.5-probability Binomial trials succeeding (that is, $n_A$ out of $n_A + n_B$ fair coin-flips coming up “heads”).

### Evaluation Tasks

Root of the Mean Square Error (RMSE), Mean Average Error (MAE) and Normalized Mean Average Error (NMAE)

RMSE tends to penalize larger errors more severely than the other metwoks, whereas NMAE normalizes MAE by the range of the ratings for ease of comparing arrors across domains.

RMSE can be more suitable (than MEA) for this task as it measures inaccuracues on all ratings either negative or positive. However sometimes predicting 1star or 2star mat not be as important if we consider 1 and 2 stars as low values. Thus it is trickly to use this metric.

For the task of recommending items, we are onlt interested in binary ratings either 1 or 0. 

Confusion matrix:

|               | Recommended                | Not recommended |
| ------------- | -------------------------- | --------------- |
| Preferred     | TP                         | FN              |
| Not preffered | FP                         | TN              |

Precision: what proportion of their recommendations were actually suitable for the user.

Recall may not be as relevant.

ROC Curves

*Note*: Knowledge of the application will dictate which region of the curve the decision will be based on. For example, in the “recommend some good items” task it is likely that we will prefer a system with a high precision, while in the “recommend all good items” task, a higher recall rate is more important than precision. 

F-score

## Design of an Offline Experiment

How data is splitted, which measurement should be taken, how to determine if differences in performaance are statistically significant etc.?

## Baselines

CF modelling - no metadata used "Users who prefered this item also prefer this"
Content-based modelling - learning set of item features through metadata "Similar Items" (Can build baseline model with movie genres, movie plots, etc.)

### CF vs MultimodalRec

To evaluate MultimodalRec with CF model, we aim to create a common test set where we do not target the movie cold-start problem. (CF models unable to work effectively for cold-start problems)

Each user and movie in the test set will be exist in the training set where we seperately train CF and MultimodalRec models. Intuitively, CF uses user and movie latent factors and makes inference with matrix multiplication, whereas MultimodalRec uses user latent factors and trailer features (visual and audio) and make inference through our developed framework.

### Content-based vs MultimodalRec

For this task we focus on movie cold-start problem where the test set does not include any movies exist in the training set and compare both performances.








