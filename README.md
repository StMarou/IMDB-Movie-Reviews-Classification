
# IMDB Movie Reviews Sentiment Classification

## Description
Using various machine learning algorithms, an ensemble classifier 
was built, that can predict if a movie review is positive or 
negative. The classifier achived 90% accuracy on the test set (20%
of the reviews).

## Dataset
The dataset (http://ai.stanford.edu/~amaas/data/sentiment/)
contains 50k highly polar movie reviews. The dataset is balanced,
25k reviews are negative and 25k are positive. 

## Process
First, the dataset was cleaned. Any punctuation, special character, 
number and stop-word was removed. Each word was also lemmatized 
using *NLTK*'s lemmatizer.  

Then, after splitting the dataset into a training and a test set 
(80%-20%), 8 different machine learning algorithms were 
trained and tuned (using cross-validation). For each algorithm, 
a pipeline was created that transformed the reviews into a TF-IDF 
matrix, selected the top *k* best features using the chi-squared 
statistic and finally fitted each model. *sklearn*'s 
*RandomizedSearchCV* was used to tune the hyperparameters of 
each model and preprocessing step. The accuracy of each model on 
the validation sets is presented below:  

[![cross-val-scores.png](https://i.postimg.cc/JzFzvqy2/cross-val-scores.png)](https://postimg.cc/sMpzZ54Y) 

Finally, using the top 4 best performing models, a *voting 
classifier* was built, which predicted the class label using the
argmax of the sums of the predicted probabilities.



