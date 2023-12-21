# Predict-Pricing-of-Rental-Properties-on-Airbnb

## Problem and dataset description
Pricing a rental property such as an apartment or house on Airbnb is a difficult challenge. A model that accurately predicts the price can potentially help renters and hosts on the platform make better decisions. In this assignment, your task is to train a model that takes features of a listing as input and predicts the price.

The [dataset](https://drive.google.com/drive/folders/1LT_jmRlScpSZKn9Y3W8rJnIXIF2zBEu_?usp=sharing) provided is collected from the Airbnb website for New York, which has a total of 29,985 entries, each with 764 features. After the model being built, the model will be train with same provided dataset, and will evaluate it on 2 other test sets (one public, and one hidden during the challenge).

Some minimal data cleaning were already done, such as converting text fields into categorical values and getting rid of the NaN values. To convert text fields into categorical values, we used different strategies depending on the field. For example, sentiment analysis was applied to convert user reviews to numerical values ('comments' column). We added different columns for state names, '1' indicating the location of the property. Column names are included in the data files and are mostly descriptive.

Also in this data cleaning step, the price value that we are trying to predict is calculated by taking the log of original price. Hence, the minimum value for our output price is around 2.302 and maximum value is around 9.21 on the training set.

## Structure
The codebook 'main.py' is the final model with XGBoost and some feature extractor created. You can download and run with datafiles in the same folder. To download dataset, you can click [here](https://drive.google.com/drive/folders/1LT_jmRlScpSZKn9Y3W8rJnIXIF2zBEu_?usp=sharing).
