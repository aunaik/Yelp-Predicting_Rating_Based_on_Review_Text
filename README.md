# Yelp - Predicting Rating Based on Review Text

Authors: Akshay Naik, Jigar Madia, Praneta Paithankar

Most of the shopping or information websites available today are equipped with Rating and Review feature which allows users to share their own experience about a certain product or service and also provide an overall rating, mostly out of 5. These ratings provide a method to estimate an overall rating of the product which can be used to provide users with a single rating to judge a productÎéÎ÷s worthiness. However the main problem with ratings these days is that they are heavily biased on the user's perspective and opinion which may not be actually true or it may be influenced by a certain feature which only miniscule percent of users consider important. For example there may be users who always insist on impeccable service and provide bad ratings for less pricey restaurants which are not service oriented and generally look out to please their customers with their food. In this case a few bad ratings can impact the overall rating of the restaurant.
Another problem arises when users generally follow a trend in their ratings. It may happen that certain users have an overall good experience but they never give out 5 star ratings and generally follow a pattern of at max 3.5 - 4 rating even if they like everything or that some users always give out good ratings even though they did not have a good experience but are generally used to giving ratings no less than 3. In such case it is essential to try and consider all these factors when we calculate overall ratings.
One solution which we feel may help here is to consider the other part of this system which allows users to provide a brief or detailed review of their experience. When yours describe it using natural language, it becomes easier to understand how their overall experience was in an unbiased manner. This is the feature we want to leverage in our project. We aim to create a system which takes users reviews and try to create a model which can accurately and in an unbiased manner predict rating for that particular review. This will not only help us provide independence to overall rating calculation but also provide a uniformity in judging all products and services on a single platform. In the next few sections we provide some details as to how we aim to achieve this solution. Our overall objective in this project is to remove any user bias which may exist in the rating using reviews.

Note: You can find details for our exxperiment in the report.

## Dataset:

All the data files are uploaded in a google drive link [here] (https://drive.google.com/open?id=1ek6N0g1M8I3bk2sOAg13PSv4R6Rq2qqS)

To run any of the programs, these files will need to be downloaded and present in the same directory as program files.
