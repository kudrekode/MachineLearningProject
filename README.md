# MachineLearningProject
Comparing different ML methods in evaluating whether audio features of a song impact a songs popularity 

**Summary**

I used the spotify API to collect songs from the website and put them into a pandas dataframe.
Using typical techniques in seaborn and matlab I visualised the data in order to inform feature selection, dimensionality reduction etc.
I then cleaned the data appropriately through dropping duplicates and standardising.
I then used a Linear Regression model, Decision Tree, Random Forest, Gradient Boost and CatBoost and compared efficiacy on predicting a target. 
I optimised through GridSearchCV and hyperparameter tuned the ensemble methods. 
This improved their scores but not drastically. 
Ultimately the RandomForest with optimisation performed the best.
I then used SHAP to analyse the data further and conclude what features in the data set were having the most positive or negative impacts on the models. 

