# MSAS_Cricket_Win_Prob
Using logistic regression models, I created a chasing win probability model and a model to structure the ideal chase

## Overview
All data was collected from cricsheet’s json collection of matches and filtered to include only their T20 matches
The data was collated together and preprocessed to create an array of states after every over
Datapoints were weighted based on how close the match was, so as to prevent blowouts from having a significant influence
For win probability, each state contained: Runs remaining, Overs remaining, Wickets remaining, Batter scores, Current run rate, Required run rate
For the chasing strategy model, each state contained the above parameters with pressure(RRR-CRR) swapping in for batter scores

## Win Probability Model
Using SKlearn, a logistic regression classifier was used to model the win probability with the states mentioned earlier
The data was partitioned into an 85-15 training/test split ratio with overs in a match being kept together
The model used 5-fold cross validation
Multiple models were trained using different logistic regression parameters and the best performing model was selected

## Chase Modeling Model
Predicts how many runs a team that successfully chased its total would score in the next over from its given state
Can iteratively build an optimal chase by predicting over by over
Trained using SKlearn random forest regression with states as inputs and runs scored that over as outputs
The data was partitioned into an 85-15 training/test split ratio with overs in a match being kept together
The model used 5-fold cross validation
Multiple models were trained using different random forest parameters and the best performing model was selected



