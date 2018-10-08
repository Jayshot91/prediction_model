# NHL Prediction Model

This is a prediction model for the NHL that uses Monte Carlo methods to predict
the winning probability of the home team for each game in the season. It does this by
sampling poisson distributions of each teams former scoring and comparing the
goals scored by the home and away teams and determing who has more goals.

The scripts in here run a 10,000 iteration simulation for each game and store it
to an SQL table in order to track the predictions and calculate accuracy and
log loss metrics. It also graphs the distributions of the probabilites and then
posts the results of the simulation and graph to Twitter.

If you have any questions please contact me at barloweanalytics@gmail.com
or @matt_barlowe on Twitter.
