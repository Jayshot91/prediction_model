'''
This model will predict the probabilites of winners in NHL games by using the
poisson distirbution and monte carlo sampling
'''
import os
import sys
import datetime
import statistics
import time
import logging
import math
import multiprocessing as mp
import seaborn as sns
from get_today_schedule import get_today_sched
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from twython import Twython


def get_twitter_keys(key_file):
    '''
    Funciton to get the twitter bot's api keys from a text file

    Input:
    key_file - text file holding api auth keys

    Output:
    keys_dict - dictionary with key names mapped to keys
    '''

    twitter_keys = key_file
    keys_dict = {}

    with open(twitter_keys, 'r') as keys:
        api_keys = []

        for line in keys:
            api_keys.append(line)

        api_keys = list(map(str.strip, api_keys))

        for key in api_keys:
            name_list = key.split(':')
            name_list = list(map(str.strip, name_list))
            key_name, key_value = name_list[0], name_list[1]
            keys_dict[key_name] = key_value
        return keys_dict

def tweet_results(home_team, away_team, date, final_win_probs, dist_plot_file_name):
    '''
    this function tweets out the results of the prediciton model

    Inputs:
    df - dataframe of the results of the prediction model

    Outputs:
    None
    '''

    #twitter_keys = get_twitter_keys(sys.argv[1])
    twitter_keys = get_twitter_keys(sys.argv[1])

    #set twitter API
    twitter = Twython(twitter_keys['Consumer Key'], twitter_keys['Consumer Secret Key'],
                      twitter_keys['Access Key'], twitter_keys['Access Secret Key'])

    tweet_string = f'{away_team} @ {home_team} {date}:\n'
    tweet_string += f'\n{home_team}: {round(final_win_probs * 100, 1)}%\n{away_team}: {round((1-final_win_probs) * 100, 1)}%\n'

    photo = open(dist_plot_file_name, 'rb')
    response = twitter.upload_media(media=photo)
    twitter.update_status(status=tweet_string, media_ids=[response['media_id']])
    photo.close()



def sched_insert(df):

    print('Inserting DataFrame to the Database')
    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    df.to_sql('predictions', schema='nhl_tables', con=engine,
              if_exists='append', index=False)

def get_avg_df(df):
    '''
    This is to create an average of the time frame of the past results. This
    was only implemented because of Vegas who had no past results as an expansion
    team. I'm hoping to use this most likely with the new Seatle expansion team
    in the next couple years.

    Inputs:
    df - dataframe of past results

    Outputs:
    avg_df - a dataframe of avg results for games with no OT, OT and no Shootout,
             and games that goto shootout
    '''

#creates three different dataframes based on whether the game went to OT, SO,
#or ended in regulation
    reg = df[df.ot_flag != 1]
    ot = df[(df.ot_flag == 1) & (df.shootout_flag != 1)]
    shootout = df[df.shootout_flag == 1]

#creates averages for the results of the three dataframes which reduces them to
#one row
    reg_avg = reg[['home_score', 'away_score','seconds_in_ot']].mean()
    ot_avg = ot[['home_score', 'away_score', 'seconds_in_ot']].mean()
    shootout_avg = shootout[['home_score', 'away_score', 'seconds_in_ot']].mean()

#creates the ot_flag and shootout_flag that is in the table the data is pulled
#from. Will be neccesary for future calculations
    reg_avg['ot_flag'] = 0
    ot_avg['ot_flag'] = 1
    shootout_avg['ot_flag'] = 1

    reg_avg['shootout_flag'] = 0
    ot_avg['shootout_flag'] = 0
    shootout_avg['shootout_flag'] = 1

#combines the three series from average the three dataframes of the three types
#of game outcomes: Regulation, Overtime, Shootout
    avg_df = pd.concat([reg_avg, ot_avg, shootout_avg], axis=1).T

    #rename avg_df columns
    avg_df.columns = ['goals_for', 'goals_against', 'seconds_in_ot', 'ot_flag', 'shootout_flag']

#create columsn for ot and non ot_goals which will be used in the monte carlo
#simulations
    avg_df['non_ot_goals_for'] = np.where(((avg_df.shootout_flag == 1) | (avg_df.ot_flag == 1)) &
                                              (avg_df.goals_for > avg_df.goals_against), avg_df.goals_for - 1,
                                              avg_df.goals_for)
    avg_df['non_ot_goals_against'] = np.where(((avg_df.shootout_flag == 1) | (avg_df.ot_flag == 1)) &
                                              (avg_df.goals_for < avg_df.goals_against), avg_df.goals_against - 1,
                                              avg_df.goals_against)
    avg_df['ot_goals'] = np.where(avg_df.shootout_flag == 0,
                                          avg_df.goals_for - avg_df.non_ot_goals_for, 0)
    avg_df['ot_goals_against'] = np.where((avg_df.ot_flag == 1) & (avg_df.shootout_flag != 1), 1, 0)

    return avg_df

def clean_results(results_df, team, date):
    '''
    this function cleans the results dataframe and just strips out the wanted
    team results and creates a column for OT goals as well
    '''
    results_df = results_df[(results_df.game_date < date) & (results_df.game_type == 'R')]
    cleaned_results = []
#TODO See if this loop could work better as an apply to speed things up 9-24-2018
    #looping through the results_df to pull out only the games the team variable played in.
    for index, row in results_df.iterrows():
        if row.home_team == team:
            #print(row)
            new_row = row[['game_id', 'game_type', 'season', 'game_date', 'home_team_id', 'home_team', 'home_abbrev',
                           'home_score', 'away_score', 'ot_flag', 'shootout_flag', 'seconds_in_ot']]

            #print(new_row)
            new_row.index = ['game_id', 'game_type', 'season', 'game_date', 'team_id', 'team', 'team_abbrev',
                           'goals_for', 'goals_against', 'ot_flag', 'shootout_flag', 'seconds_in_ot']

            new_row['is_home'] = 1

            cleaned_results.append(new_row)

        elif row.away_team == team:
            #print(row)
            new_row = row[['game_id', 'game_type', 'season', 'game_date', 'away_team_id', 'away_team', 'away_abbrev',
                           'away_score', 'home_score', 'ot_flag', 'shootout_flag', 'seconds_in_ot']]
            #print(new_row)
            new_row.index = ['game_id', 'game_type', 'season', 'game_date', 'team_id', 'team', 'team_abbrev',
                           'goals_for', 'goals_against', 'ot_flag', 'shootout_flag', 'seconds_in_ot']

            new_row['is_home'] = 0

            #print(new_row)
            cleaned_results.append(new_row)

    cleaned_df = pd.concat(cleaned_results, axis=1).T

    #calculating non ot goals by seeing if the game went to ot or shootout and if so whether the team won or not.
    #if they did then they score one less goals than their final total if not then they scored their same goals for
    #amount
    cleaned_df['non_ot_goals_for'] = np.where(((cleaned_df.shootout_flag == 1) | (cleaned_df.ot_flag == 1)) &
                                          (cleaned_df.goals_for > cleaned_df.goals_against), cleaned_df.goals_for - 1,
                                          cleaned_df.goals_for)
    cleaned_df['non_ot_goals_against'] = np.where(((cleaned_df.shootout_flag == 1) | (cleaned_df.ot_flag == 1)) &
                                          (cleaned_df.goals_for < cleaned_df.goals_against), cleaned_df.goals_against - 1,
                                          cleaned_df.goals_against)
    cleaned_df['ot_goals'] = np.where(cleaned_df.shootout_flag == 0,
                                      cleaned_df.goals_for - cleaned_df.non_ot_goals_for, 0)
    cleaned_df['ot_goals_against'] = np.where(cleaned_df.shootout_flag == 0,
                                              cleaned_df.goals_against - cleaned_df.non_ot_goals_against,0)

    cleaned_df = cleaned_df.reset_index(drop=True)

    #only return the last two seasons of games
    cleaned_df = cleaned_df.sort_values(by=['game_date'], ascending=False).iloc[:83, :]

    return cleaned_df

def monte_carlo_predict(home_results, away_results):
    '''
    taking the results of each team this function runs a monte carlo
    simulation to predict goals scored using the poissson distribution

    Inputs:
    home_results - results of the last 82 games of the home team
    away_results - results of the last 82 games of the away team

    Outputs:

    home_win_prob - the probability of the home team winning said game
    '''

    results =[]

#create the lambdas for the poisson distribution for regulation by adding the
#home team's goals for and away teams goals against and average. I reverse the
#process for the away team. This lambada is for the interval of one regulation
#game therefore 60 minutes. These are then adjusted for home ice since the avg
#home team wins 55% of the time I increase the home lambad by 5% to account for
#that and decrase the away lambda.
    home_lambda = ((home_results['non_ot_goals_for'].mean()
                    + away_results['non_ot_goals_against'].mean()) / 2) * 1.05

    away_lambda = ((home_results['non_ot_goals_against'].mean()
                    + away_results['non_ot_goals_for'].mean()) / 2) / 1.05

#This creates OT lambdas the same way as above but adjusts for a five minute
#interval which is the length of OT in the NHL by dividing by the total seconds
#in OT to get goal per second and then multiplyting by 300 to get goals per 5
#minutes. I adjust for home and away with the same method as I do above for
#regulation time
    home_ot_lambda = ((((home_results['ot_goals'].sum()/home_results['seconds_in_ot'].sum()) * 300) + \
                      (away_results['ot_goals_against'].sum()/away_results['seconds_in_ot'].sum())*300)/2) * 1.05

    away_ot_lambda = ((((away_results['ot_goals'].sum()/away_results['seconds_in_ot'].sum()) * 300) + \
                      (home_results['ot_goals_against'].sum()/home_results['seconds_in_ot'].sum())*300)/2) / 1.05

#calculate home and away OT win percentages for the Terry-Bradley models that
#determine the probabilites of the binomial flip in OT
    home_ot_win_percent = np.where((home_results.ot_flag == 1) &
                                   (home_results.shootout_flag == 0) &
                                   (home_results.goals_for > home_results.goals_against), 1, 0)\
                                   .sum()/home_results[(home_results.ot_flag == 1)
                                                       & (home_results.shootout_flag == 0)].shape[0]

    away_ot_win_percent = np.where((away_results.ot_flag == 1) &
                                   (away_results.shootout_flag == 0) &
                                   (away_results.goals_for >
                                       away_results.goals_against), 1, 0)\
                                   .sum()/away_results[(away_results.ot_flag == 1) &
                                                       (away_results.shootout_flag == 0)].shape[0]

#calculate shootout percentages for the Bradley-Terry models that determine the
#winner of the Shootout Binomial flip
    home_so_win_percent = np.where((home_results.shootout_flag == 1) &
                                   (home_results.goals_for > home_results.goals_against),
                                   1, 0).sum()/home_results[home_results.shootout_flag == 1].shape[0]

    away_so_win_percent = np.where((away_results.shootout_flag == 1) &
                                   (away_results.goals_for > away_results.goals_against),
                                   1, 0).sum()/away_results[away_results.shootout_flag == 1].shape[0]

#these lines fill any nan results from the OT and SO win percentage calculations
#with an even coin flip
    if math.isnan(home_ot_win_percent):
        home_ot_win_percent = .5

    if math.isnan(away_ot_win_percent):
        away_ot_win_percent = .5

    if math.isnan(home_so_win_percent):
        home_so_win_percent = .5

    if math.isnan(away_so_win_percent):
        away_so_win_percent = .5

#Here we draw a 10000 sample from the poisson distributions for home and away
#teams using the lambdas calculated earlier for regulation goals scored
    home_reg_goals = np.random.poisson(home_lambda, 10000)
    away_reg_goals = np.random.poisson(away_lambda, 10000)

#now I looop through both results and compare them. If the home team goals are
#larger than the away then then I append a one else a zero. If they are tied
#then it goes to an OT and if needed Shootout scenario that are binomial
#distributions decided by home OT and Shootout probability determined by a
#Bradley-Terry Model

#TODO rewrite this to vectorize across the two numpy arrays instead of
#looping 9-24-2018
    for home, away in zip(home_reg_goals, away_reg_goals):
        if home > away:
            results.append(1)
        elif away > home:
            results.append(0)
        else:
            prob_of_zero_goals = (math.exp(-home_ot_lambda) * math.exp(-away_ot_lambda))

            if np.random.binomial(1, prob_of_zero_goals) == 1:
                try:
                    prob_of_home_so_win = home_so_win_percent/(home_so_win_percent + away_so_win_percent)
                except:
                    logging.exception('Shootout probability calculation messed up')
                    prob_of_home_so_win = .5
                results.append(np.random.binomial(1, prob_of_home_so_win))
            else:
                try:
                    prob_of_home_ot_win = home_ot_win_percent/(home_ot_win_percent + away_ot_win_percent)
                except:
                    prob_of_home_ot_win = .5
                    logging.exception('OT win probability calculation messed up')
                results.append(np.random.binomial(1, prob_of_home_ot_win))

#calculate the probability of the home team winning by averaging the wins and
#losses as determined by the Poisson samples and return that probability
    home_win_prob = sum(results)/len(results)

    return home_win_prob

def multi_proc_monte(home_results, away_results, iter=10000):
    '''
    This creates a multiproccess function to use all four cores of my MacBook
    I should add a function to pull a computer's core numbers variable to pass
    to Pool.

    Inputs:
    home_results - home team results dataframe to pass to the monet_carlo_predict
                   function

    away_results - same as the home team but for the away team

    Outputs:
    results - a list of each probability returned by the monte_carlo_predict
              function to be averaged later
    '''

#creates the 4 processes this computer only has four cores so it maxes at four
#change for the number of cores for your computer
    pool =  mp.Pool(os.cpu_count())
#creates a list of process results the length of the iter keyword. I.e. it runs
#the monte_carlo_predict function the number of times as passed to the iter
#keyword in this case it will be a 1000 times
    pool_list = [pool.apply_async(monte_carlo_predict, args=(home_results, away_results)) for _ in range(iter)]

#getting the results for each process after they are finished
    results = [f.get() for f in pool_list]
    pool.close()

    return results


#TODO abstract this to another function that takes date as a variable so I can loop
#over a time frame and store my results
def main():

#set up logger
    logging.basicConfig(filename='prediction.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)

#gets todays date
    date = datetime.datetime.now().strftime('%Y-%m-%d')

#get daily schedule
    daily_sched = get_today_sched(date)

#if schedule is empty exit program
    if daily_sched.empty:
        logging.info('No Games Today')
        return

#Query the results database to return past results
    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    sql_query = 'SELECT * from nhl_tables.nhl_schedule'
    df = pd.read_sql(sql_query, con=engine,
                     parse_dates = {'game_date': '%Y-%m-%d'})
    df = df[df.game_date < date]




#simulation runs using each teams distribution of goals scored in regulation
#for ties determine probability of both teams not scoring in OT from their OT
#poisson goal distributions. USe that p and (1-p) as a bernoulli trial to
#determine if game was decided in OT. If it is decided in OT use Bradley-Terry
#model to determine probability of home team winning with another Bernoulli trial.
#If the game is not decided in OT use another Bradley-Terry model with SO win %'s
#to determine probability of home team winning shootout.
    predictions = []

    for index, row in daily_sched.iterrows():
        game_id = row.game_id

        home_team = row.home_team
        away_team = row.away_team

#getting the sample of results for the home and away teams the try and excepts
#are incase there is an expansion team in the future i.e. Seattle
        try:
            home_results = clean_results(df, home_team, date)

            #vegas bullshit
            if home_results.shape[0] < 2:
                home_results = get_avg_df(df)

        except:
            logging.exception(f'No past results for {home_team}')
            home_results = get_avg_df(df)

        try:
            away_results = clean_results(df, away_team, date)

            #more vegas bullshit
            if away_results.shape[0] < 2:
                away_results = get_avg_df(df)

        except:
            away_results = get_avg_df(df)
            logging.exception(f'No past results for {away_team}')

        #running the monte carlo simulation a 10,000 times
        #and printing outputs for testing
        logging.info(f'{away_team} vs. {home_team}')
        logging.info(date)
        logging.info(row.game_id)

        start_time = time.time()
#runs the multi proc function to calculate home team win probs
        home_win_probabilities = multi_proc_monte(home_results, away_results)
        end_time = time.time()
        time_to_run = end_time - start_time

        logging.info(f'Time for simulation to run: {time_to_run}')
        logging.info(sum(home_win_probabilities)/len(home_win_probabilities))

#calculates final win probablilty by average all the probabilities from the
#monte carlo sim
        final_win_probs = sum(home_win_probabilities)/len(home_win_probabilities)

#appends to the predictions list so I can
        try:
            predictions.append([game_id, date, home_team, home_results.team_abbrev.unique()[0], away_team, away_results.team_abbrev.unique()[0], final_win_probs])
        except AttributeError as e:
            logging.exception(f"This is for stupid foreign teams\n{e}")
            continue

#plotting the distributions of the results from the monte carlo sim and saving
#it to a file to tweet
        plt.figure()
        sns.distplot(home_win_probabilities).set_title(f'Distribution of {home_results.team_abbrev.unique()[0]} vs. {away_results.team_abbrev.unique()[0]} on {date} with 95% CI')
        plt.xlabel(f'{home_results.team_abbrev.unique()[0]} win probabilities')
        plt.ylabel('Counts')
        plt.axvline(final_win_probs + (statistics.stdev(home_win_probabilities) * 1.96))
        plt.axvline(final_win_probs - (statistics.stdev(home_win_probabilities) * 1.96))
        dist_plot_file_name = 'dist_plot.png'
        plt.savefig(dist_plot_file_name)
        logging.info(f"Plot created for {game_id}")

        tweet_results(home_team, away_team, date, final_win_probs, dist_plot_file_name)
        logging.info(f"Results tweeted out for {game_id}")


    #create a dataframe out of the predictions to merge with the schedule dataframe
    predict_df = pd.DataFrame(predictions)
    predict_df.columns = ['game_id', 'game_date', 'home_team', 'home_abbrev', 'away_team', 'away_abbrev', 'home_win_probs']

    final_predict_df = daily_sched.merge(predict_df[['game_id', 'home_win_probs', 'home_abbrev', 'away_abbrev']],
                                         on='game_id')

#create a prediction column to calculate accuracy
    final_predict_df['home_win_pred'] = np.where(final_predict_df.home_win_probs > .5, 1, 0)

    final_predict_df = final_predict_df[['game_id', 'game_date', 'home_team',
                                         'home_abbrev', 'home_team_id',
                                         'away_team', 'away_abbrev', 'away_team_id',
                                         'home_win_probs', 'home_win_pred']]

#insert results into the results table on the database
    sched_insert(final_predict_df)
    logging.info("Predictions have been inserted into prediction table")


    return

if __name__ == '__main__':
    main()

