import sys
from datetime import date, datetime, timedelta
import os
import requests
import logging
import pandas as pd
import psycopg2
from sqlalchemy import create_engine


def get_yest_schedule(date):
    '''
    This function gets the NHL schedule from the NHL api and
    returns a dictionary

    Inputs:
    start_date - string of the first date to pass to the api url
    end_date - string of the end date for the api url

    Outputs:
    schedule_dict - dictionary created from api JSON
    '''

    api_url = ('https://statsapi.web.nhl.com/api/v1/schedule?'
               'date={}').format(date)
    logging.info(api_url)

    req = requests.get(api_url)
    schedule_dict = req.json()

    return schedule_dict

def get_game_ids(schedule_dict):
    '''
    This function flatten out the API json into a flat table scructure
    with the relevant stats for the SQL table

    Inputs:
    schedule_dict - dicitonary of the API GET request

    Outputs
    sched_df - pandas dataframe to be inserted into schedule table
    '''

    game_ids = []
    for item in schedule_dict['dates']:
        games = item['games']

        for game in games:
            game_ids.append(game['gamePk'])

    return game_ids

def sched_insert(df):

    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    df.to_sql('nhl_schedule', schema='nhl_tables', con=engine,
              if_exists='append', index=False)

    logging.info('Data inserted to the Database')

def create_sched_df(pbp_dict, date):
    '''
    this function takes a pbp JSON object and converts it into a list of values
    that will be compiled into a dataframe to be inserted into SQL table

    Inputs:
    game_dict - pbp JSON

    Outputs:
    outcome - list of results of game that will become row in data frame
    '''

    outcome = []
    linescore = pbp_dict['liveData']['linescore']

    outcome.append(pbp_dict['gamePk'])
    outcome.append(pbp_dict['gameData']['game']['type'])
    outcome.append(pbp_dict['gameData']['game']['season'])
    outcome.append(date)
    outcome.append(pbp_dict['liveData']['linescore']['teams']['home']['team']['id'])
    outcome.append(pbp_dict['liveData']['linescore']['teams']['home']['team']['name'])
    outcome.append(pbp_dict['liveData']['linescore']['teams']['home']['team']['abbreviation'])
    outcome.append(pbp_dict['liveData']['linescore']['teams']['home']['goals'])
    outcome.append(pbp_dict['liveData']['linescore']['teams']['away']['team']['id'])
    outcome.append(pbp_dict['liveData']['linescore']['teams']['away']['team']['name'])
    outcome.append(pbp_dict['liveData']['linescore']['teams']['away']['team']['abbreviation'])
    outcome.append(pbp_dict['liveData']['linescore']['teams']['away']['goals'])
    if pbp_dict['liveData']['linescore']['currentPeriod'] == 4:
        outcome.append(1)
    else:
        outcome.append(0)

    if pbp_dict['liveData']['linescore']['currentPeriod'] == 5:
        outcome.append(1)
    else:
        outcome.append(0)

    if pbp_dict['liveData']['linescore']['currentPeriod'] == 4:
        try:
            game_end_time = pbp_dict['liveData']['plays']['currentPlay']['about']['periodTime'].split(':')
            seconds = int(game_end_time[0]) * 60 + int(game_end_time[1])
            outcome.append(seconds)
        except KeyError:
            logging.exception('Error in NHL pbp')
            outcome.append(0)

    elif pbp_dict['liveData']['linescore']['currentPeriod'] == 5:
        outcome.append(300)

    else:
        outcome.append(0)

    if outcome[6] > outcome[9]:
        outcome.append(1)
    else:
        outcome.append(0)

    return outcome

def get_pbp(game_id):
    '''
    This function gets the NHL schedule from the NHL api and
    returns a dictionary

    Inputs:
    start_date - string of the first date to pass to the api url
    end_date - string of the end date for the api url

    Outputs:
    schedule_dict - dictionary created from api JSON
    '''

    api_url = f'http://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live'

    logging.info(api_url)
    req = requests.get(api_url)
    schedule_dict = req.json()

    return schedule_dict

def main():
    '''
    This script pulls the schedule data of past games and the results
    of each game and inserts them into an Postgres table
    '''
    date = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')


    logging.basicConfig(filename='results.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)
    rows = []
    schedule_dict = get_yest_schedule(date)
    games = get_game_ids(schedule_dict)

    if schedule_dict['totalItems'] == 0:
        logging.info("No Games Today")
        return
    else:
        for game in games:
            try:
                pbp_dict = get_pbp(game)
                rows.append(create_sched_df(pbp_dict, date))
            except:
                logging.exception('Exception')
                continue

    sched_df_columns = ['game_id', 'game_type', 'season', 'game_date',
                        'home_team_id', 'home_team', 'home_abbrev', 'home_score',
                        'away_team_id', 'away_team', 'away_abbrev', 'away_score',
                        'ot_flag', 'shootout_flag', 'seconds_in_ot',
                        'home_win']

    sched_df = pd.DataFrame(rows, columns=sched_df_columns)
    logging.info(sched_df)

    sched_insert(sched_df)


if __name__ == '__main__':
    main()
