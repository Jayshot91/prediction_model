import datetime
import os
import requests
import pandas as pd


def get_today_sched(date):
    '''
    this function gets the schedule of games that are set to be played on the
    date pased to it in the NHL

    Input:
    date - date to get schedule

    Ouput:
    games - dictionary of games to be played
    '''

    url = f'https://statsapi.web.nhl.com/api/v1/schedule?date={date}'

    req = requests.get(url)
    schedule_dict = req.json()

    today_games = {}

#if there are no games exit the function with an empty dataframe
    if not schedule_dict['dates']:
        return pd.DataFrame()

#parse the schedule to create a dataframe to feed to the prediction model
    for x in schedule_dict['dates']:
        for game in x['games']:
            today_games[game['gamePk']] = {}
            today_games[game['gamePk']]['date'] = date
            today_games[game['gamePk']]['home_team'] = game['teams']['home']['team']['name']
            today_games[game['gamePk']]['home_team_id'] = game['teams']['home']['team']['id']
            today_games[game['gamePk']]['away_team'] = game['teams']['away']['team']['name']
            today_games[game['gamePk']]['away_team_id'] = game['teams']['away']['team']['id']

#turn dictionary of daily games to a dataframe:
    daily_games_df = pd.DataFrame.from_dict(today_games, orient='index')
    daily_games_df = daily_games_df.reset_index()
    daily_games_df.columns = ['game_id', 'game_date', 'home_team', 'home_team_id',
                              'away_team', 'away_team_id']

    return daily_games_df

def main():
    get_today_sched('2018-10-04')

if __name__ == '__main__':
    main()
