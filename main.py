#MAIN FUNCTION

#IMPORTS
import csv
import pandas as pd
from get_list_of_kalshi_games import get_list_of_all_kalshi_college_basketball_games
from get_list_of_espn_games import get_list_of_all_espn_college_basketball_games
from kalshi_espn_game_mapper import map_espn_kalshi_games
from get_espn_game_timestamp_mapings import get_espn_game_timestamp_mapping
from get_kalshi_game_data import get_kalshi_game_data

#GET LIST OF ALL KALSHI CLOSED MARKET COLLEGE BASKETBALL GAMES
#kalshi_games_list = get_list_of_all_kalshi_college_basketball_games()

#GET LIST OF ALL ESPN COMPLETED COLLEGE BASKETBALL GAMES
#espn_games_list = get_list_of_all_espn_college_basketball_games()

#CREATE A CSV FILE THAT MAPS OF KALSHI GAME ID TO ESPN GAME ID
#mapped_games = map_espn_kalshi_games()
mapped_games = {'KXNCAAMBGAME-25NOV03LEHHOU': '401824809', 'KXNCAAMBGAME-25NOV03ARIZFLA': '401826885', 'KXNCAAMBGAME-25NOV03NHCCONN': '401812785', 'KXNCAAMBGAME-25NOV03QUINSJU': '401820577', 'KXNCAAMBGAME-25NOV03OAKMICH': '401826083', 'KXNCAAMBGAME-25NOV03VILLBYU': '401819834', 'KXNCAAMBGAME-25NOV03SCSTLOU': '401817239', 'KXNCAAMBGAME-25NOV03EWUUCLA': '401813756', 'KXNCAAMBGAME-25NOV03SOUARK': '401826784', 'KXNCAAMBGAME-25NOV03UNDALA': '401812260'}

with open("team_mapping.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["kalshi_id", "espn_id"])
    
    for kalshi_id, espn_id in mapped_games.items():
        writer.writerow({'kalshi_id': kalshi_id, 'espn_id': espn_id})

print("CSV saved as team_mapping.csv")

gameData = pd.DataFrame()
results_list = []

#FOR EACH KALSHI GAME THAT IS ALSO IN ESPN GAME
for kalshi_id, espn_game_id in mapped_games.items():
    #PULL THE ESPN GAME TIMESTAMP MAPPING FOR EACH GAME

    espn_df = get_espn_game_timestamp_mapping(espn_game_id)
    espn_df['wallclock_ts'] = pd.to_datetime(espn_df['wallclock_ts']).dt.tz_localize(None)
    espn_df = espn_df.sort_values('wallclock_ts')

    #PULL THE NECESSARY KALSHI GAME DATA FOR EACH GAME
    with open("list_of_espn_games.txt", "r") as f:
        for line in f:
            if line.startswith(espn_game_id):
                line = line.strip()
                break

    team = line.split(",")[1]
    print("HERE", team)
    kalshi_df = get_kalshi_game_data(kalshi_id, team)
    kalshi_df['wallclock_ts'] = pd.to_datetime(kalshi_df['wallclock_ts']).dt.tz_localize(None)
    kalshi_df = kalshi_df.sort_values('wallclock_ts')
    
    merged_game_data = pd.merge_asof(
            kalshi_df, 
            espn_df, 
            on='wallclock_ts', 
            direction='backward'
        )

    results_list.append(merged_game_data)

if results_list:
    #MERGE THE DATA STRUCTURES INTO A SINGLE DATAFRAME
    gameData = pd.concat(results_list, ignore_index=True)

    gameData.to_csv("scraped_game_results.csv", index=False)
    print("File saved successfully!")
