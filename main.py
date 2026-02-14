#MAIN FUNCTION

#IMPORTS
from get_list_of_kalshi_games import get_list_of_all_kalshi_college_basketball_games
from get_list_of_espn_games import get_list_of_all_espn_college_basketball_games
from get_espn_game_timestamp_mapings import get_espn_game_timestamp_mapping

#GET LIST OF ALL KALSHI CLOSED MARKET COLLEGE BASKETBALL GAMES
kalshi_games_list = get_list_of_all_kalshi_college_basketball_games()

#GET LIST OF ALL ESPN COMPLETED COLLEGE BASKETBALL GAMES
espn_games_list = get_list_of_all_espn_college_basketball_games()

#CREATE A CSV FILE THAT MAPS OF KALSHI GAME ID TO ESPN GAME ID
map_espn_kalshi_games()

#FOR EACH KALSHI GAME THAT IS ALSO IN ESPN GAME

    #PULL THE ESPN GAME TIMESTAMP MAPPING FOR EACH GAME
    #get_espn_game_timestamp_mapping(espn_game_id)

    #PULL THE NECESSARY KALSHI GAME DATA FOR EACH GAME
    #get_kalshi_game_data(event_ticker, team_abbreviation)

    #MERGE THE DATA STRUCTURES INTO A SINGLE DATAFRAME
    

