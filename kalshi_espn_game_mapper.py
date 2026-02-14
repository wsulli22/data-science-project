

#call the main function: map_espn_kalshi_game()


# Game IDs are constructed from ESPN data using the format
# {SPORT_CODE}-{DATE}{AWAY_TEAM}{HOME_TEAM}
# (e.g., KXNFLGAME-25OCT13NYGBUF).
# The merge function uses this exact game_id as the Kalshi event ID
# in a direct API call to:
#     https://api.elections.kalshi.com/trade-api/v2/events/{game_id}
# A match is confirmed if Kalshi returns a successful response;
# otherwise it's marked as a mismatch.
# The majority of times, mismatches will be caused by team abbreviations
# being different between ESPN and Kalshi. The best way to handle this is to
# create a dictionary of an ESPN full team name to how Kalshi abrivates the team name.
#Use the code in get_espn_team_info for help with that.

#every kalshi game should be able to be mapped to an espn game. not the other way around.

#running this script should create a csv file in format of kalshi_game_id, espn_game_id

