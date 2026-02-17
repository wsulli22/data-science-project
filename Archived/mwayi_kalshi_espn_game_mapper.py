import requests
from bs4 import BeautifulSoup
import re
import json
from teamConversionDict import espnAbbrToKalshiAbbr

espnURL = "https://www.espn.com/mens-college-basketball/game/_/gameId/"
kalshiURL = "https://kalshi.com/markets/kxncaambgame/mens-college-basketball-mens-game/KXNCAAMBGAME-"
kalshiAPIURL = "https://api.elections.kalshi.com/trade-api/v2/events/"

espnGames = open("list_of_espn_games.txt")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

gameIdsESPN = []
gameIdsKalshi = []
awayTeams = []
homeTeams = []
dates = []
fullKalshiURLs = []
kalshiIDtoESPNId = {}

def map_espn_kalshi_games(limit=10):
    j = 0

    for i in espnGames:
        gameId = i.strip().split(",")[0]
        awayTeam = i.strip().split(",")[2]
        homeTeam = i.strip().split(",")[3]

        if awayTeam == homeTeam:
            awayTeam = i.strip().split(",")[1]

        fullESPNURL = espnURL + gameId
        espnHTML = BeautifulSoup(requests.get(fullESPNURL, headers=headers).content, "html.parser")
        #espnHTMLs.append(BeautifulSoup(requests.get(fullESPNURL, headers=headers).content, "html.parser"))

        d = espnHTML.select_one("title").contents[0]
        a = re.search("\\(.+\\)", d).group()[1:-1].split(" ")
        a[0] = a[0].upper()
        a[1] = a[1][:-1].zfill(2)
        a[2] = a[2][2:]
        a = a[-1:] + a[:-1]
        a = "".join(a)
        fullKalshiURLs.append(kalshiURL + a + awayTeam + homeTeam)

        gameIdsESPN.append(gameId)
        awayTeams.append(awayTeam)
        homeTeams.append(homeTeam)
        dates.append(a)
        #Both ways because sometimes it might not work for one way

        kalshiAwayTeam = awayTeam if awayTeam not in espnAbbrToKalshiAbbr else espnAbbrToKalshiAbbr[awayTeam]
        kalshiHomeTeam = homeTeam if homeTeam not in espnAbbrToKalshiAbbr else espnAbbrToKalshiAbbr[homeTeam]

        gameIdsKalshi.append(["KXNCAAMBGAME-" + a + kalshiAwayTeam + kalshiHomeTeam, "KXNCAAMBGAME-" + a + kalshiHomeTeam + kalshiAwayTeam])

        j += 1

        if j >= limit:
            break

    #print(fullKalshiURLs)
    print(gameIdsKalshi)

    #arr = [['KXNCAAMBGAME-25NOV03LEHHOU', 'KXNCAAMBGAME-25NOV03HOULEH'], ['KXNCAAMBGAME-25NOV03FLAARIZ', 'KXNCAAMBGAME-25NOV03ARIZFLA'], ['KXNCAAMBGAME-25NOV03NHCCONN', 'KXNCAAMBGAME-25NOV03CONNNHC'], ['KXNCAAMBGAME-25NOV03QUINSJU', 'KXNCAAMBGAME-25NOV03SJUQUIN'], ['KXNCAAMBGAME-25NOV03OAKMICH', 'KXNCAAMBGAME-25NOV03MICHOAK'], ['KXNCAAMBGAME-25NOV03VILLBYU', 'KXNCAAMBGAME-25NOV03BYUVILL'], ['KXNCAAMBGAME-25NOV03SCSTLOU', 'KXNCAAMBGAME-25NOV03LOUSCST'], ['KXNCAAMBGAME-25NOV03EWUUCLA', 'KXNCAAMBGAME-25NOV03UCLAEWU'], ['KXNCAAMBGAME-25NOV03SOUARK', 'KXNCAAMBGAME-25NOV03ARKSOU'], ['KXNCAAMBGAME-25NOV03UNDALA', 'KXNCAAMBGAME-25NOV03ALAUND']]
    k = 0

    for i in gameIdsKalshi:
        for j in i:
            kalshiJSON = requests.get(kalshiAPIURL + j, headers=headers).json()

            if "error" not in kalshiJSON:
                #gameIdsKalshi[k] = j
                gameIdsKalshi[k] = j
                kalshiIDtoESPNId[j] = gameIdsESPN[k]
                break

        k += 1

    print(kalshiIDtoESPNId)
    return kalshiIDtoESPNId

if __name__ == "__main__":
    map_espn_kalshi_games()

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
