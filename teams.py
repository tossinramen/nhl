"""Team identity across sources. Everything is keyed internally by NHL API abbreviation."""
import unicodedata

ABBR_TO_NST = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken",
    "STL": "St Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}


NICKNAMES = {
    "ANA": "Ducks", "BOS": "Bruins", "BUF": "Sabres", "CGY": "Flames", "CAR": "Hurricanes",
    "CHI": "Blackhawks", "COL": "Avalanche", "CBJ": "Blue Jackets", "DAL": "Stars", "DET": "Red Wings",
    "EDM": "Oilers", "FLA": "Panthers", "LAK": "Kings", "MIN": "Wild", "MTL": "Canadiens",
    "NSH": "Predators", "NJD": "Devils", "NYI": "Islanders", "NYR": "Rangers", "OTT": "Senators",
    "PHI": "Flyers", "PIT": "Penguins", "SJS": "Sharks", "SEA": "Kraken", "STL": "Blues",
    "TBL": "Lightning", "TOR": "Maple Leafs", "UTA": "Mammoth", "VAN": "Canucks", "VGK": "Golden Knights",
    "WSH": "Capitals", "WPG": "Jets", "ARI": "Coyotes",
}


FRANCHISE = {"ARI": "UTA"}

_ALIASES = {
    "arizona coyotes": "ARI", "phoenix coyotes": "ARI",
    "utah hockey club": "UTA", "utah mammoth": "UTA", "utah": "UTA",
    "st. louis blues": "STL", "saint louis blues": "STL",
    "montreal canadiens": "MTL",
}


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").split())


_NAME_TO_ABBR = {_norm(v): k for k, v in ABBR_TO_NST.items()}
_NAME_TO_ABBR.update({_norm(k): v for k, v in _ALIASES.items()})


def name_to_abbr(name):
    """Full team name from any source (NST, Action Network, ...) -> NHL abbreviation (None if unknown)."""
    return _NAME_TO_ABBR.get(_norm(name))


def franchise(abbr):
    return FRANCHISE.get(abbr, abbr)


def nickname_to_abbr(nick):
    n = _norm(nick)
    for abbr, nk in NICKNAMES.items():
        if _norm(nk) == n:
            return abbr
    return _NAME_TO_ABBR.get(n)
