import polars as pl
import sportsdataverse as sdv
from sportsdataverse.nba import most_recent_nba_season

pl.Config.set_tbl_rows(8)
SEASON = most_recent_nba_season()
print('current NBA season:', SEASON)

def safe(label, thunk):
    try:
        out = thunk()
        n = out.height if isinstance(out, pl.DataFrame) else (len(out) if hasattr(out, '__len__') else '?')
        print(f'✅ {label} — {n} rows')
        return out
    except Exception as e:  # noqa: BLE001 -- demo resilience
        print(f'⏭️  {label}: unavailable right now ({type(e).__name__})')
        return None



teams = safe('teams', sdv.nba.espn_nba_teams)
(teams.select(['team_id', 'team_location', 'team_name', 'team_abbreviation', 'team_color']).head(10) 
if teams is not None else 'teams feed unavailable')

tid = 9
if teams is not None and teams.height:
    # Boston Celtics if present, else the first team
    row = teams.filter(pl.col('team_abbreviation') == 'BOS')
    tid = int((row if row.height else teams)['team_id'][0])

roster = safe(f'roster {tid}', lambda: sdv.nba.espn_nba_team_roster(team_id=tid)) if tid else None
cols = ['full_name', 'jersey', 'position_abbreviation', 'height', 'weight', 'age']
(roster.select([c for c in cols if c in roster.columns]).head(10)
 if roster is not None and roster.height else 'roster unavailable')
print(roster)