import pandas as pd
from cataclop.pmu.models import Bet

def get_bet_df():
    bets = Bet.objects.filter(simulation=False, player__isnull=False).order_by('-created_at')

    def transform(bet):
        win = float(bet.player.winner_dividend)/100.0 if bet.player.winner_dividend is not None else 0 
        profit = float(bet.amount) * win - float(bet.amount) 
        return [bet.program, bet.created_at.strftime('%Y-%m-%d'), profit]

    stats = [transform(b) for b in bets]
    stats = pd.DataFrame(data=stats, columns=['program', 'date', 'profit'])
    return stats

def get_bet_profit_stats():
    df = get_bet_df()
    return df.groupby('program')['profit'].describe()
