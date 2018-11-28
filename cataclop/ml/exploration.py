import numpy as np

def loss_streak(df):
    streak = 0
    streak_max = 0
    for i in range(len(df)):
        rr = df.iloc[i]
        if rr['profit'] <= 0:
            streak = streak+1
            if streak > streak_max:
                streak_max = streak
        else:
            streak = 0
    return streak_max

def random_race(df, cols=None, n=1):

    if cols is None:
        cols = ['position', 'sub_category', 'num', 'final_odds', 'final_odds_ref']

    return df.reset_index(drop=True).set_index(['race_id', df.index]).loc[np.random.permutation(df['race_id'].unique())[0:n]][cols]
    