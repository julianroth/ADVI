import pandas as pd
import tensorflow as tf

def load_data():
    df = pd.read_csv("data/election88/polls.csv")
    # remove voters without preference
    df = df.dropna()
    # state, edu, age, female, black
    x = df.to_numpy(dtype=int)[:, 5:10]
    x[:, :3] = x[:, :3] - 1
    # vote for bush or not
    y = df['bush'].to_numpy()
    return x, y

# returns prev vote for each state
def load_prev_vote():
    df = pd.read_csv("data/election88/presvote.csv")
    prev_vote = df['g76_84pr'].to_numpy()
    return prev_vote
