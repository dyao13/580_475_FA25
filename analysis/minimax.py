import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
import torch
import torch.nn as nn

# neural network class
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    
    def forward(self, x):
        return self.net(x)

# node class to represent current draft state
class Node:
    def __init__(self, mode=None, bans1=[], bans2=[], team1=[], team2=[]):
        self.mode = mode

        self.bans1 = bans1
        self.bans2 = bans2

        self.team1 = team1
        self.team2 = team2

# minimax algorithm class
class Minimax:
    # initialize with model choice
    def __init__(self, model='rf'):
        self.model_path = os.path.join(os.path.dirname(__file__), "../models")

        self.brawlers = np.loadtxt(os.path.join(self.model_path, "names.csv"), delimiter=",", dtype='str')
        self.encoder = load(os.path.join(self.model_path, "encoder.joblib"))
        self.model_name = model
        self.model = self.get_model(self.model_name)
    
    # get model from saved parameters
    def get_model(self, model):
        if model == 'rf':
            model = load(os.path.join(self.model_path, "rf.joblib"))
        elif model == 'logreg':
            model = load(os.path.join(self.model_path, "lr.joblib"))
        elif model == 'nn':
            model = torch.load(os.path.join(self.model_path, "nn.pth"), weights_only=False)
            model.eval()
        else:
            raise ValueError("Choose model from 'rf' (random forest), 'lr' (logistic regression), or 'nn' (neural network).")
        
        return model
    
    # return prediction of victory probability
    def evaluation(self, node):
        team1 = node.team1
        team2 = node.team2

        team1.sort()
        team2.sort()
        
        val = 0

        state = [node.mode] + team1 + team2

        X = pd.DataFrame([state], columns=['mode', 'b1', 'b2', 'b3', 'r1', 'r2', 'r3'])
        X = self.encoder.transform(X)

        if self.model_name == 'rf' or self.model_name == 'logreg':
            val += self.model.predict_proba(X)[0][1]
        else:
            X = torch.tensor(X, dtype=torch.float32)
            
            with torch.no_grad():
                val += self.model(X).item()

        state = [node.mode] + team2 + team1

        X = pd.DataFrame([state], columns=['mode', 'b1', 'b2', 'b3', 'r1', 'r2', 'r3'])
        X = self.encoder.transform(X)

        if self.model_name == 'rf' or self.model_name == 'logreg':
            val += 1 - self.model.predict_proba(X)[0][1]
        else:
            X = torch.tensor(X, dtype=torch.float32)
            
            with torch.no_grad():
                val += 1 - self.model(X).item()

        return val/2

    # minimax algorithm
    def minimax(self, node, depth, alpha, beta, max_depth):
        if depth == 0 or (len(node.team1) == 3 and len(node.team2) == 3):
            return [], self.evaluation(node)
        
        num1 = len(node.team1)
        num2 = len(node.team2)

        # determine who's pick it is
        if (num1 == 0 and num2 == 0) or (num1 == 1 and num2 == 2) or (num1 == 2 and num2 == 2):
            isMaximizer = True
        else:
            isMaximizer = False

        best_pick = None
        main_line = []

        # maximize victory probability
        if isMaximizer:
            value = -float('inf')

            for brawler in (tqdm(self.brawlers) if depth == max_depth else self.brawlers):
                if brawler not in node.bans1 and brawler not in node.bans2 and brawler not in node.team1 and brawler not in node.team2:
                    node.team1.append(brawler)
                    next_line, new_value = self.minimax(node, depth-1, alpha, beta, max_depth)
                    node.team1.remove(brawler)

                    if new_value > value:
                        value = new_value
                        best_pick = brawler
                        main_line = [best_pick] + next_line
                        
                    alpha = max(alpha, value)

                    if value >= beta:
                        break 

        # minimize victory probability
        else:
            value = float('inf')

            for brawler in (tqdm(self.brawlers) if depth == max_depth else self.brawlers):
                if brawler not in node.bans1 and brawler not in node.bans2 and brawler not in node.team1 and brawler not in node.team2:
                    node.team2.append(brawler)
                    next_line, new_value = self.minimax(node, depth-1, alpha, beta, max_depth)
                    node.team2.remove(brawler)

                    if new_value < value:
                        value = new_value
                        best_pick = brawler
                        main_line = [best_pick] + next_line

                    beta = min(beta, value)

                    if value <= alpha:
                        break

        return main_line, value
    
    # return draft order and victory probability
    def get_main_line(self, node, depth):
        num1 = len(node.team1)
        num2 = len(node.team2)

        if num1 > 3 or num2 > 3 or abs(num1 - num2) > 2:
            return -float('inf')

        line = []

        if num1 > 0:
            line.append(node.team1[0])
        if num2 > 0:
            line.append(node.team2[0])
        if num2 > 1:
            line.append(node.team2[1])
        if num1 > 1:
            line.append(node.team1[1])
        if num1 > 2:
            line.append(node.team1[2])
        if num2 > 2:
            line.append(node.team2[2])

        main_line, value = self.minimax(node, depth, -float('inf'), float('inf'), depth)
            
        return line + main_line, value

# driver
def main():
    # engine = Minimax(model='rf')
    # node = Node(mode='knockout', team1=['BROCK'], team2=['GENE', 'MAX'])

    # main_line, value = engine.get_main_line(node, 6)

    # print(main_line)
    # print(value)

    engine = Minimax(model='rf')
    
    mode = input("Select a game mode: ")
    draft = input("Input a partial draft: ").split()
    
    team1 = []
    team2 = []

    for i in range(len(draft)):
        if i == 0 or i == 3 or i == 4:
            team1.append(draft[i])
        elif i == 1 or i == 2 or i == 5:
            team2.append(draft[i])
        else:
            raise ValueError("Invalid draft.")
    
    node = Node(mode=mode, team1=team1, team2=team2)
    
    main_line, value = engine.get_main_line(node, 6)
    main_line = [str(x) for x in main_line]

    print(main_line)
    print(value)

if __name__ == '__main__':
    main()