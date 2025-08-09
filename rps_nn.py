# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import random

HISTORY_LEN = 5   # how many past moves the NN sees (so the way it works is by guessing what move to do based on your past moves)
EPOCHS = 50       # Training epochs per update
BLUFF_PROB = 0.15 # chance to bluff instead of best counter (bluff means random move so i can't figure the pattern out easily and beat it)

move_to_idx = {"R": 0, "P": 1, "S": 2}
idx_to_move = {0: "R", 1: "P", 2: "S"}

#one-hot encoding means eg:
#if we rep em as integers it will for eg dog = 2 and cat = 1 then the nn will think the dog is "bigger" than the cat
#one-hot encoding makes it like: blue: [1, 0, 0], green: [0, 1, 0]... so the model classify's them correctly/so our data gets represented to our model correctly
def encode_moves(moves):
  vec = []
  for m in moves:
    one_hot = [0, 0, 0]
    one_hot[move_to_idx[m]] = 1 #we set it to "1" bc now as mentioned we're classifying the "Paper" module for out model (eg if m="p" move to idx becomes 1 and then onehot[1] = 1 is
    #[0, 1, 0]
    vec.extend(one_hot) #we're just adding one_hot to vec bc one_hot is going to change but we need a stable var that keeps the org variables (this one adds them it doesn't replace)
  return vec

#ai architecture
class RPSNetwork(nn.Module):
  def __init__(self):
    super().__init__() #initilize
    self.fc1 = nn.Linear(HISTORY_LEN * 3, 32) #history_len * 3 bc we're flattening it bc the model expects a single list, not 5 lists (& also 32 neurons for input)
    self.fc2 = nn.Linear(32, 16) #input 32, neurons 16 (hidden layer)
    self.fc3 = nn.Linear(16, 3) #input 16, 3 neurons (for R, P, S)
  def forward(self, x): #forward propagation (non-linear activation)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    return self.fc3(x)
    #ReLu is basically if x>=0 x=x else 0
    #helps detect patterns like “if this move was played 3 steps ago and that move 1 step ago, predict…"

#main function
history = []
X_train, y_train = [], []
net = RPSNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.01) #lr is learning rate

print("Enter R, P, or S. Ctrl+C to quit.")
try:
    while True:
        player_move = input("Your move: ").upper().strip()
        if player_move not in move_to_idx:
            print("Invalid move.")
            continue

        # store history and training example
        if len(history) >= HISTORY_LEN:
            X_train.append(encode_moves(history[-HISTORY_LEN:]))
            y_train.append(move_to_idx[player_move])

            # Train NN on collected data
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.long)
            for _ in range(EPOCHS):
                optimizer.zero_grad() #reset gradients
                outputs = net(X_tensor)
                loss = loss_fn(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            # Predict your next move
            with torch.no_grad():
                inp = torch.tensor([encode_moves(history[-HISTORY_LEN:])], dtype=torch.float32)
                pred = net(inp)
                predicted_move = int(torch.argmax(pred).item())

            # Decide AI's move (counter or bluff)
            counters = {0: 1, 1: 2, 2: 0}  # R->P, P->S, S->R
            if random.random() < BLUFF_PROB:
                ai_move = random.choice([0, 1, 2])  # bluff
            else:
                ai_move = counters[predicted_move]

        else:
            ai_move = random.choice([0, 1, 2])  # Not enough history yet

        print(f"AI plays: {idx_to_move[ai_move]}")
        history.append(player_move)

except KeyboardInterrupt:
    print("\nGame Over.")

torch.save(net.state_dict(), "rpsnn.pth")
from google.colab import files
files.download("rpsnn.pth") #for downloading the model