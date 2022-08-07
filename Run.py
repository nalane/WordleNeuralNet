import string
from Lib import *

# The trans 5 model had the best F1 score
model = NeuralNetwork()
model.load_state_dict(torch.load("models/model_trans5.pth"))
model.eval()

# Establish knowledge vectors
k_vecs = [[UNKNOWN] * 26 for _ in range(5)]

# Mark letters as incorrect
incorrect_letters = ""
for i in range(5):
    for c in incorrect_letters:
        c_idx = string.ascii_lowercase.index(c)
        k_vecs[i][c_idx] = INCORRECT

# Mark letters that we know are in the word, but not where
maybe_idxs = {}
# Example:
# idx = string.ascii_lowercase.index('i')
# maybe_idxs[idx] = [0]

for letter, poses in maybe_idxs.items():
    for i in range(0, 5):
        k_vecs[i][letter] = MAYBE
    for pos in poses:
        k_vecs[pos][letter] = INCORRECT
        
# Mark correct letters
correct_letters = {} # Example: {'e': 4}
for c, i in correct_letters.items():
    c_idx = string.ascii_lowercase.index(c)
    k_vecs[i][c_idx] = CORRECT
    
# Concat all vectors
k_vec = []
for v in k_vecs:
    k_vec += v
  
# Generate predictions
k_vec = torch.IntTensor(k_vec)
pred = model(k_vec).sigmoid()
pred = pred.reshape((5, 26))
print(pred)

# Identify the word that best matches the predicted probabilities
best = None
with open("words.txt", "r") as file:
    for word in file:
        word = word.strip()
        p = 1.0
        for i, c in enumerate(word):
            char_idx = string.ascii_lowercase.index(c)
            p *= pred[i][char_idx]
            
        if best == None or best[1] < p:
            best = (word, p)
print(best)
