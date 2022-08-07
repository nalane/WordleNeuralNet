from Lib import *

from multiprocessing import Pool
import pickle
import random
import string
import sys

# Ensure we have the right number of arguments
if len(sys.argv) != 2:
    sys.exit(f"Syntax: python {sys.argv[0]} <split number>")
split = int(sys.argv[1])

# Load words and shuffle them
words = []
with open("words.txt", "r") as file:
    for line in file:
        words.append(line.strip())
random.shuffle(words)

# Simulate a Wordle game
class WordleGame():
    # Set the knowledge we have for the game so far
    def __init__(self, goal):
        self.valid_guesses = set(words)
        self.wrong_letters = set()
        self.known_letters = {}
        self.not_index = {}
        self.maybe_letters = {}
        self.goal = goal

    # Determine what guesses can be valid with the given information
    def get_valid_guesses(self):
        # Determine which guesses are still valid
        new_valids = set()
        for word in self.valid_guesses:
            valid = True 
            for i, c in enumerate(word):
                if c in self.wrong_letters:
                    valid = False
                    break
                elif i in self.not_index and c in self.not_index[i]:
                    valid = False
                    break
                elif i in self.known_letters and self.known_letters[i] != c:
                    valid = False
                    break
                    
            if valid:
                new_valids.add(word)
        self.valid_guesses = new_valids

        # Ensure words are in a random order
        new_valids = list(new_valids)
        random.shuffle(new_valids)
        return new_valids

    # Get new information from a guess
    def process_guess(self, guess):
        for i, c in enumerate(guess):
            char_number = string.ascii_lowercase.index(c)
            if c == self.goal[i]:
                self.known_letters[i] = c
            elif c in self.goal:
                if i not in self.not_index:
                    self.not_index[i] = set()
                self.not_index[i].add(c)
                
                for j in range(len(guess)):
                    if j != i:
                        if j not in self.maybe_letters:
                            self.maybe_letters[j] = set()
                        self.maybe_letters[j].add(c)
            else:
                self.wrong_letters.add(c)

    # Generate an information vector representing the information we have
    def gen_vector(self):
        individual_res = [[UNKNOWN] * 26 for _ in range(5)]

        # Encode letters that aren't in the word
        for c in self.wrong_letters:
            char_number = string.ascii_lowercase.index(c)
            for v in individual_res:
                v[char_number] = INCORRECT

        # Encode letters that might be correct
        for i, cs in self.maybe_letters.items():
            for c in cs:
                char_number = string.ascii_lowercase.index(c)
                individual_res[i][char_number] = MAYBE
                
        # Encode information about knowing a letter in a word
        for i, c in self.known_letters.items():
            char_number = string.ascii_lowercase.index(c)
            individual_res[i] = [INCORRECT] * 26
            individual_res[i][char_number] = CORRECT

        # Set ones that we know are incorrect
        for i, cs in self.not_index.items():
            for c in cs:
                char_number = string.ascii_lowercase.index(c)
                individual_res[i][char_number] = INCORRECT

        # Build a full, flat vector
        result = []
        for v in individual_res:
            result += v
            
        return result

# Run a number of games where the given word is the correct word
def run_games(word):
    dataset = set()
    for _ in range(100):
        vecs = []
        game = WordleGame(word)
        guesses = game.get_valid_guesses()
        while len(guesses) > 1:
            # Get a random valid guess and process it
            guess = guesses[0]
            game.process_guess(guess)
            guesses = game.get_valid_guesses()

            # Store the knowledge vec
            knowledge_vec = game.gen_vector()
            vecs += [knowledge_vec]
         
        # Get the final result vector
        guess = guesses[0]
        game.process_guess(guess)
        final_vec = game.gen_vector()
        final_vec = [n // 3 for n in final_vec]

        # Add the data to the dataset
        for v in vecs:
            dataset.add(Pair(v, final_vec))
            
    return dataset

# Get the full dataset
with Pool(1) as p:
    datasets = p.map(run_games, words)
dataset = set()
for d in datasets:
    dataset.update(d)

# Shuffle the dataset and split into a training set and a testing set
dataset = list(dataset)
random.shuffle(dataset)
testing_set = [d for (i, d) in enumerate(dataset) if i % split == 0]
validation_set = [d for (i, d) in enumerate(dataset) if i % split == 1]
training_set = [d for (i, d) in enumerate(dataset) if i % split > 1]

# Write the sets as pickle files
with open("datasets/train.pkl", "wb") as file:
    pickle.dump(training_set, file)
with open("datasets/validation.pkl", "wb") as file:
    pickle.dump(training_set, file)
with open("datasets/test.pkl", "wb") as file:
    pickle.dump(testing_set, file)
