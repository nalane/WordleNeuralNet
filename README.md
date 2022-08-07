# WordleNeuralNet
This project uses neural networks to make guesses in Wordle, given what knowledge we have about the final word. There are three main scripts:
- GenerateDatasets.py: Runs many wordle games to generate datasets. Takes an integer as an argument that will split the final dataset into that many pieces; the test set will be one of those pieces, the training set will be the remainder
- Train.py: Trains the neural network and produces model files
- Run.py: When modified with what is currently known about the final word, produces a guess about what the final word may be

## Neural Network Structure
The input is a vector with 130 elements. Conceptually, it can be split into 5 sets of 26. Each of the 5 sets corresponds to a position in the final word. Each item in the set represents, for all of the 26 letters of the alphabet, whether we know that letter belongs in that slot, whether we know a letter is *not* in that slot, whether we think the letter may be in that slot (i.e., a letter appeared yellow in another position), or we have no information as to whether or not the letter is in that position.

The input is put into an embeddings layer. Conceptually, this is meant to convert the knowledge level, a discrete value, into a series of continuous values. The network then bifurcates. Along one path, a convolutional network is applied to all 5 sets of 26 letters. Conceptually, this output is meant to represent knowledge about what each position may be. The other path applies a convolutional network to each letter for all 5 positions. This is meant to extract knowledge based on a letter's position. For instance, if it is known that the letter 'e' is the final letter, then we can suppose that the previous letter is a consonant. Finally, these vectors are flattened and sent to a linear network, producing a final vector that represents, for each position, what is the likelihood of each of the 26 letters being in that posiiton.

## Training
When we have more knowledge about what letters are in a word (e.g., when all 5 letters are known), then we want to reinforce the learning applied at those knowledge levels. Conversely, when less knowledge is known, then we don't care as much if guesses are inaccurate. To accomplish this, transfer learning is applied. First, the model is trained with all levels of knowledge as input. Then, we use transfer learning to take that model and retrain it on datasets where at least 1 letter is known. The knowledge level is increased until we hit the point where the dataset contains only the instances where all 5 letters are known.

This means that the model sees the data with higher knowledge levels more frequently. Thus it is punished more for errors in these predictions than in ones where less is known about the final word.
