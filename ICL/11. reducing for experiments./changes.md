
Final changes going to be made:

- reduced the number of states to 4 for both the noise tokens and the ruled tokens. 
- setting the mlp freezed on log readout and set other to train
- parameters to train are Wqk and Wov for the first and second layers, unembedding matrix and it's bias.
- removing noise thresholds < 0.5 will do analysis for [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lesser noise is very low to make signal visible at all. 
- reducing the context length to 100 and reduce loss on last 50 tokens(i think appropirate to capture enough signal)
- setting the markov uniform distribution in the range [0.1, 0.9]. as the possibilites for the next two possible ruled token depending on the last ruled token. 
- optimizer we will set to vanila sgd, adam both options to train.
- use 10,000 data points to train .