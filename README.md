# othello_bot
Machine learning for playing Othello

# Setup
1. Clone the repository
```bash
git clone https://github.com/isaacdchu/othello_bot
cd othello_bot
```
2. Switch to this branch
```bash
git switch cpp
```
3. Build and run
```bash
make run
```

# C++ bot
- Ideally will run faster than the python one
- Board represented by unsigned 64-bit integers: each bit corresponds to a square
    - the least significant bit is the top left square, then the second bit is the square to the right of that

# Source of Training Data
- I used training data from Egaroucid, which can be found here
- https://www.egaroucid.nyanyan.dev/en/technology/train-data/

# Todo
- Full refactor of CNN class
    - Abstraction for layers (convolutional layers/dense layers)
    - Abstraction for model optimizers (Adam, SGD, etc)
    - Easier configuration (config file, saving, training, and evaluating)
    - Batch processing for training (epochs)
    - Data handling improvements