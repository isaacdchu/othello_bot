# othello_bot
Machine learning for playing Othello

# Setup
```bash
git clone https://github.com/isaacdchu/othello_bot.git
```
```bash
cd othello_bot
```
```bash
uv run main.py
```

# Goals
- Implement better rollout policy (neural network)
    - Currently, rollouts are done with completely random moves
    - Using a neural network to more intelligently pick each move can lead to stronger play
- Persist game tree throughout moves
    - Right now, a new tree is created during each move, so past computations could be wasted
    - Keeping an instance variable of an MCTSNode in the Otto class to store the full tree data may be beneficial
    - Highly speculative