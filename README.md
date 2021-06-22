# spiking q networks

An exploration of the use of spiking neural networks in reinforcement learning.
An overview of SNNs is available [here](https://www.notion.so/camilonotes/Spiking-neural-networks-in-reinforcement-learning-b6824ef8ce394d749ef5193c4503c3fd).

## installation
If you're on Big Sur, you'll need to be running Python3.9.

You'll also need these dependencies:
```
brew install ffmpeg
brew install llvm
brew install boost
brew install hdf5
```

And you'll need to update your .bashrc/.zshrc:
```
# Add this to your .bashrc/.zshrc:
export PATH="/usr/local/opt/llvm/bin:$PATH"

export CC="/usr/local/opt/llvm/bin/clang"
export CXX="/usr/local/opt/llvm/bin/clang++"
export CXX11="/usr/local/opt/llvm/bin/clang++"
export CXX14="/usr/local/opt/llvm/bin/clang++"
export CXX17="/usr/local/opt/llvm/bin/clang++"
export CXX1X="/usr/local/opt/llvm/bin/clang++"

export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
```

## breakout

In this directory is the SQN policy net solution to BreakoutDeterministic-v4 that I'm working on (`sqn.py`). 

You'll also find a baseline DQN solution to the env (`dqn.py`).

There's also the BindsNet example code to solve BreakoutDeterministic-v4 in `breakout.py`.

