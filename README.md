# deep q networks

An exploration of the use of spiking neural networks in reinforcement learning.

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

## cartpole

For this environment there are two different solutions.

The first is @kkweon's solution to the CartPole-v0 environment with a replay
buffer and epsilon-annealing (to stop exploring in the long run)
