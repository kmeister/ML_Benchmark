# ML_Benchmark
The goal of this project is to create a CPU benchmark for machine learning applications.

# Overview
the package contains a script called GenerateModel.py can generate an MLP with 3 hidden layers, and a few configurable hyperparameters. It'll train the MLP on a synthetic dataset, and then generate a header file containing a set of C arrays representing the weights of the trained model, and a test dataset

The benchmark folder contains a c++ class Layer, which can use the generated header file to run predictions on the test data

compile without the "DEBUG" declaration to create a version with no outputs to stderr or stdout for embedded applications

# To run this benchmark in GEM5 using the RISCV ISA follow the following set of steps...

## 1. build GEM5 for RISCV
start by installing dependencies:  
```
sudo apt install build-essential
apt install m4 zlib1g-dev scons python-six python-dev
```
clone & Build GEM5. This will take a while...  
```
cd ~/
git clone https://gem5.googlesource.com/public/gem5
cd gem5
scons build/RISCV/gem5.opt
```
to test:   
```
build/RISCV/gem5.opt configs/example/se.py -c tests/test- progs/hello/bin/riscv/linux/hello
```
follow the remaining steps in hw1 to modify the se.py file

## 2. Install the RISCV Toolchain
follow the instructions [here] (https://github.com/riscv/riscv-gnu-toolchain)

## 3. Clone this Repository and Build the Benchmark
```
cd ~
git clone https://github.com/kmeister/ML_Benchmark
cd ML_Benchmark/benchmarks
riscv64-unknown-linux-gnu-gcc -static -Wall -O0 -I. -c main.cpp -o main.o 
riscv64-unknown-linux-gnu-g++ -static -Wall -L. -o mlbench main.o 
```
## 4. Try Running the Benchmark...
```
cd ~/gem5
build/RISCV/gem5.opt configs/example/se.py --cpu-type DerivO3CPU  -c  ../ML_Benchmark/Benchmarks/mlbench --caches --l1i_size=32kB --l1i_assoc=4 --l1d_size=32kB --l1d_assoc=4 --cacheline_size=64 --cpu-clock=1.6GHz --maxinsts=1000000
```

at this point you can experiment with any of the parameters we'e used for the HW assignment

# To Retrain the Neural Net, or modify it's parameters... 
To retrain the NN in the benchmark and generate a new Weights.h file... (requires python3, TensorFlow, and sklearn)
```
cd ML_Banechmark/scripts
 ./ModelGenerator.py ../Benchmarks/Weights.h
 ```
use the following to see what options exist for configuring your model
```
./ModelGenerator.py -h 
```
note that if you overwrite the Weights.h file, even with the same parameters, you're likely going to wind up with different weights due to an element of randomness in how tensorflow sets up the initial values. Also, the NN currently in Weights.h is not based on the default parameters in the ModelGenerator script. Those were chosen for speed in testing the script, these were chosen to stress the RISCV processor.

the current weights.h was generated with the following commancd:
```
./ModelGenerator.py ../Benchmarks/Weights.h --test_size=1000 --n_classes=26 --n_features=576 --layer1=1000 --layer2=1000 --n_epochs=35
```
