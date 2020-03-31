/*
MIT License

Copyright (c) 2020 Kurt Lee Meister Jr.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#ifndef BENCHMARKS_LAYER_H
#define BENCHMARKS_LAYER_H
#include <iostream>

#include "Weights.h"
#include <math.h>

#ifdef DEBUG
#include <assert.h>
#endif

enum ACTIVATION {
    ReLU = 10,
    SOFTMAX = 20
};

class Layer {
private:
    int nWeights;
    int nNeurons;
    int nOutputs;
    float *outputs;
    float *weights;
    ACTIVATION activation;

protected:
    virtual float* relu(float * inputs, int nInputs);
    virtual float* softmax(float *inputs, int nInputs);

public:

    Layer(float *weights, float *outputs, int nNeurons, int nWeights, int nOutputs, ACTIVATION activation);
    virtual float* feedForward(float *inputs, int nInputs);

};

inline Layer::Layer(float *weights, float *outputs, int nNeurons, int nWeights, int nOutputs, ACTIVATION activation) {
    this->nWeights = nWeights;
    this->nNeurons = nNeurons;
    this->nOutputs = nOutputs;
    this->weights = weights;
    this->outputs = outputs;
    this->activation = activation;
}

inline float *Layer::feedForward(float *inputs, int nInputs) {
    if (activation == ReLU){
        return relu(inputs, nInputs);
    } else if (activation == SOFTMAX) {
        return softmax(inputs, nInputs);
    }

    return NULL;
}

inline float *Layer::relu(float *inputs, int nInputs) {
    // for development only undef DEBUG  or remove when ready to compile for embedded sys
#ifdef DEBUG
    assert(nInputs == nWeights - 1);
    assert(nNeurons == nOutputs);
#endif

    for (int iOut = 0; iOut < nOutputs; iOut++){
        //initialize output with bias and weight
        int startIdx = iOut * nWeights;
        outputs[iOut] = 1 * weights[startIdx];


        for (int iWeight = startIdx + 1, iIn = 0 ; iWeight < startIdx + nWeights; iWeight++, iIn++){
            outputs[iOut] += (weights[iWeight] * inputs[iIn]);
        }

        //apply ReLU
        if (outputs[iOut] <= 0){
            outputs[iOut] = 0;
        }
    }

    return outputs;
}

inline float *Layer::softmax(float * inputs, int nInputs) {
    // for development only undef DEBUG  or remove when ready to compile for embedded sys
#ifdef DEBUG
    assert(nInputs == nWeights - 1);
    assert(nNeurons == nOutputs);
#endif

    for (int iOut = 0; iOut < nOutputs; iOut++){
        //initialize output with bias and weight
        int startIdx = iOut * nWeights;
        outputs[iOut] = 1 * weights[startIdx];


        for (int iWeight = startIdx + 1, iIn = 0 ; iWeight < startIdx + nWeights; iWeight++, iIn++){
            outputs[iOut] += (weights[iWeight] * inputs[iIn]);
        }
    }

    float sum = 0;
    for (int iOut = 0; iOut < nOutputs; iOut++){
        sum += expf(outputs[iOut]);
    }

    for (int iOut = 0; iOut < nOutputs; iOut++){
        outputs[iOut] = expf(outputs[iOut]) / sum;
    }


    return outputs;
}


#endif //BENCHMARKS_LAYER_H
