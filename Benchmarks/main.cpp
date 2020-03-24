#include <iostream>
#include "Weights.h"
#include "Layer.h"

#define DEBUG
#ifdef DEBUG
#include <assert.h>
#endif

using namespace std;

#ifdef DEBUG
void dumpBufferf(float * buffer, int lenBuffer ){
    cout << "{ ";
    for (int i = 0; i < lenBuffer; i++){
         cout << buffer[i];

         if (i == lenBuffer - 1){
             cout << " }";
         } else {
             cout << " , ";
         }
    }

    cout << endl;
}

#endif

/*
void ReLUFeedForward(float* inputs, int nInputs, float* weights, int nNeurons, int nWeights, float* outputs, int nOutputs){

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
}
*/

void run_predictions(){
    float layer1Out[LAYER_1_N_NEURONS] = {0};
    float layer2Out[LAYER_2_N_NEURONS] = {0};
    float layer3Out[LAYER_3_N_NEURONS] = {0};

    Layer layer1(LAYER_1_WEIGHTS, layer1Out, LAYER_1_N_NEURONS, LAYER_1_N_WEIGHTS, LAYER_1_N_NEURONS, ReLU);
    Layer layer2(LAYER_2_WEIGHTS, layer2Out, LAYER_2_N_NEURONS, LAYER_2_N_WEIGHTS, LAYER_2_N_NEURONS, ReLU);
    Layer layer3(LAYER_3_WEIGHTS, layer3Out, LAYER_3_N_NEURONS, LAYER_3_N_WEIGHTS, LAYER_3_N_NEURONS, SOFTMAX);

    for (int i = 0; i < TEST_N_INPUTS; i++){
        float *inputs = &TEST_INPUTS[i * TEST_N_FEATURES];

        layer3.feedForward(layer2.feedForward(layer1.feedForward(inputs, TEST_N_FEATURES), LAYER_1_N_NEURONS), LAYER_2_N_NEURONS);

        dumpBufferf(layer3Out, LAYER_3_N_NEURONS);

    }

    return;
}

int main() {

    run_predictions();
    return 0;
}