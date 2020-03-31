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

#include <iostream>
#include "Weights.h"
#include "Layer.h"


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

#ifdef DEBUG
        dumpBufferf(layer3Out, LAYER_3_N_NEURONS);
#endif
    }

    return;
}

int main() {

    run_predictions();
    return 0;
}
