/* mlp.c -- 
 * Author: Lim Chun Yu
 */

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "dataparser.h"
#include "forwardprop.h"
#include "backprop.h"
#include "error.h"
#include "mlp.h"

BiasWeights_t initBiasWeights(int inputFields) {
    BiasWeights_t biasWeights;
    biasWeights.weights = (double*)malloc(inputFields * sizeof(double));
    biasWeights.bias = 0;

    srand(time(NULL));
    for (int cols = 0; cols < inputFields; cols++) {
        *(biasWeights.weights + cols) = (double)rand() /
            (double)RAND_MAX * 2.0 - 1.0; /* Randomly assign weights and bias */
    }
    return biasWeights;
}

Layer_t genLayer(int nodes, int conn, InputOutput_t trainTest, Layer_t* next, Layer_t* prev) {
    Layer_t layer;
    layer.numOfNodes = nodes;
    layer.nodes = (Node_t*)malloc(nodes * sizeof(Node_t));
    layer.next = next;
    layer.prev = prev;
    for (int i = 0; i < nodes; i++) {
        (layer.nodes + i)->connections = conn;
        (layer.nodes + i)->biasWeights = initBiasWeights((layer.nodes + i)->connections);
        (layer.nodes + i)->muladd = (double*)malloc(TRAINING_MAX * sizeof(double));
        (layer.nodes + i)->activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    }


    return layer;
}



double** trainLayer(Layer_t layer, double** input){
    double** values = (double**)malloc(layer.numOfNodes*sizeof(double*));
    for (int i = 0; i < layer.numOfNodes; i++){
        forwardPropagation(input, (layer.nodes + i)->biasWeights,
                                            (layer.nodes + i)->muladd, (layer.nodes + i)->activatedVal,
                                            TRAINING_MAX, (layer.nodes + i)->connections);
        *(values+i) = (layer.nodes + i)->activatedVal;
    }
    return values;

}

Node_t* trainNetwork(int layernum, int nodes, InputOutput_t trainTest,int minMae ,FILE* graph,int val) {
    int conn=nodes;
    double** values = (double**)malloc(nodes*sizeof(double*));
    double MAE_VAL;
    Layer_t* layers = (Layer_t*)malloc(layernum * sizeof(Layer_t));

    for (int i = 0; i < layernum + 1; i++){
        conn = i == 0 ? ATTR_COLUMNS : nodes;
        conn = i == layernum - 1 ? 1 : nodes;
        Layer_t* next = i == layernum - 1 ? NULL : (layers + i + 1);
        Layer_t* prev = i == 0 ? NULL : (layers + i - 1);
        *(layers+i) = genLayer(nodes, conn, trainTest, next, prev);
    }
    int t=0;
    do {
        if(t++ == 0) {
            printf("-Before Training-\nMMSE Training: %f\n",
                    minMeanSquareError(trainTest.output, (layers + layernum - 1)->nodes->activatedVal,
                                         TRAINING_MAX));
            printf("MMSE Testing: %f\n",
                    minMeanSquareError(trainTest.output, (layers + layernum - 1)->nodes->activatedVal,
                                         TESTING_MAX));
        }
        for (int i = 0; i<layernum; i++){
                if(i==0) values=trainLayer(*(layers + i), trainTest.input);
                else values=trainLayer(*(layers + i), values);
                (layers + i)->value = values;
        }


        fprintf(graph, "%i %lf \n", t, MAE_VAL);
        MAE_VAL = meanAbsoluteValue(trainTest.output, (layers + layernum)->nodes->activatedVal,
                                    TRAINING_MAX);
        if (MAE_VAL > minMae) {
            for (int i = 0; i < layernum; i++){
                Layer_t* curr=(layers + layernum - 1 - i);
                for (int j = 0; j < curr->numOfNodes; i++){
                    double** av = 0 == (curr->numOfNodes - 1) ? trainTest.input : (curr->prev)->value;
                    (curr->nodes + j)->biasWeights = backwardsPropagation(av, trainTest.output, // constants
                                                                        (curr->nodes+j)->biasWeights,((layers + layernum)->nodes)->activatedVal,
                                                                        (curr->nodes+j)->muladd, TRAINING_MAX, (curr->nodes+j)->connections);
                }
            
            }
        }

    } while (MAE_VAL > minMae);
    return (layers + layernum - 1)->nodes;
}