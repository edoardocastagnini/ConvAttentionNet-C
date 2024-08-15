#include <stdio.h>
#include <stdlib.h>

typedef struct Linear {
    float **weights;
    float *biases;
    int in_features;
    int out_features;
} Linear;

//function to create a linear layer and initialize it with weights
Linear* createLinear(int in, int out, float* weights) {
    Linear *linear = (Linear*)malloc(sizeof(Linear));
    linear->in_features = in;
    linear->out_features = out;

    linear->weights = (float**)malloc(in * sizeof(float*));
    for (int i = 0; i < in; i++) {
        linear->weights[i] = (float*)malloc(out * sizeof(float));
        for (int j = 0; j < out; j++) {
            linear->weights[i][j] = weights[j*in + i];
        }
    }  
    free(weights);
    linear->biases = (float*)malloc(out * sizeof(float));
    for (int i = 0; i < out; i++) {
        linear->biases[i] = 0;
    }
    return linear;
}

void freeLinear(Linear *linear) {
    for (int i = 0; i < linear->in_features; i++) {
        free(linear->weights[i]);
    }
    free(linear->weights);
    free(linear->biases);
    free(linear);
}

//function to perform the forward pass
float* linearForward(int in, int out, float*** input, int input_h, int input_w, int num_filters, float* weights){
    Linear *linear = createLinear(in, out, weights);
    float* totals = (float*)malloc(linear->out_features * sizeof(float));

    float* flat_input = (float*)malloc(input_h * input_w * num_filters * sizeof(float));
    for (int i = 0; i < input_h * input_w * num_filters; i++) {
        flat_input[i] = input[i/(input_h * input_w)][(i%(input_h * input_w))/input_w][(i%(input_h * input_w))%input_w];
    }
    
    for (int j = 0; j < linear->out_features; j++) {
        totals[j] = 0.0;
        for (int i = 0; i < linear->in_features; i++) {
            totals[j] += flat_input[i] * linear->weights[i][j];
        }
        totals[j] += linear->biases[j];
    }
    freeLinear(linear);
    free(flat_input);
    free3dMatrix(input, num_filters, input_h);

    return totals;
}




//function to perform the forward pass
float* linearForwardflat(int in, int out, float* input, float* weights){
    Linear *linear = createLinear(in, out, weights);
    float* totals = (float*)malloc(linear->out_features * sizeof(float));
    
    for (int j = 0; j < linear->out_features; j++) {
        totals[j] = 0.0;
        for (int i = 0; i < linear->in_features; i++) {
            totals[j] += input[i] * linear->weights[i][j];
        }
        totals[j] += linear->biases[j];
    }
    freeLinear(linear);
    free(input);

    return totals;
}