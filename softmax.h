#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>



//function to calculate softmax
float* softmax(float* input, int dim){
    float sum_exp = 0.0;


    for(int i=0; i<dim; i++){
        input[i] = exp(input[i]);
        sum_exp += input[i];    
    }

    float* output = (float*)malloc(dim*sizeof(float));
    for(int i=0; i<dim; i++){
        output[i] = input[i]/sum_exp;
    }

    free(input);


    return output;
   
}

