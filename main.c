#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "network.h"
#include "input_image.h"


#define INPUT_H 32
#define INPUT_W 32
#define PADDING 1
#define INPUT_CHANNELS 3


int main(void)
{ 

    float*** input = (float***)malloc(INPUT_CHANNELS*sizeof(float**));
    for(int i=0; i<INPUT_CHANNELS; i++){
        input[i] = (float**)malloc(INPUT_H*sizeof(float*));
        for(int j=0; j<INPUT_H; j++){
            input[i][j] = (float*)malloc(INPUT_W*sizeof(float));
        }
    }





    float* my_input = init_input(INPUT_H*INPUT_W*INPUT_CHANNELS);
    //reshape input
    for(int i=0; i<INPUT_CHANNELS; i++){
        for(int j=0; j<INPUT_H; j++){
            for(int k=0; k<INPUT_W; k++){
                input[i][j][k] = my_input[i*INPUT_H*INPUT_W + j*INPUT_W + k];
            }
        }
    }
    free(my_input);

 

    float* output = network(input, INPUT_H, INPUT_W, INPUT_CHANNELS, PADDING);
    for (int i = 0; i < 10; i++){
        printf("%f ", output[i]);
    }

    
    free(output);


}
