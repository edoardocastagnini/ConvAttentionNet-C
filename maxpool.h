#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

//function for the maxpool operation (kernel size 2x2, stride 2)
float*** maxPoolforward(float*** input, int input_h, int input_w, int num_filters ){ 

    int output_h = input_h/2;
    int output_w = input_w/2;

    float*** output = (float***)malloc(num_filters * sizeof(float**));
    for (int i = 0; i < num_filters; i++) {
        output[i] = (float**)malloc(output_h * sizeof(float*));
        for (int j = 0; j < output_h; j++) {
            output[i][j] = (float*)malloc(output_w * sizeof(float));
        }
    }

  
    for(int k=0;k<num_filters;k++){
        for (int i = 0; i < input_h-1; i=i+2) {
            for (int j = 0; j < input_w-1; j=j+2) {
                float max = 0;
                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        float temp_val = input[k][i+m][j+n];
                        if (temp_val > max){
                            max = temp_val;
                        }
                    }
                }
                output[k][i/2][j/2] = max;
            }
        }
    }
    return output;
} 


