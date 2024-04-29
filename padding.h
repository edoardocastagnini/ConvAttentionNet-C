#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


//function to create the padding
float*** create_padding(float*** input, int h, int w, int channels, int padding){
    float*** output = (float***)malloc(channels*sizeof(float**));
    for(int k=0; k<channels; k++){
        output[k] = (float**)malloc((h + 2 * padding) * sizeof(float*));
        for (int i = 0; i < h + 2 * padding; i++) {
            output[k][i] = (float*)malloc((w + 2 * padding) * sizeof(float));
        }
    }

    for(int k=0; k<channels; k++){
        for (int i = 0; i < h + 2 * padding; i++) {
            for (int j = 0; j < w + 2 * padding; j++) {
                if (i < padding || i >= h + padding || j < padding || j >= w + padding) {
                    output[k][i][j] = 0.0;
                } else {
                    output[k][i][j] = input[k][i - padding][j - padding];
                }
            }
        }
    }

    return output;
}
