#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


//function to apply reLU activation function
float*** reLU(float*** input, int h, int w, int input_channels) {
    for (int i = 0; i < input_channels; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                if (input[i][j][k] < 0) {
                    input[i][j][k] = 0;
                }
            }
        }
    }

    return input;
}

