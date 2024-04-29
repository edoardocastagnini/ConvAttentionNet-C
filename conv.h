#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

//LAYER CONV
typedef struct {
    int out_channels;
    int in_channels;
    float**** filters;
    int n;
} Conv2d;


int min (int a, int b) {
    if (a < b) {
        return a;
    }
    return b;
}

void freeConv2d(Conv2d layer) {
    for(int i=0;i<layer.out_channels;i++){
        for(int j=0;j<layer.in_channels;j++){
            for(int k=0;k<layer.n;k++){
                free(layer.filters[i][j][k]);
            }
            free(layer.filters[i][j]);
        }
        free(layer.filters[i]);
    }
  
}

void free2dMatrix(float** matrix2d, int h){
    for (int i = 0; i < h; i++) {
        free(matrix2d[i]);
    }
    free(matrix2d);
}


void free3dMatrix(float*** matrix3d, int h, int w){
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            free(matrix3d[i][j]);
        }
        free(matrix3d[i]);
    }
    free(matrix3d);
}


//function to create the convolutional layer and initialize the filters with the weights
Conv2d createConv2d(int n,int in_channels,int out_channels,float weights[]) {
    Conv2d layer;
    layer.n = n;
    layer.out_channels = out_channels;
    layer.in_channels = in_channels;

    layer.filters = (float****)malloc(in_channels * sizeof(float***));
    for(int i=0; i < in_channels; i++) {
        layer.filters[i] = (float***)malloc(out_channels * sizeof(float**));
        for(int j=0; j < out_channels; j++) {
            layer.filters[i][j] = (float**)malloc(n * sizeof(float*));
            for(int k=0; k < n; k++) {
                layer.filters[i][j][k] = (float*)malloc(n * sizeof(float));
            }
        }
    }

    int count = 0;
    for(int j=0; j < out_channels; j++) {
        for(int i=0; i < in_channels; i++) {
            for(int k=0; k < n; k++) {
                for(int l=0; l < n; l++) {
                    layer.filters[i][j][k][l] = weights[count];
                    count++;
                }
            }
        }
    }
    return layer;
}


//function for the im2col of the filters
float** filter_matrix(Conv2d layer, int in_channel){
    int filterSize = layer.n * layer.n;
    float** filterMatrix = (float**)malloc(layer.out_channels * sizeof(float*));
    for(int i=0; i < layer.out_channels; i++) {
        filterMatrix[i] = (float*)malloc(filterSize * sizeof(float));
        for(int j=0; j < layer.n; j++) {
            for(int k=0; k < layer.n; k++) {
                filterMatrix[i][j * layer.n + k] = layer.filters[in_channel][i][j][k];
            }
        }
    }

    return filterMatrix;
}



//function to iterate over the regions of the image
float** iterateRegions(Conv2d layer, float** image, int h, int w) {       
    
    int numRegions = (h - (layer.n-1)) * (w - (layer.n-1));
    float** regions_matrix = (float**)malloc((layer.n*layer.n) * sizeof(float*));
    for (int i = 0; i < layer.n*layer.n; i++) {
        regions_matrix[i] = (float*)malloc(numRegions * sizeof(float));
    }
    int regionCount = 0;
    for (int i = 0; i < h - (layer.n-1); i++) {
        for (int j = 0; j < w - (layer.n-1); j++) {
            for (int m = 0; m < layer.n; m++) {
                for (int n = 0; n < layer.n; n++) {
                    regions_matrix[m * layer.n + n][regionCount] = image[i + m][j + n];
                }
            }
            regionCount++;
        }
    }
    free2dMatrix(image, h);

    return regions_matrix;
}

//forward function for the convolutional layer
float** conv_forward(Conv2d layer, float*** input, int input_h, int input_w, int input_channels){
    int output_h = layer.out_channels;
    int output_w = (input_w - (layer.n-1))*(input_h - (layer.n-1));

    float** conv_output = (float**)malloc(output_h*sizeof(float*));
    for(int i=0;i<output_h;i++){
        conv_output[i] = (float*)malloc(output_w*sizeof(float));
        for(int j=0;j<output_w;j++){
            conv_output[i][j] = 0;
        }
    }

    for(int k=0; k<input_channels;k++){
        float** regionsMatrix = iterateRegions(layer,input[k], input_h, input_w);
        float** filterMatrix = filter_matrix(layer, k);

        int M = output_h; // size of matrix A (rows)
        int K = layer.n*layer.n; // size of matrix A (columns) / size of matrix B (rows)
        int N = output_w; // size of matrix B (columns)
        int block_size = 64;

        //GEMM WITH TILING + CACHING
        for (int i = 0; i < M; i += block_size) {
            int imin = min(i + block_size, M);
            for (int k = 0; k < K; k += block_size) {
                int kmin = min(k + block_size, K);
                for (int j = 0; j < N; j += block_size) {
                    int jmin = min(j + block_size, N);
                    for (int x = i; x < imin; x++) {
                        for (int z = k; z < kmin; z++) {
                            float temp = filterMatrix[x][z];
                            for (int y = j; y < jmin; y++) {
                                conv_output[x][y]+= temp * regionsMatrix[z][y];
                            }
                        }
                    }
                }
            }
        }
    }
    return conv_output;
}

//forwad function for the pointwise convolution
float*** pointwise_forward(float*** input, int input_h, int input_w, int input_channels, int output_channels, float** pointwise_kernel){
    float*** output = (float***)malloc(output_channels * sizeof(float**));
    for (int i = 0; i < output_channels; i++) {
        output[i] = (float**)malloc(input_h * sizeof(float*));
        for (int j = 0; j < input_h; j++) {
            output[i][j] = (float*)malloc(input_w * sizeof(float));
        }
    }

    for(int i=0; i<output_channels; i++){
        for(int j=0; j<input_h; j++){
            for(int k=0; k<input_w; k++){
                float sum = 0;
                for(int l=0; l<input_channels; l++){
                    sum += input[l][j][k] * pointwise_kernel[i][l];
                }
                output[i][j][k] = sum;
            }
        }
    }


    free3dMatrix(input, input_channels, input_h);
    free2dMatrix(pointwise_kernel, output_channels);

    return output;
}
