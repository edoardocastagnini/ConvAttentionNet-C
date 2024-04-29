#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "conv.h"
#include "padding.h"
#include "maxpool.h"
#include "attention.h"
#include "weights/conv1.h"
#include "weights/conv2.h"
#include "weights/mla_qkv.h"
#include "weights/mla_proj.h"
#include "weights/fc.h"
#include "linear.h"
#include "softmax.h"


//this network is composed by 2 convolutional layers, 2 maxpooling layers, 1 attention block and 1 linear layer


#define NUM_FILTERS 8       //number of filters in the first convolutional layer
#define CIN_ATTENTION 64    //number of input channels in the attention block
#define N 3                 //size of the convolutional kernel
#define OUT_FEATURES 10     //number of output features in the linear layer


//function for the convolutional layer 
float*** conv2d(float*** input, int h, int w, int input_channels, int output_channels, int padding, int n, float weights[]){
    Conv2d layer = createConv2d(n,input_channels,output_channels,weights);
    int h_padding = h + 2 * padding;
    int w_padding = w + 2 * padding;
    int w_output = (h_padding-(layer.n-1))*(w_padding-(layer.n-1));
    int h_output =layer.out_channels;


    float*** input_padding = create_padding(input,h,w,input_channels,padding); 
    free3dMatrix(input,input_channels,h);

    float** output = conv_forward(layer,input_padding,h_padding,w_padding,input_channels);

    //reshape
    float*** output3d = (float***)malloc(h_output*sizeof(float**));
    for(int i=0; i<h_output; i++){
        output3d[i] = (float**)malloc(sqrt(w_output)*sizeof(float*));
        for(int j=0; j<sqrt(w_output); j++){
            output3d[i][j] = (float*)malloc(sqrt(w_output)*sizeof(float));
        }
    }

    int jj=sqrt(w_output);
    for(int i=0; i<h_output; i++){
        for(int j=0; j<jj; j++){
            for(int k=0; k<jj; k++){
                output3d[i][j][k] = output[i][j*jj+k];
            }
        }
    }
    free2dMatrix(output,h_output);

    return output3d;
}

//function for the pointwise convolution
float*** pointwiseConv(float*** input, int h, int w, int input_channels, int output_channels, float weights[]){
    
    //initialize weights
    float** pointwise_kernel = malloc(output_channels*sizeof(float*));
    for(int i=0; i<output_channels; i++){
        pointwise_kernel[i] = malloc(input_channels*sizeof(float));
    }
    for(int i=0; i<output_channels; i++){
        for(int j=0; j<input_channels; j++){
            pointwise_kernel[i][j] = weights[i*input_channels+j];
        } 
    }
  
    
    float*** output = pointwise_forward(input,h,w,input_channels,output_channels,pointwise_kernel);

    return output;
}


//function to run the complete network
float* network(float*** input, int h, int w,int input_channels, int padding){

	//CONVOLUTION FROM 3x32x32 TO 8x32x32
    float*** conv1 = conv2d(input,h,w,input_channels,NUM_FILTERS,padding,N,w_conv1);
    
    //reLU
    conv1 = reLU(conv1,h,w,NUM_FILTERS);

    //MAXPOOLING FROM 8x32x32 TO 8x16x16
    float*** maxpool1 = maxPoolforward(conv1,h,w,NUM_FILTERS);
    free3dMatrix(conv1,NUM_FILTERS,h);

    //CONVOLUTION FROM 8x16x16 TO 16x16x16
    float*** conv2 = conv2d(maxpool1,h/2,w/2,NUM_FILTERS,NUM_FILTERS*2,padding,N,w_conv2);

    //reLU
    conv2 = reLU(conv2,h/2,w/2,NUM_FILTERS*2);

    //MAXPOOLING FROM 16x16x16 TO 16x8x8
    float*** maxpool2 = maxPoolforward(conv2,h/2,w/2,NUM_FILTERS*2);
    free3dMatrix(conv2,NUM_FILTERS*2,h/2);
    

    //ATTENTION BLOCK ----------------------------------------------------------------
    //POINTWISE CONVOLUTION FROM 16x8x8 TO 64x8x8
    float*** pointwise_1 = pointwiseConv(maxpool2,h/4,w/4,NUM_FILTERS*2,CIN_ATTENTION, w_mla_qkv);

    //ATTENTION MECHANISM FROM 64x8x8 TO 32x8x8
    float*** attention = QKV_attention(pointwise_1,NUM_FILTERS*4,h/4,w/4);

    //POINTWISE CONVOLUTION FROM 32x8x8 TO 16x8x8
    float*** pointwise_2 = pointwiseConv(attention,h/4,w/4,CIN_ATTENTION/2,NUM_FILTERS*2,w_mla_proj);
    
    //---------------------------------------------------------------------------------

    //LINEAR LAYER FROM 16x8x8 TO 10
    float* output = linearForward(NUM_FILTERS*2*h/4*w/4,OUT_FEATURES,pointwise_2,h/4,w/4,NUM_FILTERS*2,w_fc);
    float* output_softmax = softmax(output,OUT_FEATURES);
    
    free3dMatrix(pointwise_2,NUM_FILTERS*2,h/4);

    return output_softmax;
}
