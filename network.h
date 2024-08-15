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
#include "weights/conv3.h"
#include "weights/mla_qkv.h"
#include "weights/mla_proj.h"
#include "weights/fc.h"
#include "linear.h"
#include "softmax.h"


//this network is composed by 2 convolutional layers, 2 maxpooling layers, 1 attention block and 1 linear layer


#define NUM_FILTERS 8      //number of filters in the first convolutional layer
#define CIN_ATTENTION 256    //number of input channels in the attention block
#define N 3                 //size of the convolutional kernel
#define OUT_FEATURES 10     //number of output features in the linear layer


//function for the convolutional layer 
float*** conv2d(float*** input, int h, int w, int input_channels, int output_channels, int padding, int n, float* weights){
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
float*** pointwiseConv(float*** input, int h, int w, int input_channels, int output_channels, float* weights){
    
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
    free(weights);
  
    
    float*** output = pointwise_forward(input,h,w,input_channels,output_channels,pointwise_kernel);

    return output;
}


//function to run the complete network
float* network(float*** input, int h, int w,int input_channels, int padding){

    //32x32x3
    float*** conv1 = conv2d(input,h,w,input_channels,NUM_FILTERS,padding,N,weights_conv1(N*N*input_channels*NUM_FILTERS));
    conv1 = reLU(conv1,h,w,NUM_FILTERS);
    float*** maxpool1 = maxPoolforward(conv1,h,w,NUM_FILTERS);
    //16x16x32


    //16x16x32
    float*** conv2 = conv2d(maxpool1,h/2,w/2,NUM_FILTERS,NUM_FILTERS*2,padding,N,weights_conv2(N*N*NUM_FILTERS*NUM_FILTERS*2));
    conv2 = reLU(conv2,h/2,w/2,NUM_FILTERS*2);
    float*** maxpool2 = maxPoolforward(conv2,h/2,w/2,NUM_FILTERS*2);
    //8x8x64

    //8x8x64
    float*** conv3 = conv2d(maxpool2,h/4,w/4,NUM_FILTERS*2,NUM_FILTERS*4,padding,N,weights_conv3(N*N*NUM_FILTERS*2*NUM_FILTERS*4));
    conv3 = reLU(conv3,h/4,w/4,NUM_FILTERS*4);
    float*** maxpool3 = maxPoolforward(conv3,h/4,w/4,NUM_FILTERS*4);
    //4x4x128
    //4x4x128

    //ATTENTION BLOCK ----------------------------------------------------------------
    float*** pointwise_1 = pointwiseConv(maxpool3,h/8,w/8,NUM_FILTERS*4,CIN_ATTENTION, weights_mla_qkv(NUM_FILTERS*4*CIN_ATTENTION));

    //4x4x256
    float*** attention = QKV_attention(pointwise_1, CIN_ATTENTION,h/8,w/8);
    //4x4x128


    float*** pointwise_2 = pointwiseConv(attention,h/8,w/8,CIN_ATTENTION/2,NUM_FILTERS*4,weights_mla_proj(CIN_ATTENTION/2*NUM_FILTERS*4));
    //4x4x128
    //---------------------------------------------------------------------------------


    float* avgpool_output = avgpool(pointwise_2,NUM_FILTERS*4,h/8,w/8);

   // float* output = linearForward(NUM_FILTERS*4*h/8*w/8,OUT_FEATURES,pointwise_2,h/8,w/8,NUM_FILTERS*4,w_fc);
    float* output = linearForwardflat(NUM_FILTERS*4,OUT_FEATURES,avgpool_output,weights_fc(NUM_FILTERS*4*OUT_FEATURES));


    float* output_softmax = softmax(output,OUT_FEATURES);
    

    return output_softmax;
}

/*
32x32x3 --conv1--> 32x32x32 
--maxpool1--> 16x16x32 
--conv2--> 16x16x64 
--maxpool2--> 8x8x64 
--conv3--> 8x8x128 
--maxpool3--> 4x4x128 
--pointwise--> 4x4x256
--attention--> 4x4x128
--pointwise--> 4x4x128
--linear--> 100
--softmax--> 100



*/