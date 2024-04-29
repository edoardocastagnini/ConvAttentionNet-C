#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "reLU.h"

#define HEADS 4 //number of heads
#define DV 8    //dimension of Value matrix
#define DQK 4   //dimension of Query and Key matrices


float*** QKV_attention(float*** input,int c, int h, int w){

    // The initial matrix has 64 channels. Channels with indices 0-3, 16-19, 32-35, 48-51 are Q. Of these 16 channels, the first 4 form the 4 columns of length h*w for the first HEAD, the next 4 form the 4 columns of length h*w for the second HEAD, and so on.
    // The next 16 channels with indices 4-7, 20-23, 36-39, 52-55 are K. Of these 16 channels, the first 4 form the 4 columns of length h*w for the first HEAD, the next 4 form the 4 columns of length h*w for the second HEAD, and so on.
    // The next 32 channels with indices 8-15, 24-31, 40-47, 56-63 are V. Of these 32 channels, the first 8 form the 8 columns of length h*w for the first HEAD, the next 8 form the 8 columns of length h*w for the second HEAD, and so on.

    // Convert the cx(h*w) matrix into three matrices V, K, Q: 
    //      -V has dimensions HEADSx(h*w)xDV, 
    //      -K has dimensions HEADSxDQKx(h*w), 
    //      -Q has dimensions HEADSx(h*w)xDQK.
    // Then, add a channel of 1 to V for each row, resulting in a matrix of dimensions HEADSx(h*w)x(DV+1).


    float*** Q = (float***)malloc(HEADS*sizeof(float**));
    for(int i=0;i<HEADS;i++){
        Q[i] = (float**)malloc(h*w*sizeof(float*));
        for(int j=0;j<h*w;j++){
            Q[i][j] = (float*)malloc(DQK*sizeof(float));
        }
    }

    for(int i=0;i<HEADS;i++){
        for(int k=0;k<DQK;k++){
            for(int j=0;j<h*w;j++){
                Q[i][j][k] = input[k + i * HEADS * DQK][j / w][j % w];
            }
        }
    }


    float*** K = (float***)malloc(HEADS*sizeof(float**));
    for(int i=0;i<HEADS;i++){
        K[i] = (float**)malloc(DQK*sizeof(float*));
        for(int j=0;j<DQK;j++){
            K[i][j] = (float*)malloc(h*w*sizeof(float));
        }
    }

    for(int i=0;i<HEADS;i++){
        for(int j=0;j<DQK;j++){
            for(int k=0;k<h*w;k++){
                K[i][j][k] = input[j + DQK + i * HEADS * DQK][k / w][k % w];
            }
        }
    }


    float*** V = (float***)malloc(HEADS*sizeof(float**));
    for(int i=0;i<HEADS;i++){
        V[i] = (float**)malloc(h*w*sizeof(float*));
        for(int j=0;j<h*w;j++){
            V[i][j] = (float*)malloc((DV+1)*sizeof(float));
        }
    }

    for(int i=0;i<HEADS;i++){
        for(int j=0;j<DV;j++){
            for(int k=0;k<h*w;k++){
                V[i][k][j] = input[j + DQK * 2 + i * HEADS * DQK][k / w][k % w];
            }
        }
        //add a channel of 1 to V for each row
        for(int j=0;j<h*w;j++){
            V[i][j][DV] = 1;
        }
    }


    free3dMatrix(input,c,h);


    //reLU activation function for Query and Key matrices
    Q = reLU(Q,h*w,DQK,HEADS);
    K = reLU(K,DQK,h*w,HEADS);


    //multiply K and V matrices, obtaining a matrix of dimensions HEADSxDQKxDV+1
    float*** VK = (float***)malloc(HEADS*sizeof(float**));
    for(int i=0;i<HEADS;i++){
        VK[i] = (float**)malloc(DQK*sizeof(float*));
        for(int j=0;j<DQK;j++){
            VK[i][j] = (float*)malloc((DV+1)*sizeof(float));
        }
    }

    for(int i=0;i<HEADS;i++){
        float** tempV = V[i];
        float** tempK = K[i];
        float** tempVK = VK[i];

        int tileSize = 16;
        for(int j=0;j<DQK;j+=tileSize){
            for(int l=0;l<h*w;l+=tileSize){
                for(int jj=j; jj<min(j+tileSize, DQK); jj++){
                    for(int ll=l; ll<min(l+tileSize, h*w); ll++){
                    float temp = tempK[jj][ll];
                        for(int k=0;k<DV+1;k++){
                            tempVK[jj][k] +=  temp * tempV[ll][k];
                        }
                    }
                }
            }
        }
        VK[i] =tempVK;

    }


    free3dMatrix(V,HEADS,h*w);
    free3dMatrix(K,HEADS,DQK);


    //multiply Q and VK matrices, obtaining a matrix of dimensions HEADSx(h*w)x(DV+1)

    float*** QKV = (float***)malloc(HEADS*sizeof(float**));
    for(int i=0;i<HEADS;i++){
        QKV[i] = (float**)malloc(h*w*sizeof(float*));
        for(int j=0;j<h*w;j++){
            QKV[i][j] = (float*)malloc((DV+1)*sizeof(float));
        }
    }

    for(int i=0;i<HEADS;i++){
        float** tempQ = Q[i];
        float** tempVK = VK[i];
        float** tempQKV = QKV[i];

        int tileSize = 16;
        for(int j=0;j<h*w;j+=tileSize){
            for(int l=0;l<DQK;l+=tileSize){
                for(int jj=j; jj<min(j+tileSize, h*w); jj++){
                    for(int ll=l; ll<min(l+tileSize, DQK); ll++){
                    float temp = tempQ[jj][ll];
                        for(int k=0;k<DV+1;k++){
                            tempQKV[jj][k] += temp * tempVK[ll][k];
                        }
                    }
                }
            }
        }

        QKV[i] = tempQKV;

    }

    free3dMatrix(VK,HEADS,DQK);
    free3dMatrix(Q,HEADS,h*w);


    //divide each element of QKV for the last element of each row
    for(int i=0;i<HEADS;i++){
        float** tempQKV = QKV[i];

        for(int j=0;j<h*w;j++){
        	float temp = tempQKV[j][DV];
            for(int k=0;k<DV;k++){
                tempQKV[j][k] = tempQKV[j][k]/(temp+1.0e-15);
            }
        }

        QKV[i] = tempQKV;
    }

    //Free the last element of each row, returning to a matrix of dimensions HEADSx(h*w)xDV
    for(int i=0;i<HEADS;i++){
        for(int j=0;j<h*w;j++){
            QKV[i][j] = realloc(QKV[i][j],DV*sizeof(float));
        }
    }



    //Return to the original shape, obtaining a matrix of dimensions (HEADS*DV)xhxw
    float*** reshaped_QKV = (float***)malloc(HEADS*DV*sizeof(float**));
    for(int i=0;i<HEADS*DV;i++){
        reshaped_QKV[i] = (float**)malloc(h*sizeof(float*));
        for(int j=0;j<h;j++){
            reshaped_QKV[i][j] = (float*)malloc(w*sizeof(float));
        }
    }

    for(int i=0;i<HEADS;i++){
        for(int j=0;j<h*w;j++){
            for(int k=0;k<DV;k++){
                reshaped_QKV[i*DV+k][j/w][j%w] = QKV[i][j][k];
            }
        }
    }

    free3dMatrix(QKV,HEADS,h*w);

    return reshaped_QKV;

}
