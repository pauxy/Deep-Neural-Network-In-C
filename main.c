#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
const int TRAINING_MAX = 90;
const int TESTING_MAX = 10;
const int DATA_COLUMNS = 10;
const int DATA_ROWS=100;

float** openData(char* filename){
    float* val = calloc(DATA_ROWS*DATA_ROWS,sizeof(float));
    float** data=malloc(DATA_ROWS*sizeof(float*));
    FILE* filelist;
    filelist=fopen(filename, "r");
    char line[256];
    int count=0;
    while(fgets(line, sizeof(line), filelist)!=NULL){
        data[count] = val+(count*DATA_ROWS);
        for(int i = 0; i<DATA_COLUMNS; i++){
            char* new;
            if (i == 0){
                new = strtok(line,",");
            }else{
                new = strtok(NULL,",");
            }
            printf("%d  %f  %d \n",count,atof(new),i);
            data[count][i] = atof(new);
        }
        count = count+1;
    }
    return data;

}


float linearRegression(float** data,int bias, float* weights){
    float total = 0;
    for(int i = 0; i<TRAINING_MAX; i++){
        for(int j = 0; j<DATA_COLUMNS; j++){
            total += *(weights+j) * data[i][j];
        }
        total += bias;
    }
    return total;

}


float sigmoid(float sumLR){
    return 1/(1+exp(-sumLR));
}


float mae(float** data, float activatedVal){
    float total = 0;
    for (int i=0; i<TRAINING_MAX; i++){
        total+=activatedVal-data[i][DATA_COLUMNS-1];
    }
    return total/DATA_ROWS;
}


//float backProp

int main(){
    float** data =openData("/Users/chuny/Downloads/fertility_Diagnosis_Data_Group1_4-1.txt");
    float** training = malloc(TRAINING_MAX*sizeof(float*));
    float** testing = malloc(TESTING_MAX*sizeof(float*));
    training=data;
    testing=data+90;
    srand(time(NULL));
    float* weights = malloc(DATA_COLUMNS*sizeof(float));
    for (int i = 0;i<DATA_COLUMNS; i++){
        *(weights+i) = (float)rand()/(float)(RAND_MAX);
        printf("Random Number %i:%f\n",i,*(weights+i));
    }

    float sumLR=linearRegression(training,0,weights);
    float activatedVal=sigmoid(sumLR);
    float maeVal=mae(training,activatedVal);

    printf("Element 1 1 in training: %f\n",training[0][1]);
    printf("Element 1 1 in testing: %f\n",testing[0][1]);
    printf("Sum of LR: %f\n",sumLR);
    printf("Sigmoid value: %f\n",activatedVal);
    printf("MAE value: %f\n",maeVal);
    
    return 0;
}
