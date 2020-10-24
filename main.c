#include <stdio.h>
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
    printf("%f",data[0][1]);
    return data;

}


float linearRegression(float** data){
    int bias = 0;
    float weights[10];
    for(int i = 0; i<DATA_COLUMNS; i++){
        weights[i]=(float)rand()/(float)(RAND_MAX);
        printf("\n%f",weights[i]);
    }
    float total = 0;
    for(int i = 0; i<DATA_ROWS; i++){
        for(int j = 0; j<DATA_COLUMNS; j++){
            total += weights[j] * data[i][j];
        }
        total += bias;
    }
    return total;

}

int main(){
    float** data =openData("/Users/chuny/Downloads/fertility_Diagnosis_Data_Group1_4-1.txt");
    float** training = malloc(TRAINING_MAX*sizeof(float*));
    float** testing = malloc(TESTING_MAX*sizeof(float*));
    training=data;
    testing=data+90;
    printf("%f",training[0][1]);
    printf("%f",testing[0][1]);
    float sumLR=linearRegression(data);
    printf("\n%f",sumLR);
    return 0;
}
