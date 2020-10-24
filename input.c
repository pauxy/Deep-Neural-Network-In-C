#include <stdio.h>
#include <string.h>
#include <stdlib.h>

float** opendata(char* filename){
    float* val = calloc(100*100,sizeof(float));
    float** data=malloc(100*sizeof(float*));
    FILE* filelist;
    filelist=fopen(filename, "r");
    char line[256];
    int count=0;
    while(fgets(line, sizeof(line), filelist)!=NULL){
        data[count]=val+(count*100);
        for(int i=0;i<10;i++){
            char* new;
            if (i==0){
                new=strtok(line,",");
            }else{
                new=strtok(NULL,",");
            }
            printf("%d  %f  %d \n",count,atof(new),i);
            data[count][i]=atof(new);
        }
        count=count+1;
    }
    printf("%f",data[0][1]);
    return data;

}


int main(){
    float** data =opendata("/Users/chuny/Downloads/fertility_Diagnosis_Data_Group1_4-1.txt");
    float** training = malloc(90*sizeof(float*));
    float** testing = malloc(10*sizeof(float*));
    training=data;
    testing=data+90;
    printf("%f",training[0][1]);
    printf("%f",testing[0][1]);
    return 0;
}
