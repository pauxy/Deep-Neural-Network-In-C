#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void opendata(char* filename){
    float data[100][9];
    FILE* filelist;
    filelist=fopen(filename, "r");
    char line[256];
    int count=0;
    while(fgets(line, sizeof(line), filelist)!=NULL){
        count=count+1;
        for(int i=0;i<10;i++){
            char* new;
            if (i==0){
                new=strtok(line,",");
            }else{
                new=strtok(NULL,",");
            }
            printf("%d  %s ",count,new);
        //    data[count][i]=atof(new);
        }
    }
    //return data;
}

int main(){
    opendata("/Users/chuny/Downloads/fertility_Diagnosis_Data_Group1_4-1.txt");
    //printf("%d",data[0][0]);
}
