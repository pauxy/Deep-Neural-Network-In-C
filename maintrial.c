#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
const int TRAINING_MAX = 90;
const int TESTING_MAX = 10;
const int DATA_COLUMNS = 10;
const int DATA_ROWS=100;

double** openData(char* filename){
    double* val = calloc(DATA_ROWS*DATA_ROWS,sizeof(double));
    double** data=malloc(DATA_ROWS*sizeof(double*));
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


double linearRegression(double** data, double* biasWeights){
    double total = 0;
    double bias = *biasWeights;
    for(int i = 0; i<TRAINING_MAX; i++){
        for(int j = 0; j<DATA_COLUMNS; j++){
            total += *(biasWeights+1+j) * data[i][j];
        }
        total += bias;
    }
    return total;

}


double sigmoid(double sumLR){
    return 1/(1+exp(-sumLR));
}


double meanAbsolutevalue(double** training, double activatedVal){
    double total = 0;
    for (int i=0; i<TRAINING_MAX; i++){
        total+=activatedVal-training[i][DATA_COLUMNS-1];
    }
    return (total/TRAINING_MAX)<0?-(total/TRAINING_MAX):(total/TRAINING_MAX);
}


double* backwardsPropagation(double* biasWeights, double activatedVal, double** training,double sumLR){
    double* newBiasweights=malloc(DATA_COLUMNS+1*sizeof(double));
    double ph = (exp(sumLR)/((1+exp(sumLR))*(1+exp(sumLR))));
    double biasTotal =0;
    for(int j = 0; j<DATA_COLUMNS; j++){
        double weightTotal=0;
        for(int i = 0; i<TRAINING_MAX; i++){
            double ph1 = activatedVal-training[i][DATA_COLUMNS-1];
            weightTotal += (ph*ph1*training[i][j+1]);
            if(j==0){
                biasTotal += ph*ph1;
            }
        }
        *(newBiasweights+1+j)= *(biasWeights+1+j)-(weightTotal/TRAINING_MAX);
    }
    *newBiasweights=*(biasWeights)-(biasTotal/TRAINING_MAX);
    return newBiasweights;
    }

int main(){
    FILE* graph=fopen("graph.temp","w");

    double** data =openData("/Users/chuny/Downloads/fertility_Diagnosis_Data_Group1_4-1.txt");
    double** training = malloc(TRAINING_MAX*sizeof(double*));
    double** testing = malloc(TESTING_MAX*sizeof(double*));
    training=data;
    testing=data+90;
    srand(time(NULL));
    double* biasWeights = malloc(DATA_COLUMNS+1*sizeof(double));
    *biasWeights=-1.5;
    for (int i = 1;i<DATA_COLUMNS+1; i++){
        *(biasWeights+i) = (double)rand()/(double)(RAND_MAX);
        printf("Random Number %i:%f\n",i,*(biasWeights+i));
    }
    int t=0;
    double sumLR=0;
    double activatedVal=0;
    double maeVal=0;
     FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
    do{
        if(t>0){
            biasWeights=backwardsPropagation(biasWeights,activatedVal,training,sumLR);
        }
        sumLR=linearRegression(training,biasWeights);
        activatedVal=sigmoid(sumLR);
        maeVal=meanAbsolutevalue(training,activatedVal);
        t+=1;
        if(t%1000==0||t<1000){
            printf("Sum of LR:i %f\n",sumLR);
            printf("Sigmoid value: %0.300f\n",activatedVal);
            printf("MAE value: %f\n",maeVal);
            printf("T value: %i\n",t);
            for (int i = 0;i<DATA_COLUMNS+1; i++){
                printf("New Number %i:%f\n",i,*(biasWeights+i));
            }
            fprintf(graph,"%d\t %f\n",t,maeVal);

        
        }
    }while(maeVal>.25);
    fprintf(gnuplotPipe,"set style func linespoints \n");
    fprintf(gnuplotPipe, "plot 'graph.temp' lt 1 pt 6 lc 3 title 'mae' \n");
    return 0;
}
