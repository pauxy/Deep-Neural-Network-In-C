#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

const int TRAINING_MAX = 90;
const int TESTING_MAX = 10;
const int DATA_COLUMNS = 10;
const int DATA_ROWS = 100;
const double LEARNING_RATE = 0.05;
const int ATTR_COLUMNS = DATA_COLUMNS - 1; //cols exclusive of results

double** openData(char*);
double* linearRegression(double**, double*, int);
double* sigmoid(double*, int);
double meanAbsoluteValue(double**, double*, int);
double* backwardsPropagation(double*, double*, double**, double*);
double minMeanSquareError(double**, double*, int);

double** openData(char* filename) {
    double* val = (double*)malloc(DATA_ROWS * DATA_ROWS * sizeof(double));
    double** data = (double**)malloc(DATA_ROWS * sizeof(double*));
    FILE* filelist;
    filelist = fopen(filename, "r");
    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), filelist) != NULL) {   //while file still has lines,
        data[count] = val + (count * DATA_ROWS);            //2d array assign
        char* new = strtok(line, ",");                      //gets first data between , 
        for (int i = 0; i < DATA_COLUMNS; i++) {            
            if (i != 0) new = strtok(NULL, ",");            //gets remaining data between ,

            //printf("%d  %f  %d \n", count, atof(new), i);   //printf for testing
            data[count][i] = atof(new);                     //convert string to float and assign
        }
        count += 1;                                         //counter
    }
    return data;
}


double* linearRegression(double** data, double* biasWeights, int val){  //2a
    double* lr = (double*)malloc(val * sizeof(double));
    for (int rows = 0; rows < val; rows++) {
        *(lr + rows) = 0;
        for (int cols = 0; cols < ATTR_COLUMNS; cols++) {
            //printf("%f\n",*(lr + rows));
            *(lr + rows) += (*(biasWeights + 1 + cols) * data[rows][cols])+*biasWeights;
        }
    }
    return lr;
}


double* sigmoid(double* lr, int val) {                                  //2b
    double* activatedVal = (double*)malloc(val * sizeof(double));
    for(int rows = 0; rows < val; rows++){
        *(activatedVal + rows) = 1.0 / (1.0 + exp(- *(lr + rows)));
    }
    return activatedVal;
}


double meanAbsoluteValue(double** training, double* activatedVal, int val) {    //2c
    double total = 0.0;
    for (int rows = 0; rows < val; rows++) {
        total += fabs( *(activatedVal + rows) - training[rows][DATA_COLUMNS - 1]);
    }
    return total / val;
}


double* backwardsPropagation(double* biasWeights, double* activatedVal,
                             double** training, double* lr) {                   //2d
    double* newBiasWeights = (double*)malloc((ATTR_COLUMNS + 1) * sizeof(double));
    double biasTotal = 0.0;
    for (int cols = 0; cols < ATTR_COLUMNS; cols++) {
        double weightTotal = 0.0;
        for (int rows = 0; rows < TRAINING_MAX; rows++) {
            double ph = exp( *(lr + cols)) / pow(1.0 + exp( *(lr + cols)), 2.0);
            double ph1 = *(activatedVal + cols) - training[rows][DATA_COLUMNS - 1];
            weightTotal += (ph * ph1 * training[rows][cols]);
            if (cols == 0) {
                biasTotal += ph * ph1;
            }
        }
        *(newBiasWeights + 1 + cols) = *(biasWeights + 1 + cols) - (LEARNING_RATE * (weightTotal / TRAINING_MAX));
    }
    *newBiasWeights = *(biasWeights) - (LEARNING_RATE * (biasTotal / TRAINING_MAX));
    return newBiasWeights;
}


double minMeanSquareError(double** training, double* activatedVal, int val){
    double total = 0.0;
    for (int rows = 0; rows < val; rows++) {
        total += pow(*(activatedVal + rows) - training[rows][DATA_COLUMNS - 1],2.0);
    }
    return total / val ;
}
int* confusionMatrix(double* res, double** data,int val){
    double* confusion = (double*)malloc(val*sizeof(double));
    for (int i=0;i<val;i++){
        int origin=data[i][DATA_COLUMNS-1];
        int result=res[i];
        int con=1;//
            if (origin==result){
                if (origin==0){
                    con=0;//true neg
                }
                //1=true positive
            }else{
                con=3;//false positive
                if (origin==0){
                    con=2;//false neg
                }

            }
            printf("%i",con);
            *confusion=con;
            confusion++;
    }
}

int main() {
    FILE* graph = fopen("graph.temp","w");
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
    double** data = openData("dataset/fertility_Diagnosis_Data_Group1_4-1.txt");
    double** training = (double**)malloc(TRAINING_MAX * sizeof(double*));
    double** testing = (double**)malloc(TESTING_MAX * sizeof(double*));
    training = data;
    testing = data + 90;
    srand(time(NULL));
    double* biasWeights = (double*)malloc((ATTR_COLUMNS + 1) * sizeof(double));//allocate space for weights +1 bias
    *biasWeights = 0;
    for (int cols = 1; cols < ATTR_COLUMNS + 1; cols++) {
        *(biasWeights + cols) = (double)rand() / (double)RAND_MAX;//randomly assign weights and bias
        printf("Random Number %i:%f\n", cols, *(biasWeights + cols));
    }
    int t = 0;
    double* lr =  (double*)malloc(TRAINING_MAX * sizeof(double));
    double* activatedVal = (double*)malloc(TRAINING_MAX * sizeof(double));
    double maeVal = 0.0;

    do {
        lr = linearRegression(training, biasWeights,TRAINING_MAX);
        activatedVal = sigmoid(lr,TRAINING_MAX);
        maeVal = meanAbsoluteValue(training, activatedVal,TRAINING_MAX);
        t += 1;

        if (t % 10 == 0) {                              //testing
            printf("MAE value: %f\n",maeVal);
            printf("T value: %i\n",t);
            printf("sigmoid: %f\n",*(activatedVal));
        }
        fprintf(graph, "%i %lf \n", t, maeVal);
        if(maeVal>0.25){
            biasWeights = backwardsPropagation(biasWeights, activatedVal, training, lr);
        }
    } while(maeVal>0.25);
//    fprintf(gnuplotPipe, "plot 'graph.temp' with lines\n");
    printf("MMSE Training: %f\n",minMeanSquareError(training,activatedVal,TRAINING_MAX));
    printf("MMSE Testing: %f\n",minMeanSquareError(testing,activatedVal,TESTING_MAX));
    //TESTING
    double* testLR =  (double*)malloc(TESTING_MAX* sizeof(double));
    testLR=linearRegression(testing,biasWeights,TESTING_MAX);
    testLR=sigmoid(testLR,TESTING_MAX);
    for(int i=0;i<TRAINING_MAX;i++){
        //printf("test: %f\n",*(testLR+i));
        if(*(activatedVal+i)>0.25){
            *(activatedVal+i)=1;//printf("1                  %f\n",training[i][DATA_COLUMNS - 1]);
        }else{
            *(activatedVal+i)=0;
    }
    }
    confusionMatrix(activatedVal,testing,10);
    return 0;
}
