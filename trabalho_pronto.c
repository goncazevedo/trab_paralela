#include <stdio.h>
#include <stdlib.h>
#include <ncurses.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
int procNum;            // Number of the available processes
int procRank;           // Rank of the current process
int *pParallelPivotPos; // Number of rows selected as the pivot ones
int *pProcPivotIter;    // Number of iterations, at which the processor
                        // rows were used as the pivot ones
int *pProcInd;          // Number of the first row located on the processes
int *pProcNum;          // Number of the linear system rows located on the processes

// Function for formatted matrix output
void PrintMatrix(double *pMatrix, int rowCount, int colCount)
{
    int i, j; // Loop variables
    for (i = 0; i < rowCount; i++)
    {
        for (j = 0; j < colCount; j++)
            printf("%7.4f ", pMatrix[i * colCount + j]);
        printf("\n");
    }
}

// Function for simple definition of matrix and vector elements
void DummyDataInitialization(double *pMatrix, double *pVector, int size)
{
    int i, j; // Loop variables
    for (i = 0; i < size; i++)
    {
        pVector[i] = i + 1;
        for (j = 0; j < size; j++)
        {
            if (j <= i)
                pMatrix[i * size + j] = 1;
            else
                pMatrix[i * size + j] = 0;
        }
    }
}
// Function for random definition of matrix and vector elements
void RandomDataInitialization(double *pMatrix, double *pVector, int size)
{
    int i, j; // Loop variables
    srand(clock());
    for (i = 0; i < size; i++)
    {
        pVector[i] = rand() / 1000;
        for (j = 0; j < size; j++)
        {
            if (j <= i)
                pMatrix[i * size + j] = rand() / 1000;
            else
                pMatrix[i * size + j] = 0;
        }
    }
}
// Function for memory allocation and data initialization
void ProcessInitialization(double **pMatrix, double **pVector,
                           double **pResult, double **pProcRows, double **pProcVector,
                           double **pProcResult, int *size, int *rowNum)
{
    int restRows; // Number of rows, that haven't been distributed yet
    int i;        // Loop variable
    if (procRank == 0)
    {
        do
        {
            printf("\nEnter the size of the matrix and the vector: ");
            scanf("%d", size);
            if (*size < procNum)
            {
                printf("size must be greater than number of processes! \n");
            }
        } while (*size < procNum);
    }
    MPI_Bcast(size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    restRows = *size;
    for (i = 0; i < procRank; i++)
        restRows = restRows - restRows / (procNum - i);
    *rowNum = restRows / (procNum - procRank);
    *pProcRows = malloc(sizeof(double) * (*rowNum * (*size)));
    *pProcVector = malloc(sizeof(double)* (*rowNum));
    *pProcResult = malloc(sizeof(double)* (*rowNum));
    
    pParallelPivotPos = malloc(sizeof(int) * (*size));
    pProcPivotIter = malloc(sizeof(int) * (*rowNum));
    pProcInd = malloc(sizeof(int)* procNum);
    pProcNum = malloc(sizeof(int)* procNum);

    for (int i = 0; i < *rowNum; i++)
        pProcPivotIter[i] = -1;

    if (procRank == 0)
    {
        *pMatrix = malloc(sizeof(double)* (*size) * (*size));
        *pVector = malloc(sizeof(double)* *size);
        *pResult = malloc(sizeof(double)* *size);    
        DummyDataInitialization(*pMatrix, *pVector, *size);    
        // RandomDataInitialization(pMatrix, pVector, size);
    }
}
// Function for the data distribution among the processes
void DataDistribution(double *pMatrix, double *pProcRows, double *pVector,
                      double *pProcVector, int size, int rowNum)
{
    int *pSendNum;       // Number of the elements sent to the process
    int *pSendInd;       // Index of the first data element sent to the process
    int restRows = size; // Number of rows, that have not been distributed yet
    int i;               // Loop variable
    // Alloc memory for temporary objects
    pSendInd = malloc(sizeof(int)* procNum);
    pSendNum = malloc(sizeof(int)* procNum);
    // Define the disposition of the matrix rows for the current process
    rowNum = (size / procNum);
    pSendNum[0] = rowNum * size;
    pSendInd[0] = 0;
    for (i = 1; i < procNum; i++)
    {
        restRows -= rowNum;
        rowNum = restRows / (procNum - i);
        pSendNum[i] = rowNum * size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }
    
    // Scatter the rows
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
                 pSendNum[procRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Define the disposition of the matrix rows for current process

    restRows = size;
    pProcInd[0] = 0;
    pProcNum[0] = size / procNum;
    for (i = 1; i < procNum; i++)
    {
        restRows -= pProcNum[i - 1];
        pProcNum[i] = restRows / (procNum - i);
        pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
    }
    
    MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector,
                 pProcNum[procRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    // Free the memory
    free(pSendNum);
    free(pSendInd);
}
// Function for gathering the result vector
void ResultCollection(double *pProcResult, double *pResult)
{
    // Gather the whole result vector on every processor
    MPI_Gatherv(pProcResult, pProcNum[procRank], MPI_DOUBLE, pResult,
                pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Function for formatted vector output
void PrintVector(double *pVector, int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%7.4f ", pVector[i]);
}
// Function for formatted vector output
void PrintResultVector(double *pResult, int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%7.4f ", pResult[pParallelPivotPos[i]]);
}
// Fuction for the column elimination
void ParallelEliminateColumns(double *pProcRows, double *pProcVector,
                              double *pPivotRow, int size, int rowNum, int iter)
{
    
    // printf("antes \n");
    // for (int i = 0; i < rowNum; i++)
    // {
    //     printf("row[%d] = %f\n", i,pPivotRow[i]);
    // }
    double multiplier;
    for (int i = 0; i < rowNum; i++)
    {
        if (pProcPivotIter[i] == -1)
        {
            multiplier = pProcRows[i * size + iter] / pPivotRow[iter];
            for (int j = iter; j < size; j++)
            {
                pProcRows[i * size + j] -= pPivotRow[j] * multiplier;
            }
            pProcVector[i] -= pPivotRow[size] * multiplier;
        }
    }
//     printf("depois \n");
//     for (int i = 0; i < rowNum; i++)
//     {
//         printf("row[%d] = %f\n", i,pPivotRow[i]);
//     }
}
// Function for the Gaussian elimination
void ParallelGaussianElimination(double *pProcRows, double *pProcVector,
                                 int size, int rowNum)
{
    double maxValue; // Value of the pivot element of thе process
    int pivotPos;    // Position of the pivot row in the process stripe
    // Structure for the pivot row selection 37
    struct
    {
        double maxValue;
        int procRank;
    } ProcPivot, Pivot;
    
    // pPivotRow is used for storing the pivot row and the corresponding
    // element of the vector b
    double *pPivotRow = malloc(sizeof(double)* (size + 1));
    // The iterations of the Gaussian elimination stage
    for (int i = 0; i < size; i++)
    {

        // Calculating the local pivot row
        double maxValue = 0;
        for (int j = 0; j < rowNum; j++)
        {
            if ((pProcPivotIter[j] == -1) &&
                (maxValue < fabs(pProcRows[j * size + i])))
            {
                maxValue = fabs(pProcRows[j * size + i]);
                pivotPos = j;
            }
        }
        ProcPivot.maxValue = maxValue;
        ProcPivot.procRank = procRank;
        // Finding the pivot process
        // (process with the maximum value of maxValue)
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);
        // Broadcasting the pivot row
        if (procRank == Pivot.procRank)
        {
            pProcPivotIter[pivotPos] = i; //iteration number
            pParallelPivotPos[i] = pProcInd[procRank] + pivotPos;
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.procRank,
                  MPI_COMM_WORLD);

        if (procRank == Pivot.procRank)
        {
            // Fill the pivot row
            for (int j = 0; j < size; j++)
            {
                pPivotRow[j] = pProcRows[pivotPos * size + j];
            }
            pPivotRow[size] = pProcVector[pivotPos];
        }
        MPI_Bcast(pPivotRow, size + 1, MPI_DOUBLE, Pivot.procRank,
                  MPI_COMM_WORLD);
        ParallelEliminateColumns(pProcRows, pProcVector, pPivotRow,
                                 size, rowNum, i);
    }
}
// Function for finding the pivot row of the back substitution
void FindBackPivotRow(int rowIndex, int *iterProcRank,
                      int *iterPivotPos)
{
    for (int i = 0; i < procNum - 1; i++)
    {
        if ((pProcInd[i] <= rowIndex) && (rowIndex < pProcInd[i + 1]))
            *iterProcRank = i;
    }
    if (rowIndex >= pProcInd[procNum - 1])
        *iterProcRank = procNum - 1;
    *iterPivotPos = rowIndex - pProcInd[*iterProcRank];
}
// Function for the back substitution
void ParallelBackSubstitution(double *pProcRows, double *pProcVector,
                              double *pProcResult, int size, int rowNum)
{
    int iterProcRank;  // Rank of the process with the current pivot row
    int iterPivotPos;  // Position of the pivot row of the process
    double iterResult; // Calculated value of the current unknown
    double val;
    // Iterations of the back substitution stage
    for (int i = size - 1; i >= 0; i--)
    {
        // Calculating the rank of the process, which holds the pivot row
        FindBackPivotRow(pParallelPivotPos[i], &iterProcRank, &iterPivotPos);

        // Calculating the unknown
        if (procRank == iterProcRank)
        {
            iterResult =
                pProcVector[iterPivotPos] / pProcRows[iterPivotPos * size + i];
            pProcResult[iterPivotPos] = iterResult;
        }
        // Broadcasting the value of the current unknown
        MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);
        // Updating the values of the vector b
        for (int j = 0; j < rowNum; j++)
            if (pProcPivotIter[j] < i)
            {
                val = pProcRows[j * size + i] * iterResult;
                pProcVector[j] = pProcVector[j] - val;
            }
    }
}
// Function for testing the data distribution
void TestDistribution(double *pMatrix, double *pVector, double *pProcRows,
                      double *pProcVector, int size, int rowNum)
{
    if (procRank == 0)
    {
        printf("Initial Matrix: \n");
        PrintMatrix(pMatrix, size, size);
        printf("Initial Vector: \n");
        PrintVector(pVector, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < procNum; i++)
    {
        if (procRank == i)
        {
            printf("\nProcRank = %d \n", procRank);
            printf(" Matrix Stripe:\n");
            PrintMatrix(pProcRows, rowNum, size);
            printf(" Vector: \n");
            PrintVector(pProcVector, rowNum);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
// Function for the execution of the parallel Gauss algorithm
void ParallelResultCalculation(double *pProcRows, double *pProcVector,
                               double *pProcResult, int size, int rowNum)
{
    ParallelGaussianElimination(pProcRows, pProcVector, size, rowNum);
    ParallelBackSubstitution(pProcRows, pProcVector, pProcResult,
                             size, rowNum);
}
// Function for computational process termination
void ProcessTermination(double *pMatrix, double *pVector, double *pResult,
                        double *pProcRows, double *pProcVector, double *pProcResult)
{
    if (procRank == 0)
    {
        free(pMatrix);
        free(pVector);
        free(pResult);
    }
    free(pProcRows);
    free(pProcVector);
    free(pProcResult);
    free(pParallelPivotPos);
    free(pProcPivotIter);
    free(pProcInd);
    free(pProcNum);
}
void TestResult(double *pMatrix, double *pVector, double *pResult,
                int size)
{
    /* Buffer for storing the vector, that is a result of multiplication 
 of the linear system matrix by the vector of unknowns */
    double *pRightPartVector;
    // Flag, that shows wheather the right parts vectors are identical or not
    int equal = 0;
    double Accuracy = 1.e-6; // Comparison accuracy
    if (procRank == 0)
    {
        pRightPartVector = malloc(sizeof(double)* size);
        for (int i = 0; i < size; i++)
        {
            pRightPartVector[i] = 0;
            for (int j = 0; j < size; j++)
            {
                pRightPartVector[i] +=
                    pMatrix[i * size + j] * pResult[pParallelPivotPos[j]];
            }
        }
        for (int i = 0; i < size; i++)
        {
            if (fabs(pRightPartVector[i] - pVector[i]) > Accuracy) 
                equal = 1;
        }
        if (equal == 1)
            printf("The result of the parallel Gauss algorithm is NOT correct."
                   "Check your code.");
        else
            printf("The result of the parallel Gauss algorithm is correct.");
        free( pRightPartVector);
    }
}
void main(int argc, char *argv[])
{
    double *pMatrix;     // Matrix of the linear system
    double *pVector;     // Right parts of the linear system
    double *pResult;     // Result vector
    double *pProcRows;   // Rows of the matrix A
    double *pProcVector; // Elements of the vector b
    double *pProcResult; // Elements of the vector x
    int size;            // sizes of the matrix and the vectors
    int rowNum;          // Number of the matrix rows
    double start, finish, duration;
    setvbuf(stdout, 0, _IONBF, 0);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    
    if (procRank == 0)
        printf("Parallel Gauss algorithm for solving linear systems\n");
    // Memory allocation and data initialization
    ProcessInitialization(&pMatrix, &pVector, &pResult,
                          &pProcRows, &pProcVector, &pProcResult, &size, &rowNum);
    //PrintMatrix(pMatrix,size, size);
    // The execution of the parallel Gauss algorithm
    start = MPI_Wtime();
    DataDistribution(pMatrix, pProcRows, pVector, pProcVector, size, rowNum);

    ParallelResultCalculation(pProcRows, pProcVector, pProcResult,
                            size, rowNum);
    TestDistribution(pMatrix, pVector, pProcRows, pProcVector, size, rowNum);
    ResultCollection(pProcResult, pResult);

    finish = MPI_Wtime();
    duration = finish - start;
    if (procRank == 0)
    {
        // Printing the result vector
        printf("\n Result Vector: \n");
        PrintVector(pResult, size);
    }
    TestResult(pMatrix, pVector, pResult, size);
    // Printing the time spent by the Gauss algorithm
    if (procRank == 0)
        printf("\n Time of execution: %f\n", duration);
    // Computational process termination
    ProcessTermination(pMatrix, pVector, pResult, pProcRows, pProcVector,
                       pProcResult);
    MPI_Finalize();
}