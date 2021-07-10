#include <mpi.h>
#include <stdio.h>

// Program 9.1
// The Gauss Algorithm for solving the systems of linear equations
int ProcNum;            // Number of the available processes
int ProcRank;           // Rank of the current process
int *pParallelPivotPos; // Number of rows selected as the pivot ones
int *pProcPivotIter;    // Number of iterations, at which the processor
                        // rows were used as the pivot ones
int *pProcInd; // Number of the first row located on the processes
int *pProcNum; // Number of the linear system rows located on the processes

void main(int argc, char *argv[])
{
    double *pMatrix;     // Matrix of the linear system
    double *pVector;     // Right parts of the linear system
    double *pResult;     // Result vector
    double *pProcRows;   // Rows of the matrix A
    double *pProcVector; // Block of the vector b
    double *pProcResult; // Block of the vector x
    int Size;            // Size of the matrix and vectors
    int RowNum;          // Number of the matrix rows

    setvbuf(stdout, 0, _IONBF, 0);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    if (ProcRank == 0)
        printf("Parallel Gauss algorithm for solving linear systems\n");

    // Memory allocation and data initialization
    ProcessInitialization(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult, Size, RowNum);

    // The execution of the parallel Gauss algorithm
    DataDistribution(pMatrix, pProcRows, pVector, pProcVector, Size, RowNum);
    ParallelResultCalculation(pProcRows, pProcVector, pProcResult, Size, RowNum);
    ResultCollection(pProcResult, pResult);
    TestResult(pMatrix, pVector, pResult, Size);
    
    // Computational process termination
    ProcessTermination(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult);
    MPI_Finalize();
}

// Function for execution of the parallel Gauss algorithm
void ParallelResultCalculation(double* pProcRows, double* pProcVector, double* pProcResult, int Size, int RowNum)
{
    // Gaussian elimination
    ParallelGaussianElimination(pProcRows, pProcVector, Size, RowNum);
    // Back substitution
    ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, Size, RowNum);
}

void ParallelGaussianElimination(double *pProcRows, double *pProcVector, int Size, int RowNum)
{
    double MaxValue; // Value of the pivot element of thе process
    int PivotPos;    // Position of the pivot row in the process stripe
    // Structure for the pivot row selection
    struct
    {
        double MaxValue;
        int ProcRank;
    } ProcPivot, Pivot;

    // pPivotRow is used for storing the pivot row and the corresponding
    // element of the vector b
    double pPivotRow[Size + 1];
    // The iterations of the Gaussian elimination stage
    for (int i = 0; i < Size; i++)
    {
        // Calculating the local pivot row
        double MaxValue = 0;
        for (int j = 0; j < RowNum; j++)
        {
            if ((pProcPivotIter[j] == -1) &&
                (MaxValue < fabs(pProcRows[j * Size + i])))
            {
                MaxValue = fabs(pProcRows[j * Size + i]);
                PivotPos = j;
            }
        }
        ProcPivot.MaxValue = MaxValue;
        ProcPivot.ProcRank = ProcRank;
        // Finding the pivot process (process with the maximum value of MaxValue)
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // Broadcasting the pivot row
        if (ProcRank == Pivot.ProcRank)
        {
            pProcPivotIter[PivotPos] = i; //iteration number
            pParallelPivotPos[i] = pProcInd[ProcRank] + PivotPos;
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank, MPI_COMM_WORLD);
        if (ProcRank == Pivot.ProcRank)
        {
            // Fill the pivot row
            for (int j = 0; j < Size; j++)
            {
                pPivotRow[j] = pProcRows[PivotPos * Size + j];
            }
            pPivotRow[Size] = pProcVector[PivotPos];
        }
        MPI_Bcast(pPivotRow, Size + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);
        ParallelEliminateColumns(pProcRows, pProcVector, pPivotRow, Size, RowNum, i);
    }
}

void ParallelBackSubstitution(double *pProcRows, double *pProcVector,  double *pProcResult, int Size, int RowNum)
{
    int IterProcRank; // Rank of the process with the current pivot row
    int IterPivotPos; // Position of the pivot row of the process
    double IterResult; // Calculated value of the current unknown
    double val; // Iterations of the back substitution stage
    for (int i = Size - 1; i >= 0; i--)
    {
        // Calculating the rank of the process, which holds the pivot row
        FindBackPivotRow(pParallelPivotPos[i], Size, IterProcRank, IterPivotPos);
        // Calculating the unknown
        if (ProcRank == IterProcRank)
        {
            IterResult = pProcVector[IterPivotPos] / pProcRows[IterPivotPos * Size + i];
            pProcResult[IterPivotPos] = IterResult;
        }
        // Broadcasting the value of the current unknown
        MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank, MPI_COMM_WORLD);
        // Updating the values of the vector b
        for (int j = 0; j < RowNum; j++)
            if (pProcPivotIter[j] < i)
            {
                val = pProcRows[j * Size + i] * IterResult;
                pProcVector[j] = pProcVector[j] - val;
            }
    }
}