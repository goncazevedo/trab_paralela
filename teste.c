#include <math.h>
#include <mpi.h>
#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int nProcDisponiveis;    // Números de processos disponíveis
int procRank;            // Rank do processo atual
int *pParallelPivotPos;  // Número de linhas selecioandas como pivô
int *pProcPivotIter;  // Numero de iterações das linhas de cada processo que foi
                      // passada como pivô
int *pProcInd;  // Número da primeiro elemento da linha da matriz no processo
int *pProcNum;  // Número de linhas do sistema localizadas no processo

// Printa a matriz formatada
void PrintMatrix(double *matriz, int nLinhas, int nColunas) {
    int i, j;
    for (i = 0; i < nLinhas; i++) {
        for (j = 0; j < nColunas; j++)
            printf(" %7.2f ", matriz[i * nColunas + j]);
        printf("\n");
    }
}
// Printa o sistema linear
void PrintSistema(double *matriz, double *vetorB, int nLinhas, int nColunas) {
    int i, j;
    for (i = 0; i < nLinhas; i++) {
        for (j = 0; j < nColunas; j++)
            printf("%7.2f(x%d) ", matriz[i * nColunas + j], j);
        printf("= %7.2f\n", vetorB[i]);
    }
}
// Popula o sistema aleatoriamente
void RandomDataInitialization(double *matriz, double *vetorB, int tamanho) {
    int i, j;
    for (i = 0; i < tamanho; i++) {
        vetorB[i] = rand() % 10;
        for (j = 0; j < tamanho; j++) {
            if (j <= i)
                matriz[i * tamanho + j] = rand() % 10;
            else
                matriz[i * tamanho + j] = 0;
        }
    }
}
void matriz4(double *matriz, double *vetorB) {
    double m[] = {3.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 2.0, 3.0};
    matriz = m;
    double v[] = {9.67,8.99,8.81,8.25};
    vetorB = v;
}

void matriz7(double *matriz, double *vetorB) {
    double m[] = {3.94,0.0,0.0,0.0,0.0,0.0,0.0,4.88,9.47,0.0,0.0,0.0,0.0,0.0,9.08,2.26,1.86,0.0,0.0,0.0,0.0,6.71,6.96,4.72,2.95,0.0,0.0,0.0,5.55,9.50,4.82,8.46,6.29,0.0,0.0,1.82,8.27,1.33,7.17,9.32,8.82,0.0,8.60,3.30,5.79,9.30,1.91,6.12,7.33};
    matriz = m;
    double v[] = {1.42,5.36,3.57,8.90,1.40,7.17,1.37};
    vetorB = v;
}


void matriz12(double *matriz, double *vetorB) {
    double m[] = {4.15,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.95,9.08,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.98,9.12,9.65,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.76,3.63,9.20,3.83,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.08,6.07,1.65,8.48,2.38,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.83,2.39,1.63,4.18,1.26,3.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.04,3.02,9.55,3.10,8.55,2.29,1.69,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.52,9.13,5.40,1.06,1.11,1.66,4.69,6.39,0.0,0.0,0.0,0.0,0.0,0.0,1.76,7.51,2.10,4.72,1.67,6.09,9.63,9.98,4.82,0.0,0.0,0.0,0.0,0.0,2.40,6.37,9.01,8.75,5.82,5.83,9.96,9.63,4.76,6.66,0.0,0.0,0.0,0.0,3.54,4.65,7.95,4.47,9.57,6.85,3.32,6.38,1.23,3.60,1.44,0.0,0.0,0.0,7.58,4.86,6.30,6.79,9.68,4.86,9.96,5.94,8.69,5.23,1.35,2.30,0.0,0.0,8.23,4.78,9.80,5.47,7.67,5.21,9.98,9.91,9.14,5.17,6.84,2.57,3.38,0.0,1.99,5.49,9.80,3.30,5.03,7.28,7.54,3.12,7.24,8.02,9.27,7.11,3.00,7.53};
    matriz = m;
    double v[] = {5.13,4.90,6.55,8.80,1.25,7.39,4.70,6.42,6.95,2.47,2.81,3.66,8.93,5.62};
    vetorB = v;
}


// Aloca a memória e inicializa os dados
void ProcessInitialization(double **matriz, double **vetorB,
                           double **vetorResultado, double **linhaDaMatriz,
                           double **vetorBNoProcesso,
                           double **vetorResultadoNoProcesso, int *tamanho,
                           int *linhasDaMatrizNoProcesso) {
    int restRows;  // Numero de linhas que não foram distribuídas
    int i;
    if (procRank == 0) {
        do {
            printf("\nDigite o numero de variaveis do sistema: ");
            scanf("%d", tamanho);
            if (*tamanho < nProcDisponiveis) {
                printf(
                    "O tamanho da matriz deve ser maior que o número de "
                    "processos! \n");
            }
        } while (*tamanho < nProcDisponiveis);
    }
    MPI_Bcast(tamanho, 1, MPI_INT, 0, MPI_COMM_WORLD);
    restRows = *tamanho;
    for (i = 0; i < procRank; i++)
        restRows = restRows - restRows / (nProcDisponiveis - i);
    *linhasDaMatrizNoProcesso = restRows / (nProcDisponiveis - procRank);
    *linhaDaMatriz =
        malloc(sizeof(double) * (*linhasDaMatrizNoProcesso * (*tamanho)));
    *vetorBNoProcesso = malloc(sizeof(double) * (*linhasDaMatrizNoProcesso));
    *vetorResultadoNoProcesso =
        malloc(sizeof(double) * (*linhasDaMatrizNoProcesso));

    pParallelPivotPos = malloc(sizeof(int) * (*tamanho));
    pProcPivotIter = malloc(sizeof(int) * (*linhasDaMatrizNoProcesso));
    pProcInd = malloc(sizeof(int) * nProcDisponiveis);
    pProcNum = malloc(sizeof(int) * nProcDisponiveis);

    for (int i = 0; i < *linhasDaMatrizNoProcesso; i++) pProcPivotIter[i] = -1;

    if (procRank == 0) {
        *matriz = malloc(sizeof(double) * (*tamanho) * (*tamanho));
        *vetorB = malloc(sizeof(double) * *tamanho);
        *vetorResultado = malloc(sizeof(double) * *tamanho);
        RandomDataInitialization(*matriz, *vetorB, *tamanho);
    }
}
// Distribuição de dados
void DataDistribution(double *matriz, double *linhaDaMatriz, double *vetorB,
                      double *vetorBNoProcesso, int tamanho,
                      int linhasDaMatrizNoProcesso) {
    int *pSendNum;  // Número de elementos enviados ao processo
    int *pSendInd;  // Indice do primeiro elemento enviado para o processo
    int restRows =
        tamanho;  // Número de linhas que ainda não foram distribuidas
    int i;        // Variável do looop
    // Alocação de memória para variáveis temporárias
    pSendInd = malloc(sizeof(int) * nProcDisponiveis);
    pSendNum = malloc(sizeof(int) * nProcDisponiveis);
    // Define a disposição de linhas para o processo atual
    linhasDaMatrizNoProcesso = (tamanho / nProcDisponiveis);
    pSendNum[0] = linhasDaMatrizNoProcesso * tamanho;
    pSendInd[0] = 0;
    for (i = 1; i < nProcDisponiveis; i++) {
        restRows -= linhasDaMatrizNoProcesso;
        linhasDaMatrizNoProcesso = restRows / (nProcDisponiveis - i);
        pSendNum[i] = linhasDaMatrizNoProcesso * tamanho;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }
    // Envia os conjuntos de dados para os processos disponíveis
    MPI_Scatterv(matriz, pSendNum, pSendInd, MPI_DOUBLE, linhaDaMatriz,
                 pSendNum[procRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Define a disposição de linhas para o processo atual
    restRows = tamanho;
    pProcInd[0] = 0;
    pProcNum[0] = tamanho / nProcDisponiveis;
    for (i = 1; i < nProcDisponiveis; i++) {
        restRows -= pProcNum[i - 1];
        pProcNum[i] = restRows / (nProcDisponiveis - i);
        pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
    }
    MPI_Scatterv(vetorB, pProcNum, pProcInd, MPI_DOUBLE, vetorBNoProcesso,
                 pProcNum[procRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Libera memória
    free(pSendNum);
    free(pSendInd);
}
// Realiza o Gathering(recebimento) do vetor de resultado de todos os processos
void ResultCollection(double *vetorResultadoNoProcesso,
                      double *vetorResultado) {
    MPI_Gatherv(vetorResultadoNoProcesso, pProcNum[procRank], MPI_DOUBLE,
                vetorResultado, pProcNum, pProcInd, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
}
// Printa o vetor das variáveis formatado
void PrintResultado(double *vetorB, int tamanho) {
    int i;
    for (i = 0; i < tamanho; i++) printf("x%d = %7.2f\n", i, vetorB[i]);
}
// Printa o vetor formatado
void PrintVector(double *vetorB, int tamanho) {
    int i;
    for (i = 0; i < tamanho; i++) printf("%7.2f ", vetorB[i]);
}
// Printa o vetor resultado formatado
void PrintResultVector(double *vetorResultado, int tamanho) {
    int i;
    for (i = 0; i < tamanho; i++)
        printf("x%d = %7.2f\n", i, vetorResultado[pParallelPivotPos[i]]);
}
// Eliminação de colunas paralela
void ParallelEliminateColumns(double *linhaDaMatriz, double *vetorBNoProcesso,
                              double *pPivotRow, int tamanho,
                              int linhasDaMatrizNoProcesso, int iter) {
    double multiplier;
    for (int i = 0; i < linhasDaMatrizNoProcesso; i++) {
        if (pProcPivotIter[i] == -1) {
            multiplier = linhaDaMatriz[i * tamanho + iter] / pPivotRow[iter];
            for (int j = iter; j < tamanho; j++) {
                linhaDaMatriz[i * tamanho + j] -= pPivotRow[j] * multiplier;
            }
            vetorBNoProcesso[i] -= pPivotRow[tamanho] * multiplier;
        }
    }
}
// Eliminação de gauss paralela
void ParallelGaussianElimination(double *linhaDaMatriz,
                                 double *vetorBNoProcesso, int tamanho,
                                 int linhasDaMatrizNoProcesso) {
    double maxValue;  // valor do pivo nesse processo
    int pivotPos;     // posição da linha do pivo nesse processo

    // Estrutura para seleção de pivô
    struct {
        double maxValue;
        int procRank;
    } ProcPivot, Pivot;

    // Aramazena o pivô da linha atual e o elemento correspondente do vetor b
    double *pPivotRow = malloc(sizeof(double) * (tamanho + 1));

    // Iterações para a eliminaçãoi de gauss
    for (int i = 0; i < tamanho; i++) {
        // Caldulando o pivot local da linha
        double maxValue = 0;
        for (int j = 0; j < linhasDaMatrizNoProcesso; j++) {
            if ((pProcPivotIter[j] == -1) &&
                (maxValue < fabs(linhaDaMatriz[j * tamanho + i]))) {
                maxValue = fabs(linhaDaMatriz[j * tamanho + i]);
                pivotPos = j;
            }
        }
        ProcPivot.maxValue = maxValue;
        ProcPivot.procRank = procRank;

        // Encontrando o processo com o maior pivot
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        // Transmitindo a linha do maior pivo
        if (procRank == Pivot.procRank) {
            pProcPivotIter[pivotPos] = i;  // número da iteração
            pParallelPivotPos[i] = pProcInd[procRank] + pivotPos;

            // Preenche a linha do pívô
            for (int j = 0; j < tamanho; j++) {
                pPivotRow[j] = linhaDaMatriz[pivotPos * tamanho + j];
            }
            pPivotRow[tamanho] = vetorBNoProcesso[pivotPos];
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.procRank,
                  MPI_COMM_WORLD);

        MPI_Bcast(pPivotRow, tamanho + 1, MPI_DOUBLE, Pivot.procRank,
                  MPI_COMM_WORLD);

        ParallelEliminateColumns(linhaDaMatriz, vetorBNoProcesso, pPivotRow,
                                 tamanho, linhasDaMatrizNoProcesso, i);
    }
}
// Achando a linha do pivô par a substituição
void FindBackPivotRow(int rowIndex, int *iterProcRank, int *iterPivotPos) {
    for (int i = 0; i < nProcDisponiveis - 1; i++) {
        if ((pProcInd[i] <= rowIndex) && (rowIndex < pProcInd[i + 1]))
            *iterProcRank = i;
    }
    if (rowIndex >= pProcInd[nProcDisponiveis - 1])
        *iterProcRank = nProcDisponiveis - 1;
    *iterPivotPos = rowIndex - pProcInd[*iterProcRank];
}
// Realiza a substituição no sistema
void ParallelBackSubstitution(double *linhaDaMatriz, double *vetorBNoProcesso,
                              double *vetorResultadoNoProcesso, int tamanho,
                              int linhasDaMatrizNoProcesso) {
    int iterProcRank;   // Rank do processo com a linha de pivô atual
    int iterPivotPos;   // Posição da linha do pivô do processo
    double iterResult;  // valor da variável(?)
    double val;
    // Iterações da substituição
    for (int i = tamanho - 1; i >= 0; i--) {
        // Calculando o rank do processo que possui a linha do pivô
        FindBackPivotRow(pParallelPivotPos[i], &iterProcRank, &iterPivotPos);

        // Calculando a variável
        if (procRank == iterProcRank) {
            iterResult = vetorBNoProcesso[iterPivotPos] /
                         linhaDaMatriz[iterPivotPos * tamanho + i];
            vetorResultadoNoProcesso[iterPivotPos] = iterResult;
        }
        // Transmitindo o valor da variável atual
        MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);
        // Atualizando o vetor de variáveis
        for (int j = 0; j < linhasDaMatrizNoProcesso; j++)
            if (pProcPivotIter[j] < i) {
                val = linhaDaMatriz[j * tamanho + i] * iterResult;
                vetorBNoProcesso[j] = vetorBNoProcesso[j] - val;
            }
    }
}
// Printando a distribuição de dados
void TestDistribution(double *matriz, double *vetorB, double *linhaDaMatriz,
                      double *vetorBNoProcesso, int tamanho,
                      int linhasDaMatrizNoProcesso) {
    if (procRank == 0) {
        printf("Sistema inicial: \n");
        PrintSistema(matriz, vetorB, tamanho, tamanho);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < nProcDisponiveis; i++) {
        if (procRank == i) {
            printf("\nProcRank = %d \n", procRank);
            printf(" Pedaço da matriz:\n");
            PrintMatrix(linhaDaMatriz, linhasDaMatrizNoProcesso, tamanho);
            printf(" Vetor:\n");
            PrintVector(vetorBNoProcesso, linhasDaMatrizNoProcesso);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
// Executa a eliminação de gauss paralelamente
void ParallelResultCalculation(double *linhaDaMatriz, double *vetorBNoProcesso,
                               double *vetorResultadoNoProcesso, int tamanho,
                               int linhasDaMatrizNoProcesso) {
    ParallelGaussianElimination(linhaDaMatriz, vetorBNoProcesso, tamanho,
                                linhasDaMatrizNoProcesso);
    ParallelBackSubstitution(linhaDaMatriz, vetorBNoProcesso,
                             vetorResultadoNoProcesso, tamanho,
                             linhasDaMatrizNoProcesso);
}
// Liberando a memória alocada
void ProcessTermination(double *matriz, double *vetorB, double *vetorResultado,
                        double *linhaDaMatriz, double *vetorBNoProcesso,
                        double *vetorResultadoNoProcesso) {
    if (procRank == 0) {
        free(matriz);
        free(vetorB);
        free(vetorResultado);
    }
    free(linhaDaMatriz);
    free(vetorBNoProcesso);
    free(vetorResultadoNoProcesso);
    free(pParallelPivotPos);
    free(pProcPivotIter);
    free(pProcInd);
    free(pProcNum);
}
void TestResult(double *matriz, double *vetorB, double *vetorResultado,
                int tamanho) {
    double *pRightPartVector;
    int equal = 0;
    double Accuracy = 1.e-6;  // Compara a precisão
    if (procRank == 0) {
        pRightPartVector = malloc(sizeof(double) * tamanho);
        for (int i = 0; i < tamanho; i++) {
            pRightPartVector[i] = 0;
            for (int j = 0; j < tamanho; j++) {
                pRightPartVector[i] += matriz[i * tamanho + j] *
                                       vetorResultado[pParallelPivotPos[j]];
            }
        }
        for (int i = 0; i < tamanho; i++) {
            if (fabs(pRightPartVector[i] - vetorB[i]) > Accuracy) equal = 1;
        }
        if (equal == 1)
            printf(
                "O resultado do algoritmo de eliminaçãod e gauss está "
                "errado!\n");
        else
            printf(
                "O resultado do algoritmo de eliminaçãod e gauss está "
                "correto!\n");
        free(pRightPartVector);
    }
}
void main(int argc, char *argv[]) {
    // Ax = b
    double *matriz;                    // Matriz do sistema linear
    double *vetorB;                    // Parte direita do sistema linear
    double *vetorResultado;            // Vetor de variáveis
    double *linhaDaMatriz;             // Linhas da matriz A
    double *vetorBNoProcesso;          // Elementos do vetor b
    double *vetorResultadoNoProcesso;  // Elementos do vetor de variáveis
    int tamanho;                       // tamanho da matriz e dos vetores
    int linhasDaMatrizNoProcesso;      // numero de linhas da matriz
    double inicio, fim, duracao;

    setvbuf(stdout, 0, _IONBF, 0);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcDisponiveis);

    if (procRank == 0)
        printf(
            "Algoritmo paralelo para resolver sistemas lineares com eliminação"
            "de Gauss\n");

    // Alocação de memória e inicialização de dados
    ProcessInitialization(&matriz, &vetorB, &vetorResultado, &linhaDaMatriz,
                          &vetorBNoProcesso, &vetorResultadoNoProcesso,
                          &tamanho, &linhasDaMatrizNoProcesso);

    inicio = MPI_Wtime();

    // Distribuição de dados nos processos
    DataDistribution(matriz, linhaDaMatriz, vetorB, vetorBNoProcesso, tamanho,
                     linhasDaMatrizNoProcesso);

    // Cálculo em paralelo
    ParallelResultCalculation(linhaDaMatriz, vetorBNoProcesso,
                              vetorResultadoNoProcesso, tamanho,
                              linhasDaMatrizNoProcesso);

    // Debuggar a distribuição da matriz entre os processos
    // TestDistribution(matriz, vetorB, linhaDaMatriz, vetorBNoProcesso,
    // tamanho, linhasDaMatrizNoProcesso);

    if (procRank == 0) {
        printf("Sistema inicial: \n");
        PrintSistema(matriz, vetorB, tamanho, tamanho);
    }

    // Recebendo o vetor de resultado final
    ResultCollection(vetorResultadoNoProcesso, vetorResultado);
    fim = MPI_Wtime();
    duracao = fim - inicio;

    // Esperando todos os processo finalizarem
    MPI_Barrier(MPI_COMM_WORLD);

    // Verificando a resposta obtida
    TestResult(matriz, vetorB, vetorResultado, tamanho);

    if (procRank == 0) {
        // Printando o resultado do sitema
        printf("\n Resultados: \n");
        PrintResultado(vetorResultado, tamanho);

        // Printando o tempo gasto pelo algoritmo
        printf("\n Tempo de execução: %f\n", duracao);
    }

    // Liberando a memória alocada
    ProcessTermination(matriz, vetorB, vetorResultado, linhaDaMatriz,
                       vetorBNoProcesso, vetorResultadoNoProcesso);

    // Finalizando o MPI
    MPI_Finalize();
}