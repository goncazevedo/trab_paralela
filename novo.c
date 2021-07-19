#include <math.h>
#include <mpi.h>
#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int nProcDisponiveis;    // Números de processos disponíveis
int procRank;            // Rank do processo atual
int *ordemPedacoTriangulado;  // Número de linhas selecioandas como pivô
int *pProcPivotIter;  // Numero de iterações das linhas de cada processo que foi
                      // passada como pivô
int *pProcInd;  // Número da primeiro elemento da linha da matriz no processo
int *pProcNum;  // Número de linhas do sistema localizadas no processo

void Inicializacao(double *matriz, double *vetorB, int tamanho) {
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
// Printa o sistema linear
void PrintaSistema(double *matriz, double *vetorB, int nLinhas, int nColunas) {
    int i, j;
    for (i = 0; i < nLinhas; i++) {
        for (j = 0; j < nColunas; j++)
            printf("%7.2f(x%d) ", matriz[i * nColunas + j], j);
        printf("= %7.2f\n", vetorB[i]);
    }
}
// Aloca a memória e inicializa os dados
void InicializaProcesso(double **matriz, double **vetorB,
                           double **vetorResultado, double **matrizNoProcesso,
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
                    "processos e deve ter um tamanho válido! \n");
            }
        } while ((*tamanho < nProcDisponiveis));
    }
    MPI_Bcast(tamanho, 1, MPI_INT, 0, MPI_COMM_WORLD);
    restRows = *tamanho;
    for (i = 0; i < procRank; i++)
        restRows = restRows - restRows / (nProcDisponiveis - i);
    *linhasDaMatrizNoProcesso = restRows / (nProcDisponiveis - procRank);
    *matrizNoProcesso =
        malloc(sizeof(double) * (*linhasDaMatrizNoProcesso * (*tamanho)));
    *vetorBNoProcesso = malloc(sizeof(double) * (*linhasDaMatrizNoProcesso));
    *vetorResultadoNoProcesso =
        malloc(sizeof(double) * (*linhasDaMatrizNoProcesso));

    ordemPedacoTriangulado = malloc(sizeof(int) * (*tamanho));
    pProcPivotIter = malloc(sizeof(int) * (*linhasDaMatrizNoProcesso));
    pProcInd = malloc(sizeof(int) * nProcDisponiveis);
    pProcNum = malloc(sizeof(int) * nProcDisponiveis);

    for (int i = 0; i < *linhasDaMatrizNoProcesso; i++) pProcPivotIter[i] = -1;

    if (procRank == 0) {
        *matriz = malloc(sizeof(double) * (*tamanho) * (*tamanho));
        *vetorB = malloc(sizeof(double) * *tamanho);
        *vetorResultado = malloc(sizeof(double) * *tamanho);
        Inicializacao(*matriz, *vetorB, *tamanho);
    }
}
// Distribuição de dados
void DistribuiDados(double *matriz, double *matrizNoProcesso, double *vetorB,
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
    MPI_Scatterv(matriz, pSendNum, pSendInd, MPI_DOUBLE, matrizNoProcesso,
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
void ColetaResultados(double *vetorResultadoNoProcesso,
                      double *vetorResultado) {
    MPI_Gatherv(vetorResultadoNoProcesso, pProcNum[procRank], MPI_DOUBLE,
                vetorResultado, pProcNum, pProcInd, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
}
// Printa o vetor das variáveis formatado
void PrintaResultado(double *vetorB, int tamanho) {
    int i;
    for (i = 0; i < tamanho; i++) printf("x%d = %7.2f\n", i, vetorB[ordemPedacoTriangulado[i]]);
}
// Printa o vetor formatado
void PrintaVetor(double *vetorB, int tamanho) {
    int i;
    for (i = 0; i < tamanho; i++) printf("%7.2f ", vetorB[i]);
}
// Printa o vetor formatado
void PrintaP(int *vetor, int tamanho) {
    int i;
    for (i = 0; i < tamanho; i++) printf("%d ", vetor[i]);
}
// Eliminação de colunas paralela
void EliminaColunasParalelamente(double *matrizNoProcesso, double *vetorBNoProcesso,
                              double *pPivotRow, int tamanho,
                              int linhasDaMatrizNoProcesso, int iter) {
    double multiplier;
    for (int i = 0; i < linhasDaMatrizNoProcesso; i++) {
        if (pProcPivotIter[i] == -1) {
            multiplier = matrizNoProcesso[i * tamanho + iter] / pPivotRow[iter];
            for (int j = iter; j < tamanho; j++) {
                matrizNoProcesso[i * tamanho + j] -= pPivotRow[j] * multiplier;
            }
            vetorBNoProcesso[i] -= pPivotRow[tamanho] * multiplier;
        }
    }
}
// Eliminação de gauss paralela
void EliminacaoDeGauss(double *matrizNoProcesso,
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
                (maxValue < fabs(matrizNoProcesso[j * tamanho + i]))) {
                maxValue = fabs(matrizNoProcesso[j * tamanho + i]);
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
            ordemPedacoTriangulado[i] = pProcInd[procRank] + pivotPos;

            // Preenche a linha do pívô
            for (int j = 0; j < tamanho; j++) {
                pPivotRow[j] = matrizNoProcesso[pivotPos * tamanho + j];
            }
            pPivotRow[tamanho] = vetorBNoProcesso[pivotPos];
        }
        MPI_Bcast(&ordemPedacoTriangulado[i], 1, MPI_INT, Pivot.procRank,
                  MPI_COMM_WORLD);

        MPI_Bcast(pPivotRow, tamanho + 1, MPI_DOUBLE, Pivot.procRank,
                  MPI_COMM_WORLD);

        EliminaColunasParalelamente(matrizNoProcesso, vetorBNoProcesso, pPivotRow,
                                 tamanho, linhasDaMatrizNoProcesso, i);
    }
}
// Achando a linha do pivô par a substituição
void AchaLinhaDoPivoAnterior(int rowIndex, int *iterProcRank, int *iterPivotPos) {
    for (int i = 0; i < nProcDisponiveis - 1; i++) {
        if ((pProcInd[i] <= rowIndex) && (rowIndex < pProcInd[i + 1]))
            *iterProcRank = i;
    }
    if (rowIndex >= pProcInd[nProcDisponiveis - 1])
        *iterProcRank = nProcDisponiveis - 1;
    *iterPivotPos = rowIndex - pProcInd[*iterProcRank];
}
// Realiza a substituição no sistema
void SubstituicaoParalela(double *matrizNoProcesso, double *vetorBNoProcesso,
                              double *vetorResultadoNoProcesso, int tamanho,
                              int linhasDaMatrizNoProcesso) {
    int iterProcRank;   // Rank do processo com a linha de pivô atual
    int iterPivotPos;   // Posição da linha do pivô do processo
    double iterResult;  // valor da variável(?)
    double val;
    // Iterações da substituição
    for (int i = tamanho - 1; i >= 0; i--) {
        // Calculando o rank do processo que possui a linha do pivô
        AchaLinhaDoPivoAnterior(ordemPedacoTriangulado[i], &iterProcRank, &iterPivotPos);

        // Calculando a variável
        if (procRank == iterProcRank) {
            iterResult = vetorBNoProcesso[iterPivotPos] /
                         matrizNoProcesso[iterPivotPos * tamanho + i];
            vetorResultadoNoProcesso[iterPivotPos] = iterResult;
        }
        // Transmitindo o valor da variável atual
        MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);
        // Atualizando o vetor de variáveis
        for (int j = 0; j < linhasDaMatrizNoProcesso; j++)
            if (pProcPivotIter[j] < i) {
                val = matrizNoProcesso[j * tamanho + i] * iterResult;
                vetorBNoProcesso[j] = vetorBNoProcesso[j] - val;
            }
    }
}
// Executa a eliminação de gauss paralelamente
void CalculaResultado(double *matrizNoProcesso, double *vetorBNoProcesso,
                               double *vetorResultadoNoProcesso, int tamanho,
                               int linhasDaMatrizNoProcesso) {
    EliminacaoDeGauss(matrizNoProcesso, vetorBNoProcesso, tamanho,
                                linhasDaMatrizNoProcesso);
    SubstituicaoParalela(matrizNoProcesso, vetorBNoProcesso,
                             vetorResultadoNoProcesso, tamanho,
                             linhasDaMatrizNoProcesso);
}
// Liberando a memória alocada
void LiberaMemoria(double *matriz, double *vetorB, double *vetorResultado,
                        double *matrizNoProcesso, double *vetorBNoProcesso,
                        double *vetorResultadoNoProcesso) {
    if (procRank == 0) {
        free(matriz);
        free(vetorB);
        free(vetorResultado);
    }
    free(matrizNoProcesso);
    free(vetorBNoProcesso);
    free(vetorResultadoNoProcesso);
    free(ordemPedacoTriangulado);
    free(pProcPivotIter);
    free(pProcInd);
    free(pProcNum);
}
void TestaResultados(double *matriz, double *vetorB, double *vetorResultado,
                int tamanho) {
    double *vetorDireito;
    int equal = 0;
    double Accuracy = 1.e-6;  // Compara a precisão
    if (procRank == 0) {
        vetorDireito = malloc(sizeof(double) * tamanho);
        for (int i = 0; i < tamanho; i++) {
            vetorDireito[i] = 0;
            for (int j = 0; j < tamanho; j++) {
                vetorDireito[i] += matriz[i * tamanho + j] *
                                       vetorResultado[ordemPedacoTriangulado[j]];
            }
        }
        for (int i = 0; i < tamanho; i++) {
            if (fabs(vetorDireito[i] - vetorB[i]) > Accuracy) equal = 1;
        }
        if (equal == 1)
            printf(
                "O resultado do algoritmo de eliminação de gauss está "
                "errado!\n");
        else
            printf(
                "O resultado do algoritmo de eliminação de gauss está "
                "correto!\n");
        free(vetorDireito);
    }
}
void main(int argc, char *argv[]) {
    // Ax = b
    double *matriz;                    // Matriz do sistema linear
    double *vetorB;                    // Parte direita do sistema linear
    double *vetorResultado;            // Vetor de variáveis
    double *matrizNoProcesso;             // Linhas da matriz A
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
    InicializaProcesso(&matriz, &vetorB, &vetorResultado, &matrizNoProcesso,
                          &vetorBNoProcesso, &vetorResultadoNoProcesso,
                          &tamanho, &linhasDaMatrizNoProcesso);

    inicio = MPI_Wtime();

    // Distribuição de dados nos processos
    DistribuiDados(matriz, matrizNoProcesso, vetorB, vetorBNoProcesso, tamanho,
                     linhasDaMatrizNoProcesso);

    // Cálculo em paralelo
    CalculaResultado(matrizNoProcesso, vetorBNoProcesso,
                              vetorResultadoNoProcesso, tamanho,
                              linhasDaMatrizNoProcesso);

    if (procRank == 0) {
      printf("Sistema inicial: \n");
      PrintaSistema(matriz, vetorB, tamanho, tamanho);
    }
    // Recebendo o vetor de resultado final
    ColetaResultados(vetorResultadoNoProcesso, vetorResultado);
    fim = MPI_Wtime();
    duracao = fim - inicio;

    // Esperando todos os processo finalizarem
    MPI_Barrier(MPI_COMM_WORLD);

    // Verificando a resposta obtida
    TestaResultados(matriz, vetorB, vetorResultado, tamanho);

    if (procRank == 0) {
        // Printando o resultado do sitema
        printf("\n Resultados: \n");
        PrintaResultado(vetorResultado, tamanho);

        // Printando o tempo gasto pelo algoritmo
        printf("\n Tempo de execução: %f\n", duracao);
    }

    // Liberando a memória alocada
    LiberaMemoria(matriz, vetorB, vetorResultado, matrizNoProcesso,
                       vetorBNoProcesso, vetorResultadoNoProcesso);

    // Finalizando o MPI
    MPI_Finalize();
}