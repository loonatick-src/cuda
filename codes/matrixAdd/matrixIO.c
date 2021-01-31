void printMat(float *a, int n) {
    int i, j;
    for (i = 0; i < n; i ++) {
        for (j = 0; j < n; j++) {
            printf("%f ", a[n*i + j]);
        }
        putchar('\n');
    }
}

void readMat(float *a, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            scanf("%f", a + n*i + j);
        }
    }
}
