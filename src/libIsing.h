

typedef struct _ising_info ising_t;

// 
void runIsing(char *IsingGrid, int *IsingNtk, double T, int tmax, int seed, double orders[], char fname[]);
void runIsing_w_pthread(char *IsingGrid, int *IsingNtk, double T, int tmax, int seed, double orders[], char fname[]);
void *run_sub(void *arg);
//
void initIsingNtk(int *ntk);
//