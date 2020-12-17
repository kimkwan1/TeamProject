#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "mt64.h"
#include "libIsing.h"

extern double kb, J;
extern int GRID_SIZE;

void print_elapsed(struct timeval start_t, struct timeval end_t);

int main(){
    char *IsingGrid, fname[128];
    int *IsingNtk, n, i, nmax;
    double orders[3], T, Tmax, dT, tmax;
    struct timeval start_t, end_t;
    FILE *fid;

    // set initial parameters
    kb = 1; J = 1;
    GRID_SIZE = 10000;
    tmax = 10000;
    Tmax = 5;
    dT = 0.01;
    nmax = 50;
    
    IsingNtk = (int*) malloc(sizeof(int) * GRID_SIZE*4);
    IsingGrid = (char*) malloc(sizeof(char) * GRID_SIZE);

    initIsingNtk(IsingNtk);
    fid = fopen("../data/orders.csv", "w");
    fprintf(fid, "T,m,sus,bin_cum\n");

    // run model
    for (n=0; n<nmax; n++){
        // init Ising Grid
        for (i=0; i<GRID_SIZE; i++) { IsingGrid[i] = -1; }
        // run
        for (T=0; T<=Tmax; T+=dT){
            sprintf(fname, "../data/spins_%.2f_%d.csv", T, n);
            gettimeofday(&start_t, NULL);
            runIsing(IsingGrid, IsingNtk, T, tmax, time(NULL), orders, fname);
            gettimeofday(&end_t, NULL);
            printf("itr=%2d, T=%5.2f, m=%5.4f Done  ", n, T, orders[0]);
            print_elapsed(start_t, end_t);

            fprintf(fid, "%f,%f,%f,%f\n", T, orders[0], orders[1], orders[2]);
        }
    }

    fclose(fid);
    free(IsingNtk);
    free(IsingGrid);
}

void print_elapsed(struct timeval start_t, struct timeval end_t){
    int sec, msec, usec, x;

    sec = end_t.tv_sec - start_t.tv_sec;
    usec = end_t.tv_usec - start_t.tv_usec;
    x = usec / 1e3;
    msec = x;
    usec -= x * 1e3;

    if (usec < 0){
        msec -= 1;
        usec += 1e3;
    }
    if (msec < 0){
        sec -= 1;
        msec += 1e3;
    }

    printf("elapsed time = %ds %dms %dus\n", sec, msec, usec);
}