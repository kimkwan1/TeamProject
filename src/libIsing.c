#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include "libIsing.h"
#include "mt64.h"
// #include "../../libs/network_analysis/mt64.c"
// #include "/home/jung/Projects/lib/mt64.c"

#define NTHREAD 8
int TSTEADY = 1000;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int GRID_SIZE;
double kb=1, J=1;

struct _ising_info{
    int pid, id_start, id_end, *IsingNtk, tmax;
    char *IsingGrid;
    double *pdE;
    // save params
    double *m;
    FILE *fid;
};

void runIsing(char *IsingGrid, int *IsingNtk, double T, int tmax, int seed, double orders[], char fname[]){

    int i,j,t,n,s;
    double m=0,m2=0,m4=0,mt,sus,bin_cum;
    double p,pdE[2],dE;
    FILE *fid;

    if (fname == NULL) { fid = NULL; } 
    else { fid = fopen(fname, "w"); }

    for (i=0; i<2; i++){
        pdE[i] = exp(-J/(kb*T)*4*(i+1));
    }

    init_genrand64(seed);

    // run
    for (t=0; t<tmax; t++){
        for (i=0; i<GRID_SIZE; i++){
            n = (int) (genrand64_real2() * GRID_SIZE);

            s=0;
            for (j=0; j<4; j++){
                s += IsingGrid[IsingNtk[4*n+j]];
            }
            s *= IsingGrid[n];
            dE = 2*J*s;

            if (dE <= 0){
                IsingGrid[n] *= -1;
            } else { // s = 2, 4
                p = genrand64_real2();
                
                if (p < pdE[s/2-1]){
                    IsingGrid[n] *= -1;
                }
            }
        }

        // if t>tsteady, calculate order parameter
        if (t >= tmax-TSTEADY){
            mt = 0;
            for (n=0; n<GRID_SIZE; n++){
                mt += IsingGrid[n];
            }
            mt /= (double) GRID_SIZE;
            m += fabs(mt);
            m2 += mt*mt;
            m4 += mt*mt*mt*mt;

            if (fid != NULL){
                for (i=0; i<GRID_SIZE; i++){
                    fprintf(fid, "%d,", IsingGrid[i]);
                }
                fprintf(fid, "\n");
            }
        }
    }

    m /= TSTEADY;
    m2 /= TSTEADY; // m^2
    m4 /= TSTEADY; // m^4

    sus = GRID_SIZE * (m2 - m*m);
    bin_cum = 1 - (m4 / (3*m2*m2));

    orders[0] = m; // magnetization
    orders[1] = sus; // susceptability
    orders[2] = bin_cum; // binder cumulation

    if (fid != NULL){
        fclose(fid);
    }

    return;

}


void runIsing_w_pthread(char *IsingGrid, int *IsingNtk, double T, int tmax, int seed, double orders[], char fname[]){

    int i;
    double mall=0, m2=0, m4=0, *m, sus, bin_cum, pdE[2];
    ising_t infos[NTHREAD];
    pthread_t pids[NTHREAD];
    FILE *fid;

    if (fname == NULL) { fid = NULL; } 
    else { fid = fopen(fname, "w"); }

    m = (double*) malloc(sizeof(double) * TSTEADY);
    init_genrand64(seed);

    for (i=0; i<TSTEADY; i++) { m[i] = 0; }
    for (i=0; i<2; i++) { pdE[i] = exp(-J/(kb*T)*4*(i+1)); }

    // put vars to info struct
    for (i=0; i<NTHREAD; i++){
        infos[i].pid        = i;
        infos[i].id_start   = i*(GRID_SIZE/NTHREAD);
        infos[i].id_end     = (i+1)*(GRID_SIZE/NTHREAD);

        infos[i].tmax       = tmax;
        
        infos[i].IsingNtk   = IsingNtk;
        infos[i].IsingGrid  = IsingGrid;

        infos[i].pdE        = pdE;
        infos[i].m          = m;
        infos[i].fid        = fid;
    }
    if (infos[NTHREAD-1].id_end != GRID_SIZE){
        infos[NTHREAD-1].id_end = GRID_SIZE;
    }

    // run pthread
    for (i=0; i<NTHREAD; i++){
        pthread_create(&(pids[i]), NULL, run_sub, (void*) &(infos[i]));
    }

    for (i=0; i<NTHREAD; i++){
        pthread_join(pids[i], NULL);
    }

    // get order parameter
    for (i=0; i<3; i++) { orders[i] = 0; }

    for (i=0; i<TSTEADY; i++){
        mall += m[i];
        m2 += m[i]*m[i];
        m4 += m[i]*m[i]*m[i]*m[i];
    }
    mall /= TSTEADY;
    m2 /= TSTEADY;
    m4 /= TSTEADY;

    sus = GRID_SIZE * (m2 - mall*mall);
    bin_cum = 1 - (m4 / (3*m2*m2));
    
    orders[0] = mall;
    orders[1] = sus;
    orders[2] = bin_cum;

    if (fid != NULL){
        fclose(fid);
    }
    free(m);

    return;

}

void *run_sub(void *arg){

    int i, j, t, n, len, s;
    ising_t *info = (ising_t*) arg;
    double dE, mt, p;
    
    len = info->id_end-info->id_start;
    for (t=0; t<info->tmax; t++){
        for (i=0; i<len; i++){
            n = (int) (genrand64_real2()*len + info->id_start);
            s = 0;
            for (j=0; j<4; j++){
                s += info->IsingGrid[info->IsingNtk[4*n+j]];
            }
            s *= info->IsingGrid[n];
            dE = 2*J*s;

            if (dE <= 0){
                info->IsingGrid[n] *= -1;
            } else {
                p = genrand64_real2();
                if (p < info->pdE[s/2-1]){
                    info->IsingGrid[n] *= -1;
                }
            }
        }

        if (t >= info->tmax-TSTEADY){
            mt = 0;
            for (i=info->id_start; i<info->id_end; i++){
                mt += info->IsingGrid[i];
            }
            mt /= (double) GRID_SIZE;

            n = t-(info->tmax-TSTEADY);
            pthread_mutex_lock(&mutex);
            // save order parameter
            info->m[n] += fabs(mt);

            if (info->fid != NULL){
                fprintf(info->fid, "%d:%d:", n, info->pid);
                for (i=info->id_start; i<info->id_end; i++){
                    fprintf(info->fid, "%d,", info->IsingGrid[i]);
                }
                fprintf(info->fid, "\n");
            }
            pthread_mutex_unlock(&mutex);
        }
    }
}

void initIsingNtk(int *ntk){

    int n,i,j,w;
    w = sqrt(GRID_SIZE);

    for (n=0; n<GRID_SIZE; n++){
        i = n/w;
        j = n%w;

        if (i == 0){
            ntk[n*4] = w*(w-1)+j;
            ntk[n*4+2] = w+j;
        } else if (i == w-1){
            ntk[n*4] = n-w;
            ntk[n*4+2] = j;
        } else{
            ntk[n*4] = n-w;
            ntk[n*4+2] = n+w;
        }

        if (j == 0){
            ntk[n*4+1] = n+1;
            ntk[n*4+3] = n+w-1;
        } else if (j == w-1){
            ntk[n*4+1] = n-w+1;
            ntk[n*4+3] = n-1;
        } else {
            ntk[n*4+1] = n+1;
            ntk[n*4+3] = n-1;
        }

    }

}