/*
* Copyright (c) 2013 Battelle Memorial Institute.
* All rights reserved.
*
* The software in this package is published under the terms of the BSD
* style license a copy of which has been included with this distribution in
* the LICENSE.txt file.
*
* Created on 2013 by John Feo
*/

/* C version of the SCF code */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cscc.h"
#include "integ.h"
#include "input.h"
#include "output.h"
#include "timer.h"
#include "twoel.h"

//function declaration
double makesz(void);
void ininrm(void);
void makden(void);
double h(int, int);
double s(int, int);
double g(int, int, int, int);
double oneel(double);
void Dgemm(char, char, int, int, int, double **, double **, double **);
int Eigen_gen(double **, double **, double **);
int Eigen_std(double **, double **, double *);
void damp(double fac);
double dendif(void);
double diagon(int);
void makeob(void);
void denges(void);
void setarrays(void);
void closearrays(void);
void swap_g_fock(void);

double enrep, q[maxatom], ax[maxatom], ay[maxatom], az[maxatom];
Halide::Runtime::Buffer<double, 1> x_buf, y_buf, z_buf, expnt_buf, rnorm_buf, g_schwarz_max_j_buf;
double *x, *y, *z, *expnt, *rnorm, *g_schwarz_max_j;
Halide::Runtime::Buffer<double, 2> g_dens_buf, g_fock_double_buf[2], *g_fock_buf, *g_fock_out_buf, g_schwarz_buf;
double ** g_dens, ** g_fock, **g_fock_out, ** g_schwarz, ** g_tfock, ** g_work, ** g_ident, ** g_orbs, * eigv;

int iky[maxnbfn], nocc, nbfn;
int natom;

int main(int argc, char **argv) {

  double tinit = 0.0, tonel = 0.0, ttwoel = 0.0, tdiag  = 0.0, tdens = 0.0, tprint = 0.0;
  double eone  = 0.0, etwo  = 0.0, energy = 0.0, deltad = 0.0;

  int iter;
  double scale, totsec, tester, schwmax;

// get input from file be.inpt;
  input();

// create and allocate global arrays;
  setarrays();
  ininrm();

// create initial guess for density matrix by using single atom densities;
  denges();
  tinit = timer();

// make initial orthogonal orbital set for solution method using similarity transform;
  makeob();

// make info for sparsity test;
  schwmax = makesz();

// *** ITERATE ****
  for (iter = 0; iter <  mxiter; iter++) {

// make the one particle contribution to the fock matrix and get the partial contribution to the energy;
    eone  = oneel(schwmax);
    tonel = tonel + timer();

// compute the two particle contributions to the fock matrix and get the total energy;
    {
        Halide::Runtime::Buffer<double, 1> etwo_buffer(1);
        extern double rdelta, delta, delo2; // integ.c
        extern Halide::Runtime::Buffer<double> fm; // integ.c
        int error = twoel(delo2, delta, rdelta, expnt_buf, rnorm_buf, x_buf, y_buf, z_buf, fm_buf, *g_fock_buf, g_dens_buf, etwo_buffer, *g_fock_out_buf);
        assert(!error);
        swap_g_fock();
#ifdef TRACING
        printf("twoel took %f seconds\n", timer());
        exit(0);
#endif /* TRACING */
        etwo = etwo_buffer(0);
    }
    ttwoel = ttwoel + timer();

// diagonalize the fock matrix.
    tester = diagon(iter);
    tdiag  = tdiag + timer();

// make the new density matrix in g_work from orbitals in g_orbs, compute the norm of the change in
// the density matrix and then update the density matrix in g_dens with damping.
    makden();
    deltad = dendif();

    if (iter == 0)
       scale = 0.00;
    else if (iter < 5)
       scale = (nbfn > 60) ? 0.50 : 0.00;
    else
       scale = 0.00;

    damp(scale);
    tdens = tdens + timer();

// add up energy and print out convergence information;
    energy = enrep + eone + etwo;
    printf(" iter= %3d, energy=%15.8f, deltad= %9.7f, deltaf=%9.7f\n", iter, energy, deltad, tester);
    tprint = tprint + timer();

// if converged exit iteration loop;
    if (deltad < tol) break;

  }

  if (iter == mxiter) {printf("SCF failed to converge in %d iters\n", iter); iter --;}

//...v....1....v....2....v....3....v....4....v....5....v....6....v....7..;
//
//     finished ... print out eigenvalues and occupied orbitals;
//

// print out timing information;

  printf("\n\nfinal energy = %18.11f\n",energy);
  printf("\neigenvalues\n\n");
  output(eigv, 0, MIN(nbfn, nocc + 5), 0, 1, nbfn, 1, 1);

  printf("      init       onel      twoel       diag       dens       print       ncpu    \n");
  printf("      ----       ----      -----       ----       ----       ----        ----    \n");

  printf("%10.2f %10.2f %10.2f %10.2f %10.2f %10.2f", tinit, tonel, ttwoel, tdiag, tdens, tprint);

  totsec = tinit + tonel + ttwoel + tdiag + tdens + tprint;
  printf("\n elapsed time in seconds %10.2f\n\n", totsec);

  closearrays();
}

double makesz() {
  int i, j;
  double smax = 0.0;

  for (j = 0; j < nbfn; j++) {
  double jmax = 0.0;
  for (i = 0; i < nbfn; i++) {
    double gg = sqrt( g(i, j, i, j) );
    if (gg > smax) smax = gg;
    g_schwarz[i][j] = gg;
    if (gg > jmax) jmax = gg;
  }
  g_schwarz_max_j[j] = jmax;
  }

  return smax;
}

void ininrm(void)  {
  int i;
  long long int bf4 = pow((long long int) nbfn, 4);

// write a little welcome message;
  printf(" Example Direct Self Consistent Field Program \n");
  printf(" -------------------------------------------- \n\n");
  printf(" no. of atoms .............. %5d\n", natom);
  printf(" no. of occupied orbitals .. %5d\n", nocc);
  printf(" no. of basis functions .... %5d\n", nbfn);
  printf(" basis functions^4 ......... %5lld\n", bf4);
  printf(" convergence threshold ..... %9.4f\n", tol);

// generate normalisation coefficients for the basis functions and the index array iky;
  for (i = 0; i < nbfn; i++) iky[i] = (i + 1) * i / 2;
  for (i = 0;i < nbfn;i++) rnorm[i] = pow((expnt[i] * 2.00 / pi), 0.750);

// initialize common for computing f0;
  setfm();
}

// generate the one particle hamiltonian matrix element over the normalized primitive 1s functions
double h(int i,int j) {
  double f0val = 0.00, sum = 0.00;
  double facij,expij,repij, xp,yp,zp,rpc2, rab2;
  int iat;

  rab2  = (x[i]-x[j])*(x[i]-x[j]) + (y[i]-y[j])*(y[i]-y[j]) + (z[i]-z[j])*(z[i]-z[j]);
  facij = expnt[i]*expnt[j]/(expnt[i]+expnt[j]);
  expij = exprjh(-facij * rab2);
  repij = (2.00 * pi / (expnt[i] + expnt[j])) * expij;

// first do the nuclear attraction integrals;
  for (iat = 0;iat < natom;iat++) {
      xp = (x[i]*expnt[i] + x[j]*expnt[j])/(expnt[i]+expnt[j]);
      yp = (y[i]*expnt[i] + y[j]*expnt[j])/(expnt[i]+expnt[j]);
      zp = (z[i]*expnt[i] + z[j]*expnt[j])/(expnt[i]+expnt[j]);
      rpc2 = (xp-ax[iat])*(xp-ax[iat]) + (yp-ay[iat])*(yp-ay[iat]) + (zp-az[iat])*(zp-az[iat]);
      f0(&f0val, (expnt[i] + expnt[j]) * rpc2);
      sum = sum - repij * q[iat] * f0val;
   }

// add on the kinetic energy term;
   sum = sum + facij*(3.00-2.00*facij*rab2) * pow((pi/(expnt[i]+expnt[j])),1.50) * expij;

// finally multiply by the normalization constants;
   return sum * rnorm[i] * rnorm[j];
}


// generate the overlap matrix element between the normalized primitve gaussian 1s functions i and j;
double s(int i, int j) {
  double rab2, facij, temp;

  rab2  = (x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]) + (z[i] - z[j]) * (z[i] - z[j]);
  facij = expnt[i] * expnt[j] / (expnt[i] + expnt[j]);
  temp  = pow((pi / (expnt[i] + expnt[j])), 1.50) * exprjh(-facij * rab2) * rnorm[i] * rnorm[j];
  return temp;
}


// generate density matrix from orbitals in g_orbs. the first nocc orbitals are doubly occupied.
void makden(void) {
  int i, j, k;

  for (i = 0; i < nbfn; i++) {
  for (j = 0; j < nbfn; j++) {
    double p = 0.0;
    for (k = 0; k < nocc; k++) p = p + g_orbs[i][k] * g_orbs[j][k];
    g_work[i][j] = 2.0 * p;
  } }

  return;
}


// fill in the one-electron part of the fock matrix and compute the one-electron energy contribution;
double oneel(double schwmax) {
  int i, j;

  for (i = 0; i < nbfn; i++) {
  for (j = 0; j < nbfn; j++) {
    double gg = g_schwarz[i][j] * schwmax;
    g_fock[i][j] = (gg > tol2e) ? h(i, j) : 0.0;
  } }

  double rv = 0.0;

  for (i = 0; i < nbfn; i++) {
  for (j = 0; j < nbfn; j++) {
      rv += g_fock[i][j] * g_dens[i][j];
  } }

  return 0.50 * rv;
}


// create damped density matrix as a linear combination of old density matrix
// and density matrix formed from new orbitals;
void damp(double fac) {
  int i, j;
  double ofac = 1.00 - fac;

  for (i = 0; i < nbfn; i++) {
  for (j = 0; j < nbfn; j++) {
      g_dens[i][j] = fac * g_dens[i][j] + ofac * g_work[i][j];
} } }


// compute largest change in {density - work} matrix elements;
double dendif(void) {
  int i, j;
  double denmax = 0.00;

  for (i = 0; i < nbfn; i++) {
  for (j = 0; j < nbfn; j++) {
      double xdiff = fabs(g_dens[i][j] - g_work[i][j]);
      if (xdiff > denmax) denmax = xdiff;
  } }

  return denmax;
}


// Compute the two electon integral (ij|kl) over normalized; primitive 1s gaussians;
double g(int i, int j, int k, int l) {
  double f0val;
  double dxij = x[i] - x[j];
  double dyij = y[i] - y[j];
  double dzij = z[i] - z[j];
  double dxkl = x[k] - x[l];
  double dykl = y[k] - y[l];
  double dzkl = z[k] - z[l];

  double rab2 = dxij * dxij + dyij * dyij + dzij * dzij;
  double rcd2 = dxkl * dxkl + dykl * dykl + dzkl * dzkl;

  double expntIJ     = expnt[i] + expnt[j];
  double expntKL     = expnt[k] + expnt[l];
  double expntIJ_inv = 1.0 / expntIJ;
  double expntKL_inv = 1.0 / expntKL;

  double facij  = expnt[i] * expnt[j] * expntIJ_inv;
  double fackl  = expnt[k] * expnt[l] * expntKL_inv;
  double exijkl = exprjh(-facij * rab2 - fackl * rcd2);
  double denom  = expntIJ * expntKL * sqrt(expntIJ + expntKL);
  double fac    = (expntIJ) * (expntKL) / (expntIJ + expntKL);

  double xp   = (x[i] * expnt[i] + x[j] * expnt[j]) * expntIJ_inv;
  double yp   = (y[i] * expnt[i] + y[j] * expnt[j]) * expntIJ_inv;
  double zp   = (z[i] * expnt[i] + z[j] * expnt[j]) * expntIJ_inv;
  double xq   = (x[k] * expnt[k] + x[l] * expnt[l]) * expntKL_inv;
  double yq   = (y[k] * expnt[k] + y[l] * expnt[l]) * expntKL_inv;
  double zq   = (z[k] * expnt[k] + z[l] * expnt[l]) * expntKL_inv;
  double rpq2 = (xp - xq) * (xp - xq) + (yp - yq) * (yp - yq) + (zp - zq) * (zp - zq);

  f0(&f0val, fac * rpq2);
  return (2.00 * pow(pi, 2.50) / denom) * exijkl * f0val * rnorm[i] * rnorm[j] * rnorm[k] * rnorm[l];
}


double diagon(int iter) {
  int i, j;
  double shift, tester = 0.00;

// use similarity transform to solve standard eigenvalue problem
// (overlap matrix has been transformed out of the problem);

  Dgemm('n', 'n', nbfn, nbfn, nbfn, g_fock, g_orbs,  g_tfock);
  Dgemm('t', 'n', nbfn, nbfn, nbfn, g_orbs, g_tfock, g_fock );

// compute largest change in off-diagonal fock matrix elements;
  for (j = 0; j < nbfn; j++) {
  for (i = 0; i < nbfn; i++) {
      if (i == j) continue;
      double xtmp = fabs(g_fock[i][j]);
      if (xtmp > tester) tester = xtmp;
  } }

  shift = 0.00;
  if      (tester > 0.30) shift = 0.30;
  else if (nbfn   > 60  ) shift = 0.10;
  else                    shift = 0.00;

  if (iter >= 1 && shift != 0.00)
     for (i = nocc; i < nbfn; i++) g_fock[i][i] += shift;

  for (i = 0; i < nbfn; i ++)
  for (j = 0; j < nbfn; j ++)
      g_tfock[i][j] = g_orbs[i][j];

  Eigen_std(g_fock, g_work, eigv);

// back transform eigenvectors;
  Dgemm('n', 'n', nbfn, nbfn, nbfn, g_tfock, g_work, g_orbs);

  if (iter >= 1 && shift != 0.00)
    for (i = nocc; i < nbfn; i++) eigv[i] = eigv[i] - shift;

  return tester;
}


// generate set of orthonormal vectors by creating a random symmetric matrix
// and solving associated generalized eigenvalue problem using the correct overlap matrix.
void makeob(void) {
  int i, j;

  for (i = 0; i < nbfn; i ++) {
  for (j = 0; j < nbfn; j ++) {
      g_ident[i][j] = s(i, j);
      g_fock [i][j] = 0.5;
  } }

  Eigen_gen(g_fock, g_ident, g_orbs);
  return;
}


// form guess density from superposition of atomic densities in the AO basis set ...
// instead of doing the atomic SCF hardwire for this small basis set for the Be atom;
void denges(void) {
  int i, j, k, l;

  double atdens[15][15] = {
    {0.000002,0.000027,0.000129,0.000428,0.000950,0.001180,
     0.000457,-0.000270,-0.000271,0.000004,0.000004,0.000004,
     0.000004,0.000004,0.000004},
    {0.000027,0.000102,0.000987,
     0.003269,0.007254,0.009007,0.003492,-0.002099,-0.002108,
     0.000035,0.000035,0.000035,0.000035,0.000035,0.000035},
    {0.000129,0.000987,0.002381,0.015766,0.034988,0.043433,
     0.016835,-0.010038,-0.010082,0.000166,0.000166,0.000166,
     0.000166,0.000166,0.000166},
    {0.000428,0.003269,0.015766,
     0.026100,0.115858,0.144064,0.055967,-0.035878,-0.035990,
     0.000584,0.000584,0.000584,0.000584,0.000584,0.000584},
    {0.000950,0.007254,0.034988,0.115858,0.128586,0.320120,
     0.124539,-0.083334,-0.083536,0.001346,0.001346,0.001346,
     0.001346,0.001346,0.001346},
    {0.001180,0.009007,0.043433,
     0.144064,0.320120,0.201952,0.159935,-0.162762,-0.162267,
     0.002471,0.002471,0.002471,0.002471,0.002471,0.002471},
    {0.000457,0.003492,0.016835,0.055967,0.124539,0.159935,
     0.032378,-0.093780,-0.093202,0.001372,0.001372,0.001372,
     0.001372,0.001372,0.001372},
    {-0.000270,-0.002099,-0.010038,
     -0.035878,-0.083334,-0.162762,-0.093780,0.334488,0.660918,
     -0.009090,-0.009090,-0.009090,-0.009090,-0.009090,-0.009090},
    {-0.000271,-0.002108,-0.010082,-0.035990,-0.083536,-0.162267,
     -0.093202,0.660918,0.326482,-0.008982,-0.008982,-0.008981,
     -0.008981,-0.008981,-0.008982},
    {0.000004,0.000035,0.000166,
     0.000584,0.001346,0.002471,0.001372,-0.009090,-0.008982,
     0.000062,0.000124,0.000124,0.000124,0.000124,0.000124},
    {0.000004,0.000035,0.000166,0.000584,0.001346,0.002471,
     0.001372,-0.009090,-0.008982,0.000124,0.000062,0.000124,
     0.000124,0.000124,0.000124},
    {0.000004,0.000035,0.000166,
     0.000584,0.001346,0.002471,0.001372,-0.009090,-0.008981,
     0.000124,0.000124,0.000062,0.000124,0.000124,0.000124},
    {0.000004,0.000035,0.000166,0.000584,0.001346,0.002471,
     0.001372,-0.009090,-0.008981,0.000124,0.000124,0.000124,
     0.000062,0.000124,0.000124},
    {0.000004,0.000035,0.000166,
     0.000584,0.001346,0.002471,0.001372,-0.009090,-0.008981,
     0.000124,0.000124,0.000124,0.000124,0.000062,0.000124},
    {0.000004,0.000035,0.000166,0.000584,0.001346,0.002471,
     0.001372,-0.009090,-0.008982,0.000124,0.000124,0.000124,
     0.000124,0.000124,0.000062}};

// correct for a factor of two along the diagonal;
  for (i = 0; i < 15; i++) atdens[i][i] = 2.00 * atdens[i][i];

// fill in each block of 15 rows and 15 columns
  for (i = 0; i < nbfn; i++)
  for (j = 0; j < nbfn; j++) g_dens[i][j] = 0.0;

  for (i = 0; i < nbfn; i += 15) {
      for (k = 0; k < 15; k++) {
      for (l = 0; l < 15; l++) {
          g_dens[i + k][i + l] = atdens[k][l];
  } } }

  return;
}


void setarrays(void) {
  int i, j;

  g_dens_buf    = Halide::Runtime::Buffer<double>(nbfn, nbfn);
  g_schwarz_buf = Halide::Runtime::Buffer<double>(nbfn, nbfn);
  g_fock_double_buf[0] = Halide::Runtime::Buffer<double>(nbfn, nbfn);
  g_fock_double_buf[1] = Halide::Runtime::Buffer<double>(nbfn, nbfn);
  g_dens    = (double **) malloc(nbfn * sizeof(double *));
  g_schwarz = (double **) malloc(nbfn * sizeof(double *));
  g_fock    = (double **) malloc(nbfn * sizeof(double *));
  g_fock_out = (double **) malloc(nbfn * sizeof(double *));
  g_tfock   = (double **) malloc(nbfn * sizeof(double *));
  g_work    = (double **) malloc(nbfn * sizeof(double *));
  g_ident   = (double **) malloc(nbfn * sizeof(double *));
  g_orbs    = (double **) malloc(nbfn * sizeof(double *));
  eigv      = (double *)  malloc(nbfn * sizeof(double));

  g_fock_buf = &g_fock_double_buf[0];
  g_fock_out_buf = &g_fock_double_buf[1];

  g_dens[0]    = g_dens_buf.begin();
  g_fock[0]     = g_fock_buf->begin();
  g_fock_out[0] = g_fock_out_buf->begin();
  g_schwarz[0] = g_schwarz_buf.begin();
  g_tfock[0]   = (double *) malloc(nbfn * nbfn * sizeof(double));
  g_work[0]    = (double *) malloc(nbfn * nbfn * sizeof(double));
  g_ident[0]   = (double *) malloc(nbfn * nbfn * sizeof(double));
  g_orbs [0]   = (double *) malloc(nbfn * nbfn * sizeof(double));

  for (i = 1; i < nbfn; i ++) {
      g_dens[i]    = i * nbfn + g_dens[0];
      g_schwarz[i] = i * nbfn + g_schwarz[0];
      g_fock[i]    = i * nbfn + g_fock[0];
      g_fock_out[i] = i * nbfn + g_fock_out[0];
      g_tfock[i]   = i * nbfn + g_tfock[0];
      g_work[i]    = i * nbfn + g_work[0];
      g_ident[i]   = i * nbfn + g_ident[0];
      g_orbs [i]   = i * nbfn + g_orbs[0];
  }

  for (i = 0; i < nbfn; i ++) {
  for (j = 0; j < nbfn; j ++) {
      g_dens[i][j]    = 0.0;
      g_schwarz[i][j] = 0.0;
      g_fock[i][j]    = 0.0;
      g_fock_out[i][j] = 0.0;
      g_tfock[i][j]   = 0.0;
      g_work[i][j]    = 0.0;
      g_ident[i][j]   = 0.0;
      g_orbs [i][j]   = 0.0;
} } }

void swap_g_fock(void) {
    Halide::Runtime::Buffer<double, 2> *tmp_buf = g_fock_buf;
    g_fock_buf = g_fock_out_buf;
    g_fock_out_buf = tmp_buf;
    double **tmp = g_fock;
    g_fock = g_fock_out;
    g_fock_out = tmp;
}

void closearrays(void) {

  free(eigv);
  free(g_dens);
  free(g_schwarz);
  free(g_fock);
  free(g_fock_out);
  g_dens_buf.~Buffer();
  g_schwarz_buf.~Buffer();
  g_fock_double_buf[0].~Buffer();
  g_fock_double_buf[1].~Buffer();
  g_schwarz_max_j_buf.~Buffer();
  g_fock = NULL;
  g_fock_out = NULL;

  free(g_tfock[0]);   free(g_tfock);
  free(g_work[0]);    free(g_work);
  free(g_ident[0]);   free(g_ident);
  free(g_orbs[0]);    free(g_orbs);

  return;
}
