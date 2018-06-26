
/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include<mpi.h>
#include<string.h>
#include<assert.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER 0

// some constant values used throught the program
#define C_SQ 			(1.0 / 3.0) /* square of speed of sound */
#define W0 			(4.0 / 9.0)  /* weighting factor */
#define W1 		     	(1.0 / 9.0)  /* weighting factor */
#define W2 	       		(1.0 / 36.0) /* weighting factor */
#define CSQ_2_CSQ 		(2.0 * C_SQ * C_SQ)
#define CSQ_2 			(2.0 * C_SQ)

#define check(x) if(isnan(x) || isinf(x) == 1 || isinf(x) == -1) { printf("%lf at line = %d", x, __LINE__); exit(1); }

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/   
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);


/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
int getTotCells(int* obstacles, int NY, int NX);
int calc_nrows_from_rank_unfair(int rank,int size, int NX);
int calc_nrows_from_rank_fair(int rank, int size, int ny);
void initialiseLocal(t_speed* grid, int nrows, int  ncols, float density);
void initialiseObstacles(int* obst, int nrows, int ncols, int rank, int *obstacles, int nx);
void initLoc( t_param params, t_speed** grid, t_speed** tmp_grid, int** obst,int* obstacles,float** loc_vels,int nrows,int ncols, int rank, int len_procs[], int start[]);
/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]){
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic = 0, toc = 0;              /* doubleing point numbers to calculate elapsed wallclock time */
  double usrtim = 0;                /* doubleing point number to record elapsed user CPU time */
  double systim = 0;                /* doubleing point number to record elapsed system CPU time */
  int tot_cells;  /* no. of cells used in calculation */
  float tot_u;        /* accumulated magnitudes of velocity for each cell */

  // MPI VARIABLES
  int rank;              /* the rank of this process */
  int up;              /* the rank of the process above */
  int down;             /* the rank of the process below */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int nrows;       /* number of rows apportioned to this rank */
  int ncols;       /* number of columns apportioned to this rank */
  t_speed* grid;         /* local grid */
  t_speed* tmp_grid;     /* local temp grid */
  int* obst;             // local obstacles grid
  float* sendbuf;       /* buffer to hold values to send */
  float* recvbuf;      /* buffer to hold received values */
  float* loc_vels;     /* local average velocities */
  float* printbuf;     // to receive entire grids

  /* MPI_Init returns once it has started up processes, get size and rank */ 
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Request requests[4];
  
  /* parse the command line */
  if (argc != 3){
    usage(argv[0]);
  } else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }
  
  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
  
  int NX = params.nx; // columns
  int NY = params.ny; // rows
  float omega = params.omega;
  float v1 = params.density * params.accel / 9.0;
  float v2 = params.density * params.accel / 36.0;
  int maxIters = params.maxIters;
  
  // determine process ranks to the left and right of rank respecting periodic boundary conditions
  down = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  up = (rank + 1) % size;
  
  // determine local grid size: each rank gets all the cols, but a subset of the number of rows
  nrows = calc_nrows_from_rank_unfair(rank, size, NY);
  //nrows = calc_nrows_from_rank_fair(rank, size, NY);
  if(nrows < 1){
    fprintf(stderr, "too many processes!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  ncols = NX;
  
  // get rows start and length of each process
  int remote_nrows[size];
  int len_procs[size];
  int start[size];
  for(int i = 0; i < size; i++){
    // remote_nrows[i] = calc_nrows_from_rank_fair(i, size, NY);
      remote_nrows[i] = calc_nrows_from_rank_unfair(i, size, NY);
    len_procs[i] = ncols * remote_nrows[i];
    if(i == 0){
      start[0] = 0;
    } else {
      start[i] = start[i-1] + len_procs[i-1];
    }
  }
  
  // initialise local grids
  initLoc(params, &grid, &tmp_grid, &obst, obstacles, &loc_vels, nrows, ncols, rank, len_procs, start);
  // each rank has the same sum of speeds and the total is equal to the global one.
  
  // allocate space for buffers
  int buflen = ncols * NSPEEDS;
  sendbuf    = (float*) malloc(sizeof(float) * buflen);
  recvbuf    = (float*) malloc(sizeof(float) * buflen);
  int lenproc = ncols * nrows;
  printbuf = (float*)malloc(sizeof(float) * lenproc * NSPEEDS);
  
  
  tot_cells = getTotCells(obstacles, NY, NX);
  int inner_up = nrows * ncols;
  int halo_up = (nrows+1) * ncols;
  
  
  if(rank == MASTER){
    /* iterate for maxIters timesteps */
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  }
  
  for (int tt = 0; tt < maxIters; tt++){ 								  		  	     
    
     if(rank == size - 1){
      // ACCELERATE FLOW
      // modify the 2nd row of the grid
      int ii = nrows - 1; // one is halo (nrows - 2 + 1)
      int sndRow = ii * ncols; 
      for (int jj = 0; jj < ncols; jj++) {
	int pos = sndRow + jj;
	int obst_pos = (ii - 1) * ncols + jj;
	float* s = grid[pos].speeds;
	
	// if the cell is not occupied and we don't send a negative density 
	if (!obst[obst_pos] && (s[3] - v1) > 0.0 && (s[6] - v2) > 0.0 && (s[7] - v2) > 0.0){
	  // increase 'east-side' densities 
	  s[1] += v1;	s[5] += v2;	s[8] += v2;
	  // decrease 'west-side' densities 
	  s[3] -= v1;     s[6] -= v2;     s[7] -= v2;
	}
      }
    }    
    
    // HALO EXCHANGE
    /* send to  down, receive from up */
    MPI_Isend(&grid[ncols], buflen, MPI_FLOAT, down, tag, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&grid[halo_up], buflen, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &requests[1]);
    
    MPI_Isend(&grid[inner_up], buflen, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&grid[0], buflen, MPI_FLOAT, down, 0, MPI_COMM_WORLD, &requests[3]);
    // --- END HALO EXCHANGE
    
    // wait all requests
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    
    // PROPAGATE
    /* loop over _all_ cells */
    for (int ii = 1; ii < nrows + 1;ii++){
      for (int jj = 0; jj < ncols; jj++) {	
	int pos = ii * ncols + jj;
	float* ts = tmp_grid[pos].speeds;	
	
	// determine indices of axis-direction neighbours respecting periodic boundary conditions (wrap around) 
	int y_n = ii + 1;
	int x_e = (jj == ncols - 1) ? 0 : (jj + 1) ;
	int y_s = ii - 1;
	int x_w = (!jj) ? (ncols - 1) : (jj - 1);
	// propagate densities to neighbouring cells, following  appropriate directions of travel and writing into scratch space grid 
	ts[0] = grid[ ii * ncols + jj ].speeds[0];  // central cell, no movement 
	ts[1] = grid[ ii * ncols + x_w].speeds[1];  // east 
	ts[2] = grid[y_s * ncols + jj ].speeds[2]; // north 
    ts[3] = grid[ ii * ncols + x_e].speeds[3];  // west 
	ts[4] = grid[y_n * ncols + jj ].speeds[4]; // south 
	ts[5] = grid[y_s * ncols + x_w].speeds[5]; // north-east 
	ts[6] = grid[y_s * ncols + x_e].speeds[6]; // north-west 
	ts[7] = grid[y_n * ncols + x_e].speeds[7]; // south-west 
	ts[8] = grid[y_n * ncols + x_w].speeds[8]; // south-east 	
      } 
    }
    
    /* initialise */
    tot_u = 0.0;

     // COLLISION - REBOUND - AV_VELOCITY
    
    /* loop over the cells in the grid 
       NB: the collision step is called after the propagate step and so values of interest are in the scratch-space grid */
    for (int ii = 1; ii < nrows + 1; ii++){ 
      for (int jj = 0; jj < ncols; jj++){
	int pos = ii * ncols  + jj; 
	int ob_pos = (ii-1) * ncols + jj;
	float* ts = tmp_grid[pos].speeds;
	float* s = grid[pos].speeds;
	
	// COLLISION
	// don't consider occupied cells 
	if (!obst[ob_pos]){
	  // compute local density total 
	  float local_density = ts[0] + ts[1] + ts[2] +  ts[3] + ts[4] + ts[5] + ts[6] + ts[7] + ts[8];
      
	  // compute x velocity component 
	  float u_x = (ts[1] + ts[5] + ts[8] - (ts[3] + ts[6] + ts[7])) / local_density;
	  // compute y velocity component 
	  float u_y = (ts[2] + ts[5] + ts[6] - (ts[4] + ts[7] + ts[8])) / local_density;
	  
	  // velocity squared 
	  float u_sq = u_x * u_x + u_y * u_y;
	  // AVERAGE VELOCITY
	  // accumulate the norm of x- and y- velocity components 
	  tot_u += sqrt(u_sq);
	  
	  // directional velocity components 
	  float u[NSPEEDS];
	  u[1] =   u_x;        // east          
	  u[2] =         u_y;  // north    
	  u[3] = - u_x;        // west  
	  u[4] =       - u_y;  // south 
	  u[5] =   u_x + u_y;  // north-east  
	  u[6] = - u_x + u_y;  // north-west  
	  u[7] = - u_x - u_y;  // south-west 
	  u[8] =   u_x - u_y;  // south-east 
	  
	  // equilibrium densities 
	  float d_equ[NSPEEDS];
	  // zero velocity density: weight w0 
	  d_equ[0] = W0 * local_density * (1.0 - u_sq / (CSQ_2));
	  
	  // axis speeds: weight w1 
	  float partial = W1 * local_density;
	  float par = u_sq / (CSQ_2);
	  d_equ[1] = partial * (1.0 + u[1] / C_SQ + (u[1] * u[1]) / (CSQ_2_CSQ) - par);
	  d_equ[2] = partial * (1.0 + u[2] / C_SQ + (u[2] * u[2]) / (CSQ_2_CSQ) - par);
	  d_equ[3] = partial * (1.0 + u[3] / C_SQ + (u[3] * u[3]) / (CSQ_2_CSQ) - par);
	  d_equ[4] = partial * (1.0 + u[4] / C_SQ + (u[4] * u[4]) / (CSQ_2_CSQ) - par);
	  // diagonal speeds: weight w2 
	  partial = W2 * local_density;
	  d_equ[5] = partial * (1.0 + u[5] / C_SQ + (u[5] * u[5]) / (CSQ_2_CSQ) - par);
	  d_equ[6] = partial * (1.0 + u[6] / C_SQ + (u[6] * u[6]) / (CSQ_2_CSQ) - par);
	  d_equ[7] = partial * (1.0 + u[7] / C_SQ + (u[7] * u[7]) / (CSQ_2_CSQ) - par);
	  d_equ[8] = partial * (1.0 + u[8] / C_SQ + (u[8] * u[8]) / (CSQ_2_CSQ) - par);
          
	  // relaxation step 
	  s[0] = ts[0] + omega * (d_equ[0] - ts[0]);
	  s[1] = ts[1] + omega * (d_equ[1] - ts[1]);
	  s[2] = ts[2] + omega * (d_equ[2] - ts[2]);
	  s[3] = ts[3] + omega * (d_equ[3] - ts[3]);
	  s[4] = ts[4] + omega * (d_equ[4] - ts[4]);
	  s[5] = ts[5] + omega * (d_equ[5] - ts[5]);
	  s[6] = ts[6] + omega * (d_equ[6] - ts[6]);
	  s[7] = ts[7] + omega * (d_equ[7] - ts[7]);
	  s[8] = ts[8] + omega * (d_equ[8] - ts[8]); 
      	} else {
	  // REBOUND
	  // called after propagate, so taking values from scratch space mirroring, and writing into main grid 
	  s[1] = ts[3];
	  s[2] = ts[4]; 
	  s[3] = ts[1]; 
	  s[4] = ts[2]; 
	  s[5] = ts[7]; 
	  s[6] = ts[8]; 
	  s[7] = ts[5]; 
	  s[8] = ts[6]; 
      	}
      }	
    }

    loc_vels[tt] = tot_u / (float) tot_cells;
  } 
  
  // reduce average velocities while the non-blocking sends and receives are executing
  MPI_Reduce(loc_vels, av_vels, maxIters, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  
  
  if(rank == MASTER){ 
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    
   }
   
  // --------COMBINE
  if(rank == MASTER) {
    int nread = len_procs[0];
    // first fill the value of its own local grid
    memcpy(&cells[0], &grid[1 * ncols], sizeof(float) * nread * NSPEEDS);
    // iterate through processes 
    
    for(int kk = 1; kk < size; kk++) { 
      // receive a grid  
      MPI_Recv(&cells[nread], len_procs[kk] * NSPEEDS, MPI_FLOAT, kk, 0, MPI_COMM_WORLD, &status);
      nread += len_procs[kk];
      //MPI_Irecv(&cells[lenproc*kk], NSPEEDS*lenproc, MPI_FLOAT, kk, 0, MPI_COMM_WORLD, &requests[kk-1]);
    }
  } else {
    MPI_Send(&grid[ncols], NSPEEDS * lenproc, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
    
    //MPI_Isend(&grid[ncols], NSPEEDS * lenproc, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &reqSend);
  }
  //----------- END COMBINE
  
  if(rank == MASTER){
#ifdef DEBUG
    for (int tt = 0; tt < params.maxIters; tt++){
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
    }
#endif

    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
  } 
  
  // every process free global grids
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  
  MPI_Finalize();
  
  // free memory for local grids
  free(grid);
  free(tmp_grid);
  free(sendbuf);
  free(recvbuf);
  free(obst);
  free(loc_vels);
  free(printbuf);
  
  
  return EXIT_SUCCESS;
}




int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0 / 9.0;
  float w1 = params->density      / 9.0;
  float w2 = params->density      / 36.0;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}



int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;

  /* loop over all non-blocked cells */
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        float local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii * params.nx + jj].speeds[1]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[8]
                      - (cells[ii * params.nx + jj].speeds[3]
                         + cells[ii * params.nx + jj].speeds[6]
                         + cells[ii * params.nx + jj].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii * params.nx + jj].speeds[2]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[6]
                      - (cells[ii * params.nx + jj].speeds[4]
                         + cells[ii * params.nx + jj].speeds[7]
                         + cells[ii * params.nx + jj].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }
 return tot_u / (float)tot_cells;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.0;  /* accumulator */

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii * params.nx + jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = C_SQ; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
    	// compute the current position of the grid just once
    	int pos = ii * params.nx + jj;
      /* an occupied cell */
      if (obstacles[pos])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
	{
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[pos].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[pos].speeds[1]
               + cells[pos].speeds[5]
               + cells[pos].speeds[8]
               - (cells[pos].speeds[3]
                  + cells[pos].speeds[6]
                  + cells[pos].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[pos].speeds[2]
               + cells[pos].speeds[5]
               + cells[pos].speeds[6]
               - (cells[pos].speeds[4]
                  + cells[pos].speeds[7]
                  + cells[pos].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

int getTotCells(int* obstacles, int NY, int NX){
  int count = 0;
  for(int ii = 0; ii < NY; ii++)
    for(int jj = 0; jj < NX; jj++)
      if(!obstacles[ii * NX + jj])
	count++;
  
  return count;
}

int calc_nrows_from_rank_unfair(int rank, int size, int nx){
  int cols = nx / size;
  int remainder = nx % size;
  if(remainder != 0)
    if(rank == size-1)
      cols += remainder;
  return cols;
}

int calc_nrows_from_rank_fair(int rank, int size, int ny){
	int rows = ny / size;
	int reminder = ny % size;
	if(reminder != 0)
		if(rank < reminder)
			rows++;
	return rows; 
}

void initLoc(t_param params,t_speed** grid,t_speed** tmp_grid, int** obst, int* obstacles, float** loc_vels, int nrows, int ncols, int rank, int len_procs[], int start[]){
  /* local grid */
  int gridSize = (nrows + 2) * ncols;
  *grid = (t_speed*) calloc(gridSize, sizeof(t_speed));
  
  if (*grid == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  /* 'helper' grid, used as scratch space */
  *tmp_grid = (t_speed*) calloc(gridSize, sizeof(t_speed));
  if (*tmp_grid == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  /* the map of obstacles */
  *obst = calloc( nrows * ncols, sizeof(int));
  if (*obst == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
  
  
  /* initialise densities */ 
  float w0 = params.density * 4.0 / 9.0;
  float w1 = params.density      / 9.0;
  float w2 = params.density      / 36.0;
  
  for (int ii = 1; ii < nrows + 1; ii++) {
    for (int jj = 0; jj < ncols; jj++)    {
      // compute the current position of the grid just once
      int pos = ii * ncols + jj;
      /* centre */
      (*grid)[pos].speeds[0] = w0;
      /* axis directions */
      (*grid)[pos].speeds[1] = w1;
      (*grid)[pos].speeds[2] = w1;
      (*grid)[pos].speeds[3] = w1;
      (*grid)[pos].speeds[4] = w1;
      /* diagonals */
      (*grid)[pos].speeds[5] = w2;
      (*grid)[pos].speeds[6] = w2;
      (*grid)[pos].speeds[7] = w2;
      (*grid)[pos].speeds[8] = w2;
    }
  }


  
  for(int i = 0; i < nrows; i++)
    for(int j = 0; j < ncols; j++){
      (*obst)[i * ncols + j] = obstacles[start[rank] + (i * ncols) + j];
    }

  *loc_vels = (float*)calloc(params.maxIters, sizeof(float));  
}
