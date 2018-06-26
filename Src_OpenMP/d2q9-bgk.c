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
#include<omp.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

// some constant values used throught the program
#define C_SQ 			(1.0 / 3.0) /* square of speed of sound */
#define W0 				(4.0 / 9.0)  /* weighting factor */
#define W1 				(1.0 / 9.0)  /* weighting factor */
#define W2 				(1.0 / 36.0) /* weighting factor */
#define CSQ_2_CSQ 		(2.0 * C_SQ * C_SQ)
#define CSQ_2 			(2.0 * C_SQ)

// constant values determined at runtime
double DENSITY_ACCEL_D9; // params.density * params.accel / 9.0
double DENSITY_ACCEL_D36; // params.density * params.accel /9.0


/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  double speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr);

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
int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

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
  	double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  	struct timeval timstr;        /* structure to hold elapsed time */
  	struct rusage ru;             /* structure to hold CPU time--system and user */
  	double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  	double usrtim;                /* floating point number to record elapsed user CPU time */
  	double systim;                /* floating point number to record elapsed system CPU time */

  	/* parse the command line */
  	if (argc != 3){
    	usage(argv[0]);
  	} else{
    	paramfile = argv[1];
    	obstaclefile = argv[2];
  	}

  	/* initialise our data structures and load values from file */
  	initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  	/* iterate for maxIters timesteps */
  	gettimeofday(&timstr, NULL);
  	tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);


	int NX = params.nx;
	int NY = params.ny;	
	
	int    tot_cells = 0;  /* no. of cells used in calculation */
  	double tot_u;          /* accumulated magnitudes of velocity for each cell */
  	int secondRow = (NY - 2) * NX;
  	double omega = params.omega;
  	
	for (int tt = 0; tt < params.maxIters; tt++){	
  	#pragma omp parallel proc_bind(close), shared(tot_cells, tot_u) 
  	{								  		  	
  		//loop over _all_ cells 
		#pragma omp for schedule(static)
  		for (int ii = 0; ii < NY; ii++){
		#pragma omp simd
    		for (int jj = 0; jj < NX; jj++){
    			// compute the current position of the grid just once
			int pos = ii * NX  + jj;	    			
			double* s = cells[pos].speeds;
			
			// ACCELERATE FLOW
			if (ii*NX == secondRow)
 			if( !obstacles[pos] && (s[3] - DENSITY_ACCEL_D9) && (s[6]- DENSITY_ACCEL_D36) && (s[7] - DENSITY_ACCEL_D36)){ 
      			s[1] += DENSITY_ACCEL_D9; s[5] += DENSITY_ACCEL_D36; s[8] += DENSITY_ACCEL_D36; 
      			s[3] -= DENSITY_ACCEL_D9; s[6] -= DENSITY_ACCEL_D36; s[7] -= DENSITY_ACCEL_D36; 
			}
    	
    		// 	PROPAGATE
      			/* determine indices of axis-direction neighbours
      			** respecting periodic boundary conditions (wrap around) */
      			int y_n = (ii == NY-1) ? 0 : (ii+1); 
      			int x_e = (jj == NX-1) ? 0 : (jj+1); 
      			int y_s = (!ii) ? (ii + NY - 1) : (ii - 1);
      			int x_w = (!jj) ? (jj + NX - 1) : (jj - 1);
      			/* propagate densities to neighbouring cells, following
      			** appropriate directions of travel and writing into
      			** scratch space grid */
      			int iiNX = ii * NX; 
      			int ynNX = y_n * NX;
      			int ysNX = y_s * NX;
     	 		tmp_cells[iiNX + jj].speeds[0]  = s[0];  /* central cell, no movement */
      			tmp_cells[iiNX + x_e].speeds[1] = s[1];  /* east */
      			tmp_cells[ynNX + jj].speeds[2]  = s[2];  /* north */
      			tmp_cells[iiNX + x_w].speeds[3] = s[3];  /* west */
      			tmp_cells[ysNX + jj].speeds[4]  = s[4];  /* south */
      			tmp_cells[ynNX + x_e].speeds[5] = s[5];  /* north-east */
      			tmp_cells[ynNX + x_w].speeds[6] = s[6];  /* north-west */
      			tmp_cells[ysNX + x_w].speeds[7] = s[7];  /* south-west */
      			tmp_cells[ysNX + x_e].speeds[8] = s[8];  /* south-east */
    	}
  	}
  	
  	/* initialise */
  	tot_cells = 0;
  	tot_u = 0.0;
  	
  	// COLLISION
  	
  	/* loop over the cells in the grid
  	** NB the collision step is called after
  	** the propagate step and so values of interest
  	** are in the scratch-space grid */
  	#pragma omp for reduction(+:tot_u,tot_cells) schedule(static) nowait
  	for (int ii = 0; ii < NY; ii++){ 
  	int part = ii * NX;
  	#pragma simd 
    	for (int jj = 0; jj < NX; jj++){
    		// compute the current position of the grid just once
    		int pos = part + jj;
    	   	double* ts = tmp_cells[pos].speeds;
    	   	double* s = cells[pos].speeds;
      		/* don't consider occupied cells */
      		if (!obstacles[pos]){
        		/* compute local density total */
        		double local_density = ts[0] + ts[1] + ts[2] +  ts[3] + ts[4] + ts[5] + ts[6] + ts[7] + ts[8];

        		/* compute x velocity component */
        		double u_x = (ts[1]+ ts[5] + ts[8] - (ts[3] + ts[6] + ts[7])) / local_density;
        		/* compute y velocity component */
        		double u_y = (ts[2] + ts[5] + ts[6] - (ts[4] + ts[7] + ts[8])) / local_density;

        		/* velocity squared */
        		double u_sq = u_x * u_x + u_y * u_y;
		
        		/* directional velocity components */
        		double u[NSPEEDS];
        		u[1] =   u_x;        /* east */
        		u[2] =         u_y;  /* north */
       	 		u[3] = - u_x;        /* west */
        		u[4] =       - u_y;  /* south */
        		u[5] =   u_x + u_y;  /* north-east */
        		u[6] = - u_x + u_y;  /* north-west */
        		u[7] = - u_x - u_y;  /* south-west */
        		u[8] =   u_x - u_y;  /* south-east */

       			/* equilibrium densities */
        		double d_equ[NSPEEDS];
        		/* zero velocity density: weight w0 */
        		d_equ[0] = W0 * local_density * (1.0 - u_sq / (CSQ_2));
        		/* axis speeds: weight w1 */
        		double partial = W1 * local_density;
        		double par = u_sq / (CSQ_2);
        		d_equ[1] = partial * (1.0 + u[1] / C_SQ + (u[1] * u[1]) / (CSQ_2_CSQ) - par);
        		d_equ[2] = partial * (1.0 + u[2] / C_SQ + (u[2] * u[2]) / (CSQ_2_CSQ) - par);
        		d_equ[3] = partial * (1.0 + u[3] / C_SQ + (u[3] * u[3]) / (CSQ_2_CSQ) - par);
        		d_equ[4] = partial * (1.0 + u[4] / C_SQ + (u[4] * u[4]) / (CSQ_2_CSQ) - par);
        		/* diagonal speeds: weight w2 */
        		partial = W2 * local_density;
        		d_equ[5] = partial * (1.0 + u[5] / C_SQ + (u[5] * u[5]) / (CSQ_2_CSQ) - par);
        		d_equ[6] = partial * (1.0 + u[6] / C_SQ + (u[6] * u[6]) / (CSQ_2_CSQ) - par);
        		d_equ[7] = partial * (1.0 + u[7] / C_SQ + (u[7] * u[7]) / (CSQ_2_CSQ) - par);
        		d_equ[8] = partial * (1.0 + u[8] / C_SQ + (u[8] * u[8]) / (CSQ_2_CSQ) - par);
            
        		/* relaxation step */
       			s[0] = ts[0] + omega * (d_equ[0] - ts[0]);
       			s[1] = ts[1] + omega * (d_equ[1] - ts[1]);
       			s[2] = ts[2] + omega * (d_equ[2] - ts[2]);
       			s[3] = ts[3] + omega * (d_equ[3] - ts[3]);
       			s[4] = ts[4] + omega * (d_equ[4] - ts[4]);
       			s[5] = ts[5] + omega * (d_equ[5] - ts[5]);
       			s[6] = ts[6] + omega * (d_equ[6] - ts[6]);
       			s[7] = ts[7] + omega * (d_equ[7] - ts[7]);
       			s[8] = ts[8] + omega * (d_equ[8] - ts[8]);
       			
       			// AVERAGE VELOCITY
       			
       			/* accumulate the norm of x- and y- velocity components */
        		tot_u += sqrt(u_sq);
        		/* increase counter of inspected cells */
        		++(tot_cells);
      	} else {
      		// REBOUND
      		
      		/* called after propagate, so taking values from scratch space
        		** mirroring, and writing into main grid */
        	s[1] = ts[3]; s[2] = ts[4]; s[3] = ts[1]; s[4] = ts[2]; 
        	s[5] = ts[7]; s[6] = ts[8]; s[7] = ts[5]; s[8] = ts[6]; 
      	}
   	 }	
	}   

}
	av_vels[tt] = tot_u / (double)tot_cells;
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	
	#ifdef DEBUG
	for (int tt = 0; tt < params.maxIters; tt++){
		printf("==timestep: %d==\n", tt);
    		printf("av velocity: %.12E\n", av_vels[tt]);
    		printf("tot density: %.12E\n", total_density(params, cells));
	}
	#endif


  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}




int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr)
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

  retval = fscanf(fp, "%lf\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->omega));

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
  double w0 = params->density * 4.0 / 9.0;
  double w1 = params->density      / 9.0;
  double w2 = params->density      / 36.0;
	DENSITY_ACCEL_D9 = params->density * params->accel / 9.0;
	DENSITY_ACCEL_D36 = params->density * params->accel / 36.0;
	
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
    	// compute the current position of the grid just once
    	int pos = ii * params->nx + jj;
      /* centre */
      (*cells_ptr)[pos].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[pos].speeds[1] = w1;
      (*cells_ptr)[pos].speeds[2] = w1;
      (*cells_ptr)[pos].speeds[3] = w1;
      (*cells_ptr)[pos].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[pos].speeds[5] = w2;
      (*cells_ptr)[pos].speeds[6] = w2;
      (*cells_ptr)[pos].speeds[7] = w2;
      (*cells_ptr)[pos].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  // I can rid of this if I use calloc, will it be faster?
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
  *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr)
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

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

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
        double local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* x-component of velocity */
        double u_x = (cells[ii * params.nx + jj].speeds[1]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[8]
                      - (cells[ii * params.nx + jj].speeds[3]
                         + cells[ii * params.nx + jj].speeds[6]
                         + cells[ii * params.nx + jj].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        double u_y = (cells[ii * params.nx + jj].speeds[2]
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
 return tot_u / (double)tot_cells;
}

double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
  double total = 0.0;  /* accumulator */

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

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
  FILE* fp;                     /* file pointer */
  const double c_sq = C_SQ; /* sq. of speed of sound */
  double local_density;         /* per grid cell sum of densities */
  double pressure;              /* fluid pressure in grid cell */
  double u_x;                   /* x-component of velocity in grid cell */
  double u_y;                   /* y-component of velocity in grid cell */
  double u;                     /* norm--root of summed squares--of u_x and u_y */

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

