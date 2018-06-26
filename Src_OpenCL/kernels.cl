#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;


constant float c_sq = 1.0 / 3.0; /* square of speed of sound */
constant float w0_ = 4.0 / 9.0;  /* weighting factor */
constant float w1_ = 1.0 / 9.0;  /* weighting factor */
constant float w2_ = 1.0 / 36.0; /* weighting factor */
constant float csq_csq = 2.0 * (1.0 / 3.0) * (1.0/3.0);
constant float csq2 = 2.0 * (1.0/3.0);

void reduce(local float* local_vels, global float* partial, global float* av_vels, int tt, int tot_cells);
void associative_reduce(local float* local_vel, global float* partial);
void commutative_associative_reduce(local float* local_vels, global float* partial);

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);
  int pos = ii * nx + jj;
  //cl_float* s = cells[pos].speeds;
  /* if the cell is not occupied and
  ** we don't send a negative density */
  //  if (!obstacles[pos])
  // {
    /* increase 'east-side' densities */
    cells[pos].speeds[1] += w1;
    cells[pos].speeds[5] += w2;
    cells[pos].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[pos].speeds[3] -= w1;
    cells[pos].speeds[6] -= w2;
    cells[pos].speeds[7] -= w2;
    //} 
  
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny, float omega)
{
  
  /* get column and row indices */
  int pos = get_global_id(0);
  int ii = pos / nx;
  int jj = pos % nx;
  //  cl_float* s = cells[pos].speeds;
  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii == ny - 1) ? 0 : (ii + 1); 
  int x_e = (jj == nx - 1) ? 0 : (jj + 1); 
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[pos].speeds[0] = cells[pos].speeds[0]; /* central cell, no movement */
  tmp_cells[pos].speeds[1] = cells[ii * nx + x_w].speeds[1]; /* east */
  tmp_cells[pos].speeds[2] = cells[y_s * nx + jj ].speeds[2]; /* north */
  tmp_cells[pos].speeds[3] = cells[ii * nx + x_e].speeds[3]; /* west */
  tmp_cells[pos].speeds[4] = cells[y_n * nx + jj ].speeds[4]; /* south */
  tmp_cells[pos].speeds[5] = cells[y_s * nx + x_w].speeds[5]; /* north-east */
  tmp_cells[pos].speeds[6] = cells[y_s * nx + x_e].speeds[6]; /* north-west */
  tmp_cells[pos].speeds[7] = cells[y_n * nx + x_e].speeds[7]; /* south-west */
  tmp_cells[pos].speeds[8] = cells[y_n * nx + x_w].speeds[8]; /* south-east */  
}

// collision and rebound
kernel void collision(global t_speed* cells,
		      global t_speed* tmp_cells, 
		      global int* obstacles,
		      int nx, int ny,
		      float omega,
		      global float* av_vels,
		      local  float* local_vels,  // sizeof(float) * work_group_size
		      global float* partial,
		      int tot_cells,
		      int tt){
  float tot_u = 0; 
  /* get column and row indices */
  int pos = get_global_id(0);
  // int start = gid * 2;
  //int end = start + 2; 
    
  //for(int pos = start; pos < end; pos++){
  int ii = pos / nx;
  int jj = pos % nx;
 
  int y_n = (ii == ny - 1) ? 0 : (ii + 1); 
  int x_e = (jj == nx - 1) ? 0 : (jj + 1); 
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  
  float speed_0 = cells[pos].speeds[0]; /* central cell, no movement */
  float speed_1 = cells[ii * nx + x_w].speeds[1]; /* east */
  float speed_2 = cells[y_s * nx + jj ].speeds[2]; /* north */
  float speed_3 = cells[ii * nx + x_e].speeds[3]; /* west */
  float speed_4 = cells[y_n * nx + jj ].speeds[4]; /* south */
  float speed_5 = cells[y_s * nx + x_w].speeds[5]; /* north-east */
  float speed_6 = cells[y_s * nx + x_e].speeds[6]; /* north-west */
  float speed_7 = cells[y_n * nx + x_e].speeds[7]; /* south-west */
  float speed_8 = cells[y_n * nx + x_w].speeds[8]; /* south-east */ 

  
  /* don't consider occupied cells */
  if (!obstacles[pos]){
    /* compute local density total */
    float local_density =  speed_0 + speed_1 + speed_2 + speed_3 + speed_4 + speed_5 + speed_6 + speed_7 + speed_8;
    
    /* compute x velocity component */
    float u_x = (speed_1 + speed_5 + speed_8 - (speed_3 + speed_6 + speed_7)) / local_density;
    /* compute y velocity component */
    float u_y = (speed_2 + speed_5 + speed_6 - (speed_4 + speed_7 + speed_8)) / local_density;
    
    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;
    tot_u += sqrt(u_sq); 
    /* directional velocity components */
    
    /*relaxation step*/
    tmp_cells[pos].speeds[0] = speed_0 + omega * (w0_ * local_density * (1.0 - u_sq / csq2) - speed_0);
    tmp_cells[pos].speeds[1] = speed_1 + omega * (w1_ * local_density * (1.0 + u_x / c_sq  + (u_x * u_x) / csq_csq - u_sq / csq2) - speed_1);
    tmp_cells[pos].speeds[2] = speed_2 + omega * (w1_ * local_density * (1.0 + u_y / c_sq  + (u_y * u_y) / csq_csq - u_sq / csq2) - speed_2);
    tmp_cells[pos].speeds[3] = speed_3 + omega * (w1_ * local_density * (1.0 - u_x / c_sq  + (u_x * u_x) / csq_csq - u_sq / csq2) - speed_3);
    tmp_cells[pos].speeds[4] = speed_4 + omega * (w1_ * local_density * (1.0 - u_y / c_sq  + (u_y * u_y) / csq_csq - u_sq / csq2) - speed_4);
    tmp_cells[pos].speeds[5] = speed_5 + omega * ( w2_ * local_density * (1.0 + (u_x + u_y) / c_sq  + ((u_x + u_y) * (u_x + u_y)) / csq_csq - u_sq / csq2) - speed_5);
    tmp_cells[pos].speeds[6] = speed_6 + omega * ( w2_ * local_density * (1.0 + (-u_x + u_y) / c_sq  + ((-u_x+u_y) * (-u_x+u_y)) / csq_csq - u_sq / csq2) - speed_6);
    tmp_cells[pos].speeds[7] = speed_7 + omega * (w2_ * local_density * (1.0 + (-u_x-u_y) / c_sq  + ((-u_x-u_y) * (-u_x-u_y)) / csq_csq - u_sq / csq2) - speed_7);
    tmp_cells[pos].speeds[8] = speed_8 + omega * (w2_ * local_density * (1.0 + (u_x-u_y) / c_sq  + ((u_x-u_y) * (u_x-u_y)) / csq_csq - u_sq / csq2) - speed_8);
  } else {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    //REBOUND
    tmp_cells[pos].speeds[1] = speed_3;
    tmp_cells[pos].speeds[2] = speed_4;
    tmp_cells[pos].speeds[3] = speed_1;
    tmp_cells[pos].speeds[4] = speed_2;
    tmp_cells[pos].speeds[5] = speed_7;
    tmp_cells[pos].speeds[6] = speed_8;
    tmp_cells[pos].speeds[7] = speed_5;
    tmp_cells[pos].speeds[8] = speed_6;
  }  
  // }
  
  local_vels[get_local_id(0)] = (float) tot_u;
  barrier(CLK_LOCAL_MEM_FENCE);
  // reduce(local_vels, partial, av_vels, tt, tot_cells);
  associative_reduce(local_vels, partial);
  //commutative_associative_reduce(local_vels, partial);
}

kernel void av_velocity(global t_speed* cells, 
			global t_speed* tmp_cells, 
			global int* obstacles,
			global float* av_vels,
			local  float* local_vels,  // sizeof(float) * work_group_size
			global float* partial,
			int nx,
			int ny, 
			int tot_cells,
			int tt)
{
  float tot_u = 0;          /* accumulated magnitudes of velocity for each cell */
  
  /* get column and row indices */
  int pos = get_global_id(0);
  /* ignore occupied cells */
  if (!obstacles[pos]){
    /* local density total */
    float local_density = 0.0;
    
    for (int kk = 0; kk < NSPEEDS; kk++){
      local_density += cells[pos].speeds[kk];
    }
    
    /* x-component of velocity */
    float u_x = (cells[pos].speeds[1]
		  + cells[pos].speeds[5]
		  + cells[pos].speeds[8]
		  - (cells[pos].speeds[3]
		     + cells[pos].speeds[6]
		     + cells[pos].speeds[7]))
      / local_density;
    /* compute y velocity component */
    float u_y = (cells[pos].speeds[2]
		  + cells[pos].speeds[5]
		  + cells[pos].speeds[6]
		  - (cells[pos].speeds[4]
		     + cells[pos].speeds[7]
		     + cells[pos].speeds[8]))
      / local_density;
    /* accumulate the norm of x- and y- velocity components */
    tot_u += sqrt((u_x * u_x) + (u_y * u_y));
  }
  
  local_vels[get_local_id(0)] = (float) tot_u;
  barrier(CLK_LOCAL_MEM_FENCE);
  //reduce(local_vels, partial, av_vels, tt, tot_cells);
  associative_reduce(local_vels, partial);
}

void reduce(local float* local_vels, global float* partial, global float* av_vels, int tt, int tot_cells){                     
  int num_wrk_items  = get_local_size(0);                 
  int local_id       = get_local_id(0);                   
  int group_id       = get_group_id(0);                    
 
  int i;  
  float sum;
  //this should be both on host and devicefloat* partial = malloc(sizeof(float) * get_num_groups());
  
  if(local_id == 0){
    sum = 0;
    // reduce local sums
    for(i = 0; i < num_wrk_items; i++)
      sum += local_vels[i];

    partial[group_id] = sum; 
  }

  // av_vels[tt] = NOT HERE
  // then lounch another kernel to reduce the partial sums of each group, because you can't synchronize between groups. at the beginning do it at each time step.
  // - try reduction on the device
  // - try reduction on the host
  // - try big reduction at the very end, cause you only need the av_vels values when computing reynolds?

}

void associative_reduce(local float* local_vels,
			global float* partial) {
  
  int num_wrk_items  = get_local_size(0);    
  // int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  
  for(int offset = 1; offset < num_wrk_items; offset <<= 1) {
    int mask = (offset << 1) - 1;
    if ((local_index & mask) == 0) 
      local_vels[local_index] += local_vels[local_index + offset]; 
    barrier(CLK_LOCAL_MEM_FENCE);
  }
 
  if (local_index == 0) {
    partial[get_group_id(0)] = local_vels[0];
  }
}

void commutative_associative_reduce(local float* local_vels,
				    global float* partial) {
  
  int num_wrk_items  = get_local_size(0);    
  //int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  
  for(int offset = num_wrk_items / 2; offset > 0; offset >>= 1) {
    if (local_index < offset)
      local_vels[local_index] += local_vels[local_index + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if (local_index == 0) {
    partial[get_group_id(0)] = local_vels[0];
  }
  
}


kernel void rebound(global t_speed* cells,
                    global t_speed* tmp_cells,
                    global int* obstacles,
                    int nx, int ny)
{
  /* get column and row indices */
  int pos = get_global_id(0);
  //cl_float* s = cells[pos].speeds;
  //cl_float* ts = tmp_cells[pos].speeds;
  
  /* if the cell contains an obstacle */
  if (obstacles[pos]) {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    cells[pos].speeds[1] = tmp_cells[pos].speeds[3];
    cells[pos].speeds[2] = tmp_cells[pos].speeds[4];
    cells[pos].speeds[3] = tmp_cells[pos].speeds[1];
    cells[pos].speeds[4] = tmp_cells[pos].speeds[2];
    cells[pos].speeds[5] = tmp_cells[pos].speeds[7];
    cells[pos].speeds[6] = tmp_cells[pos].speeds[8];
    cells[pos].speeds[7] = tmp_cells[pos].speeds[5];
    cells[pos].speeds[8] = tmp_cells[pos].speeds[6];
  } 
}
