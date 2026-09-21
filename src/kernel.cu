#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#ifndef FLOCKING_BLOCK_SIZE
#define FLOCKING_BLOCK_SIZE 128
#endif
#define blockSize FLOCKING_BLOCK_SIZE

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_pos_coherent;
glm::vec3 *dev_vel1_coherent;
glm::vec3 *dev_vel2_coherent;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize); // computing the number of blocks needed given N and blockSize.

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects,
    dev_vel1, 0.1f);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  cudaMemcpy(dev_vel2, dev_vel1, sizeof(glm::vec3) * N, cudaMemcpyDeviceToDevice);

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  cudaMalloc((void**)&dev_vel1_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_coherent failed!");

  cudaMalloc((void**)&dev_vel2_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2_coherent failed!");

  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) { // VBO expects 4 floats per boid, padding the 4th component with 1.0f 
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  const glm::vec3 selfPos = pos[iSelf];

  const float rule1DistanceSq = rule1Distance * rule1Distance;
  const float rule2DistanceSq = rule2Distance * rule2Distance;
  const float rule3DistanceSq = rule3Distance * rule3Distance;

  // Rule 1: cohesion
  int rule1NeighborCount = 0;
  glm::vec3 perceived_center(0.0f);

  // Rule 2: separation
  glm::vec3 positionChangeAccumulate(0.0f);

  // Rule 3: alignment
  int rule3NeighborCount = 0;
  glm::vec3 perceived_velocity(0.0f);

  for (int i = 0; i < N; i++) {
    if (i == iSelf) {
      continue;
    }

    const glm::vec3 boidPos = pos[i];
    const glm::vec3 offset = boidPos - selfPos;
    const float distSq = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;

    if (distSq < rule1DistanceSq) {
      rule1NeighborCount++;
      perceived_center += boidPos;
    }

    if (distSq < rule2DistanceSq) {
      positionChangeAccumulate -= offset;
    }

    if (distSq < rule3DistanceSq) {
      rule3NeighborCount++;
      perceived_velocity += vel[i];
    }
  }

  glm::vec3 rule1VelocityChange(0.0f);
  if (rule1NeighborCount > 0) {
    perceived_center /= static_cast<float>(rule1NeighborCount); // make sure the result is float
    rule1VelocityChange = (perceived_center - selfPos) * rule1Scale;
  }

  const glm::vec3 rule2VelocityChange = positionChangeAccumulate * rule2Scale;

  glm::vec3 rule3VelocityChange(0.0f);
  if (rule3NeighborCount > 0) {
    perceived_velocity /= static_cast<float>(rule3NeighborCount);
    rule3VelocityChange = perceived_velocity * rule3Scale;
  }

  return rule1VelocityChange + rule2VelocityChange + rule3VelocityChange;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index >= N) {
    return;
  }

  glm::vec3 newVel = vel1[index];
  newVel += computeVelocityChange(N, index, pos, vel1);
  // Clamp the speed
  const float maxSpeedSq = maxSpeed * maxSpeed;
  const float speedSq = newVel.x * newVel.x + newVel.y * newVel.y + newVel.z * newVel.z;
  if (speedSq > maxSpeedSq && speedSq > 0.0f) {
    const float scale = maxSpeed * rsqrtf(speedSq);
    newVel *= scale;
  }
  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[index] = newVel; 
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index >= N) {
      return;
    }

    // compute cell of current boid
    const glm::vec3 selfPos = pos[index];
    int cellX = imax((int)floor((selfPos.x - gridMin.x) * inverseCellWidth), 0);
    int cellY = imax((int)floor((selfPos.y - gridMin.y) * inverseCellWidth), 0);
    int cellZ = imax((int)floor((selfPos.z - gridMin.z) * inverseCellWidth), 0);

    // clamp
    cellX = imin(cellX, gridResolution - 1);
    cellY = imin(cellY, gridResolution - 1);
    cellZ = imin(cellZ, gridResolution - 1);

    // 3d to 1d
    int cellId = gridIndex3Dto1D(cellX, cellY, cellZ, gridResolution); 

    gridIndices[index] = cellId;
    indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(i >= N)  
    return;

  if(i == 0 || particleGridIndices[i] != particleGridIndices[i-1]) {
    gridCellStartIndices[particleGridIndices[i]] = i;
  }
  if(i == N-1 || particleGridIndices[i] != particleGridIndices[i+1]) {
    gridCellEndIndices[particleGridIndices[i]] = i+1;
  }

}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index >= N)
    return;
  
    // compute cell of current boid
    const glm::vec3 selfPos = pos[index];
    int cellX = imax((int)floor((selfPos.x - gridMin.x) * inverseCellWidth), 0);
    int cellY = imax((int)floor((selfPos.y - gridMin.y) * inverseCellWidth), 0);
    int cellZ = imax((int)floor((selfPos.z - gridMin.z) * inverseCellWidth), 0);

    // clamp
    cellX = imin(cellX, gridResolution - 1);
    cellY = imin(cellY, gridResolution - 1);
    cellZ = imin(cellZ, gridResolution - 1);

    
  // - Identify which cells may contain neighbors. This isn't always 8.
  const float rule1DistanceSq = rule1Distance * rule1Distance;
  const float rule2DistanceSq = rule2Distance * rule2Distance;
  const float rule3DistanceSq = rule3Distance * rule3Distance;

  // Rule 1: cohesion
  int rule1NeighborCount = 0;
  glm::vec3 perceived_center(0.0f);

  // Rule 2: separation
  glm::vec3 positionChangeAccumulate(0.0f);

  // Rule 3: alignment
  int rule3NeighborCount = 0;
  glm::vec3 perceived_velocity(0.0f);

  // - For each cell, read the start/end indices in the boid pointer array.
  for(int dz = -1; dz <= 1; dz++) {
    int nz = dz + cellZ;
    if(nz < 0 || nz >= gridResolution)
      continue;
    for(int dy = -1; dy <= 1; dy++) {
      int ny = cellY + dy;
      if(ny < 0 || ny >= gridResolution)
        continue;
      for(int dx = -1; dx <= 1; dx++){
        int nx = cellX + dx;
        if(nx < 0 || nx >= gridResolution)
          continue;
        
        // 3d to 1d
        int cellId = gridIndex3Dto1D(nx, ny, nz, gridResolution);
        int start = gridCellStartIndices[cellId];
        int end = gridCellEndIndices[cellId];

        if(start == -1 || end == -1 || start >= end)
          continue;
        
        // - Access each boid in the cell and compute velocity change from
        //   the boids rules, if this boid is within the neighborhood distance.
        for(int j = start; j < end; j++)
        {
          int iOther = particleArrayIndices[j];
          if(iOther == index) 
            continue;
          
          glm::vec3 boidPos = pos[iOther];
          glm::vec3 offset = boidPos - selfPos;
          const float distSq = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;

          if (distSq < rule1DistanceSq) {
            rule1NeighborCount++;
            perceived_center += boidPos;
          }

          if (distSq < rule2DistanceSq) {
            positionChangeAccumulate -= offset;
          }

          if (distSq < rule3DistanceSq) {
            rule3NeighborCount++;
            perceived_velocity += vel1[iOther];
          }
        }
      }


    }
  }
  
  glm::vec3 rule1VelocityChange(0.0f);
  if (rule1NeighborCount > 0) {
    perceived_center /= static_cast<float>(rule1NeighborCount); // make sure the result is float
    rule1VelocityChange = (perceived_center - selfPos) * rule1Scale;
  }

  const glm::vec3 rule2VelocityChange = positionChangeAccumulate * rule2Scale;

  glm::vec3 rule3VelocityChange(0.0f);
  if (rule3NeighborCount > 0) {
    perceived_velocity /= static_cast<float>(rule3NeighborCount);
    rule3VelocityChange = perceived_velocity * rule3Scale;
  }

  glm::vec3 newVel = vel1[index] + rule1VelocityChange + rule2VelocityChange + rule3VelocityChange;
  
  // Clamp the speed
  const float maxSpeedSq = maxSpeed * maxSpeed;
  const float speedSq = newVel.x * newVel.x + newVel.y * newVel.y + newVel.z * newVel.z;
  if (speedSq > maxSpeedSq && speedSq > 0.0f) {
    const float scale = maxSpeed * rsqrtf(speedSq);
    newVel *= scale;
  }
  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[index] = newVel;
}

// TODO-2.3 helper function to rerange data
__global__ void kernRearrangeData(int N, int *particleArrayIndices, glm::vec3* pos, glm::vec3* vel, 
  glm::vec3* pos_coherent, glm::vec3* vel_coherent) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if(index >= N)
      return;

    int realIndex = particleArrayIndices[index];

    pos_coherent[index] = pos[realIndex];
    vel_coherent[index] = vel[realIndex];
  }




__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index >= N)
    return;
  
  // compute cell of current boid
  const glm::vec3 selfPos = pos[index];
  int cellX = imax((int)floor((selfPos.x - gridMin.x) * inverseCellWidth), 0);
  int cellY = imax((int)floor((selfPos.y - gridMin.y) * inverseCellWidth), 0);
  int cellZ = imax((int)floor((selfPos.z - gridMin.z) * inverseCellWidth), 0);
  
  // clamp
  cellX = imin(cellX, gridResolution - 1);
  cellY = imin(cellY, gridResolution - 1);
  cellZ = imin(cellZ, gridResolution - 1);

    
  // - Identify which cells may contain neighbors. This isn't always 8.
  const float rule1DistanceSq = rule1Distance * rule1Distance;
  const float rule2DistanceSq = rule2Distance * rule2Distance;
  const float rule3DistanceSq = rule3Distance * rule3Distance;

  // Rule 1: cohesion
  int rule1NeighborCount = 0;
  glm::vec3 perceived_center(0.0f);

  // Rule 2: separation
  glm::vec3 positionChangeAccumulate(0.0f);

  // Rule 3: alignment
  int rule3NeighborCount = 0;
  glm::vec3 perceived_velocity(0.0f);


  // - For each cell, read the start/end indices in the boid pointer array.
  for(int dz = -1; dz <= 1; dz++) {
    int nz = dz + cellZ;
    if(nz < 0 || nz >= gridResolution)
      continue;
    for(int dy = -1; dy <= 1; dy++) {
      int ny = cellY + dy;
      if(ny < 0 || ny >= gridResolution)
        continue;
      for(int dx = -1; dx <= 1; dx++){
        int nx = cellX + dx;
        if(nx < 0 || nx >= gridResolution)
          continue;
        
        // 3d to 1d
        int cellId = gridIndex3Dto1D(nx, ny, nz, gridResolution);
        int start = gridCellStartIndices[cellId];
        int end = gridCellEndIndices[cellId];

        if(start == -1 || end == -1 || start >= end)
          continue;
        
        // - Access each boid in the cell and compute velocity change from
        //   the boids rules, if this boid is within the neighborhood distance.
        for(int j = start; j < end; j++)
        {
          if(j == index)
            continue;
          
          glm::vec3 boidPos = pos[j];
          glm::vec3 offset = boidPos - selfPos;
          const float distSq = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;

          if (distSq < rule1DistanceSq) {
            rule1NeighborCount++;
            perceived_center += boidPos;
          }

          if (distSq < rule2DistanceSq) {
            positionChangeAccumulate -= offset;
          }

          if (distSq < rule3DistanceSq) {
            rule3NeighborCount++;
            perceived_velocity += vel1[j];
          }
        }
      }


    }
  }


  glm::vec3 rule1VelocityChange(0.0f);
  if (rule1NeighborCount > 0) {
    perceived_center /= static_cast<float>(rule1NeighborCount); // make sure the result is float
    rule1VelocityChange = (perceived_center - selfPos) * rule1Scale;
  }

  const glm::vec3 rule2VelocityChange = positionChangeAccumulate * rule2Scale;

  glm::vec3 rule3VelocityChange(0.0f);
  if (rule3NeighborCount > 0) {
    perceived_velocity /= static_cast<float>(rule3NeighborCount);
    rule3VelocityChange = perceived_velocity * rule3Scale;
  }

  glm::vec3 newVel = vel1[index] + rule1VelocityChange + rule2VelocityChange + rule3VelocityChange;
  
  // Clamp the speed
  const float maxSpeedSq = maxSpeed * maxSpeed;
  const float speedSq = newVel.x * newVel.x + newVel.y * newVel.y + newVel.z * newVel.z;
  if (speedSq > maxSpeedSq && speedSq > 0.0f) {
    const float scale = maxSpeed * rsqrtf(speedSq);
    newVel *= scale;
  }
  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[index] = newVel;


}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("UpdateVelocityBruteForce failed!");
  kernUpdatePos<<<fullBlocksPerGrid, blockSize >>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("UpdatePos failed!");
  // TODO-1.2 ping-pong the velocity buffers
  glm::vec3* tmp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp;

}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // - label each particle with its array index as well as its grid index.
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, 
  gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  
  // sort based on grid index, so that boids in the same cell are adjacent in the array
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // Initialize the grid cell start and end indices to be -1, which represent no boids in the cell
  dim3 gridFullBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);
  kernResetIntBuffer<<<gridFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<gridFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);

  // scan through the sorted boid grid indices and identify the start and end of each cell's data pointers in the array of boid indices
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices,
  dev_gridCellStartIndices, dev_gridCellEndIndices);

  // - Perform velocity updates using neighbor search
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
  dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // - Update positions
  kernUpdatePos<<<fullBlocksPerGrid, blockSize >>>(numObjects, dt, dev_pos, dev_vel2);

  // - Ping-pong buffers as needed
  glm::vec3* tmpVel = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmpVel;
  
}

void Boids::stepSimulationCoherentGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // - label each particle with its array index as well as its grid index.
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, 
  gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  
  // sort based on grid index, so that boids in the same cell are adjacent in the array
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // rerange data
  kernRearrangeData<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices, dev_pos, 
  dev_vel1, dev_pos_coherent, dev_vel1_coherent);

  // Initialize the grid cell start and end indices to be -1, which represent no boids in the cell
  dim3 gridFullBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);
  kernResetIntBuffer<<<gridFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<gridFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);

  // scan through the sorted boid grid indices and identify the start and end of each cell's data pointers in the array of boid indices
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices,
  dev_gridCellStartIndices, dev_gridCellEndIndices);

  // - Perform velocity updates using neighbor search
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
  dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos_coherent, dev_vel1_coherent, dev_vel2_coherent);

  // - Update positions
  kernUpdatePos<<<fullBlocksPerGrid, blockSize >>>(numObjects, dt, dev_pos_coherent, dev_vel2_coherent);

  // - Ping-pong buffers as needed
  {
    glm::vec3* tmpPos = dev_pos;
    dev_pos = dev_pos_coherent;
    dev_pos_coherent = tmpPos;
  }

  {
    glm::vec3* tmpVel = dev_vel1;
    dev_vel1 = dev_vel2_coherent;
    dev_vel2_coherent = tmpVel;
  }
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel1_coherent);
  cudaFree(dev_vel2_coherent);

}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
