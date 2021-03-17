/*

    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <omp.h>
#include "matrix_mul.h"
#include <string.h>

namespace omp
{
    
  unsigned int min(unsigned int A, unsigned int B) {
    if (A < B)
    {
      return A;
    }
    return B;
  }

    void
  matrix_multiplication_kij(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
    memset(sq_matrix_result, 0, sizeof(float) * sq_dimension * sq_dimension);
    float temp;
#pragma omp parallel for
    for (unsigned int k = 0; k < sq_dimension; k++) 
      {
  for(unsigned int i = 0; i < sq_dimension; i++) 
    {
      temp = sq_matrix_1[i*sq_dimension + k];
      for (unsigned int j = 0; j < sq_dimension; j++)
        sq_matrix_result[i*sq_dimension + j] +=  temp * sq_matrix_2[k*sq_dimension + j];
    }
      }// End of parallel region
  
  }
  void
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
    unsigned int block_size, ii, jj, kk, i, j, k, sum;
    memset(sq_matrix_result, 0, sizeof(float) * sq_dimension * sq_dimension);
    block_size = 4;
#pragma omp parallel for private(ii, jj, kk, i, j, k, sum)
    for(i = 0; i < sq_dimension; i += block_size)
    {
      for(j = 0; j < sq_dimension; j += block_size)
      {
          for(k = 0; k < sq_dimension; k += block_size)
          {
              for(ii = i; ii < min(sq_dimension, i + block_size); ++ii)
              {
                  for(jj = j; jj < min(sq_dimension, j + block_size); ++jj)
                  {
                      for(kk = k; kk < min(sq_dimension, k + block_size); ++kk)
                      {
                        sq_matrix_result[ii * sq_dimension + jj] += sq_matrix_1[ii * sq_dimension + kk] * sq_matrix_2[kk * sq_dimension + jj];
                      }
                  }
              }               
          }
      }
    }// End of parallel region
  }

  void
  matrix_multiplication_block(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
    unsigned int block_size, block_upperbound, ii, jj, kk, i, j, k;
    memset(sq_matrix_result, 0, sizeof(float) * sq_dimension * sq_dimension);
    block_size = 4;
    block_upperbound = (sq_dimension / block_size) * block_size;
#pragma omp parallel for private(ii, jj, kk, i, j, k)
    for(ii = 0; ii < block_upperbound; ii += block_size)
    {
      for(jj = 0; jj < block_upperbound; jj += block_size)
      {
          for(kk = 0; kk < block_upperbound; kk += block_size)
          {
              for(i=ii; i < min(sq_dimension, ii + block_size); ++i)
              {
                  for(j=jj; j < min(sq_dimension, jj + block_size); ++j)
                  {
                      register unsigned int sum = 0;
                      for(k=kk; k<min(sq_dimension, kk + block_size); ++k)
                      {
                        sum += sq_matrix_1[i * sq_dimension + k] * sq_matrix_2[k * sq_dimension + j];
                      } 
                      sq_matrix_result[i * sq_dimension + j] += sum;
                  }
              }               
          }
      }
    }// End of parallel region
  }
  void
  matrix_multiplication_baseline(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
#pragma omp parallel for
    for (unsigned int i = 0; i < sq_dimension; i++) 
      {
  for(unsigned int j = 0; j < sq_dimension; j++) 
    {       
      sq_matrix_result[i*sq_dimension + j] = 0;
      for (unsigned int k = 0; k < sq_dimension; k++)
        sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
    }
      }// End of parallel region
  }
  
} //namespace omp
