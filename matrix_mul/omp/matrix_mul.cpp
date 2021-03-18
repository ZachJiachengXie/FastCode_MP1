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
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
    unsigned int block_size, ii, jj, kk, i, j, k;
    // Initialize result matrix to be all 0's
    memset(sq_matrix_result, 0, sizeof(float) * sq_dimension * sq_dimension);
    block_size = 16;
#pragma omp parallel for private(ii, jj, kk, i, j, k)
    for(i = 0; i < sq_dimension; i += block_size)
    {
      for(k = 0; k < sq_dimension; k += block_size)
      {
          for(j = 0; j < sq_dimension; j += block_size)
          {
              for(ii = i; ii < min(sq_dimension, i + block_size); ++ii)
              {
                for(kk = k; kk < min(sq_dimension, k + block_size); ++kk)
                {
                    for(jj = j; jj < min(sq_dimension, j + block_size); ++jj)
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
  matrix_multiplication_b16_ijk(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
    unsigned int block_size, ii, jj, kk, i, j, k;
    // Initialize result matrix to be all 0's
    memset(sq_matrix_result, 0, sizeof(float) * sq_dimension * sq_dimension);
    block_size = 16;
#pragma omp parallel for private(ii, jj, kk, i, j, k)
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
