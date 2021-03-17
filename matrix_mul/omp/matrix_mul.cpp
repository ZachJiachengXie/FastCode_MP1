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
  matrix_multiplication_block(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
    unsigned int block_size;
    memset(sq_matrix_result, 0, sizeof(float) * sq_dimension * sq_dimension);
    block_size = 4;
#pragma omp parallel for

    for(int ii = 0; ii < sq_dimension; ii += block_size)
    {
      for(int jj = 0; jj < sq_dimension; jj += block_size)
      {
          for(int kk = 0; kk < sq_dimension; kk += block_size)
          {
              for(int i=ii; i < ii + block_size; ++i)
              {
                  for(int j=jj; j < jj + block_size; ++j)
                  {
                      register unsigned int sum = 0;
                      for(int k=kk; k<kk+block_size; ++k)
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
