#include"fpga_api.h"
#include<stdio.h>
#include<iostream>
#include<cstring>

using namespace std;

#define min(x,y) (((x)<(y))?(x):(y))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;

  m1_size_ = v_size * v_size;
  m2_size_ = v_size * v_size;
  data_size_ = (m_size_+1)*v_size_; // fpga bram data size

  data_size_M = (v_size_+v_size_)*v_size_; // for Matrix matrix multiplication

  output_ = new unsigned int[m_size_];    // use output_ as tempolar output
  output_M = new unsigned int[v_size_*v_size_];

  data_ = new float[data_size_];	
  data_M = new float[data_size_M]; // for Matrix matrix multiplication

  num_block_call_ = 0;
}
FPGA::~FPGA()
{
  // delete[] output_;
  delete[] output_M;
  // delete[] data_;
  delete[] data_M;
}

float* FPGA::matrix(void)
{
  return data_ + v_size_;
}

float* FPGA::vector(void)
{
  return data_;
}

float* FPGA::matrix_M1(void)
{
  return data_M;
}

float* FPGA::matrix_M2(void)
{
  return data_M + m1_size_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

const float* FPGA::blockMM()
{
  num_block_call_ += 1;

  // cpu version
  float* m1 = this->matrix_M1();
  float* m2 = this->matrix_M2();
  float* out  = reinterpret_cast<float*>(output_M);  

  for(int i = 0; i < v_size_; ++i)
  {
    for(int j = 0; j < v_size_; ++j){    
      out[v_size_*i+j] = 0;
      for(int k = 0; k < v_size_; ++k){
        out[v_size_*i+j] += m1[v_size_*i+k] * m2[v_size_*k + j];
      }
    }
  }

  for(int i = 0; i < m1_size_; ++i)
    data_M[i] = out[i];

  return data_M;    
}

const float* FPGA::blockMV()
{
  num_block_call_ += 1;

  // cpu version
  float* vec = this->vector();
  float* mat = this->matrix();
  float* out  = reinterpret_cast<float*>(output_);  

  for(int i = 0; i < m_size_; ++i)
  {
    out[i] = 0;
    for(int j = 0; j < v_size_; ++j)
      out[i] += vec[j] * mat[v_size_*i + j];
  }

  for(int i = 0; i < m_size_; ++i)
    data_[i] = out[i];

  return data_;    
}

void FPGA::largeMM(const float* weight_mat, const float* input_mat, float* output, int num_input, int num_output, int num_matrix2)
{
  float* m1 = this->matrix_M1();
  float* m2 = this->matrix_M2();

  // 0) Initialize output vector		
  for(int i = 0; i < num_output*num_matrix2; ++i)
    output[i] = 0;

  for(int i = 0; i < num_output; i += v_size_)
  {
    for(int j = 0; j < num_input; j += v_size_)
    {			
      for(int k = 0; k < num_matrix2; k += v_size_)
      {
        // 0) Initialize input vector
        int block_row = min(v_size_, num_output-i);
        int block_col_1 = min(v_size_, num_input-j);
        int block_col_2 = min(v_size_, num_matrix2-k);

        // 1) Assign a m1
        // IMPLEMENT THIS
        int i1, j1, k1;
        for (i1=0;i1<block_row;i1++){
					for (j1=0;j1<block_col_1;j1++) m1[i1 * v_size_ + j1] = weight_mat[(i+i1) * (num_input) + (j+j1)];
					for (;j1<v_size_;j1++) m1[i1 * v_size_ + j1] = 0;
				}
				for (;i1<v_size_;i1++) for (j1=0;j1<v_size_;j1++)
					m1[i1 * v_size_ + j1] = 0;
        // 2) Assign a m2
        // IMPLEMENT THIS
        for (j1=0;j1<block_col_1;j1++){
					for (k1=0;k1<block_col_2;k1++) m2[j1 * v_size_ + k1] = input_mat[(j+j1) * (num_matrix2) + (k+k1)];
					for (;k1<v_size_;k1++) m2[j1 * v_size_ + k1] = 0;
				}
				for (;j1<v_size_;j1++) for (k1=0;k1<v_size_;k1++)
					m2[j1 * v_size_ + k1] = 0;
        // 3) Call a function `blockMM() to execute Matrix matrix multiplication
        const float* ret = this->blockMM();

        // 4) Accumulate intermediate results
        for(int n = 0; n<block_row; ++n)
        {
          for(int m = 0; m<block_col_2; ++m)
          {
            output[(i + n) + (k + m)*num_output] += ret[n*v_size_ + m];
          }
        }
      }
    } 
  }
}

void FPGA::largeMV(const float* large_mat, const float* input, float* output, int num_input, int num_output)
{
  float* vec = this->vector();
  float* mat = this->matrix();

  // 0) Initialize output vector		
  for(int i = 0; i < num_output; ++i)
    output[i] = 0;

  for(int i = 0; i < num_output; i += m_size_)
  {
    for(int j = 0; j < num_input; j += v_size_)
    {			
      // 0) Initialize input vector
      int block_row = min(m_size_, num_output-i);
      int block_col = min(v_size_, num_input-j);

      // 1) Assign a vector
      // IMPLEMENT THIS
      memcpy(vec, input + j, sizeof(float) * block_col);      
      // 2) Assign a matrix
      // IMPLEMENT THIS
      int k=0;
      for(; k< block_row ; k++)
      {
            memcpy(mat+ v_size_* k, large_mat + (i+k) * num_input + j, sizeof(float) * block_col);
            if(block_col < v_size_) memset(mat+ v_size_ * k + block_col, 0, sizeof(float) * (v_size_ - block_col));
        }
        if(k < m_size_)
        {
            for(int x = 0; x < m_size_ - k ; x++)
            {
                memset(mat+ v_size_ * ( k + x ), 0, sizeof(float) * v_size_);
            }
      }
      // 3) Call a function `blockMV() to execute MV multiplication
      const float* ret = this->blockMV();

      // 4) Accumulate intermediate results
      for(int row = 0; row < block_row; ++row)
        output[i + row] += ret[row];
    } 
  }
}

void FPGA::convLowering(const std::vector<std::vector<std::vector<std::vector<float>>>>& cnn_weights,
    std::vector<std::vector<float>>& new_weights,
    const std::vector<std::vector<std::vector<float>>>& inputs,
    std::vector<std::vector<float>>& new_inputs) {
  /*
   * Arguments:
   *
   * conv_weights: [conv_channel, input_channel, conv_height, conv_width]
   * new_weights: [?, ?]
   * inputs: [input_channel, input_height, input_width]
   * new_inputs: [?, ?]
   *
   */

  int conv_channel = cnn_weights.size();
  int input_channel = cnn_weights[0].size();
  int conv_height = cnn_weights[0][0].size();
  int conv_width = cnn_weights[0][0][0].size();
  //int input_channel = cnn_weights.size();
  int input_height = inputs[0].size();
  int input_width = inputs[0][0].size();

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];
  int ic,cc,i,j,i2,j2;
	for (ic=0;ic<input_channel;ic++){
		for (i=0;i<input_height - conv_height + 1;i++) for (j=0;j<input_width - conv_width + 1;j++){
			for (i2=0;i2<conv_height;i2++) for (j2=0;j2<conv_width;j2++){
				new_inputs[ic * (conv_height * conv_width) + i2 * (conv_width) + j2][i * (input_width - conv_width + 1) + j] = inputs[ic][i+i2][j+j2];
			}
		}
	}


  for (cc=0;cc<conv_channel;cc++){
		for (ic=0;ic<input_channel;ic++){
			for (i=0;i<conv_height;i++) for (j=0;j<conv_width;j++){
				new_weights[cc][ic * (conv_height * conv_width) + (i * conv_width + j)] = cnn_weights[cc][ic][i][j];
			}
		}
	}

}
