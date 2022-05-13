#include"fpga_api.h"
#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/mman.h>
#include<cstring>

#define min(x,y) (((x)<(y))?(x):(y))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;

  m1_size_ = v_size * v_size;

  data_size_ = (m_size_+1)*v_size_; // fpga bram data size
  data_size_M = (2*v_size_)*v_size_*sizeof(float);

  fd_ = open("/dev/mem", O_RDWR);
  data_M = static_cast<float*>(mmap(NULL, data_size_M, PROT_READ|PROT_WRITE, MAP_SHARED, fd_, data_addr));
  data_ = new float[data_size_];	

  output_ = static_cast<unsigned int*>(mmap(NULL, sizeof(unsigned int), PROT_READ|PROT_WRITE, MAP_SHARED,fd_, output_addr));
  output_MV = new unsigned int[m_size_];
  // output_M = static_cast<unsigned int*>(NULL);

  num_block_call_ = 0;
}

FPGA::~FPGA()
{
  munmap(data_M, data_size_M);
  munmap(output_, sizeof(unsigned int));
  close(fd_);

  delete[] data_;
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

const float* FPGA::blockMV()
{
  num_block_call_ += 1;

  // cpu version
  float* vec = this->vector();
  float* mat = this->matrix();
  float* out  = reinterpret_cast<float*>(output_MV);  

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

const float* __attribute__((optimize("O0"))) FPGA::blockMM()
{
  num_block_call_ += 1;

  // fpga version
  *output_ = 0x5555;
  while(*output_ == 0x5555);

  return data_M;    
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

      int col_pad = (block_col == v_size_) ? 0 : (v_size_ - block_col);
      int row_pad = (block_row == m_size_)  ? 0 : (m_size_ - block_row);

      // 1) Assign a vector
      memcpy(vec, input + j, sizeof(float) * block_col);
      // 2) Assign a matrix
      for (int row = 0; row < block_row; row++){
          memcpy(mat + (row * v_size_), large_mat + ((i + row) * num_input)+j, sizeof(float) * block_col);
          if (col_pad != 0) memset(mat + ((row *v_size_) + block_col), 0, sizeof(float) * col_pad);
          }
      if (row_pad != 0) memset(mat + (block_row * v_size_), 0, sizeof(float) * v_size_ * row_pad);

      // 3) Call a function `blockMV() to execute MV multiplication
      const float* ret = this->blockMV();

      // 4) Accumulate intermediate results
      for(int row = 0; row < block_row; ++row)
        output[i + row] += ret[row];
    } 
  }
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
        int l, m, v;
        for (l=0;l<block_row;l++){
					for (m=0;m<block_col_1;m++){
            int index1 = (i+l) * (num_input) + (j+m); 
            m1[l * v_size_ + m] = weight_mat[index1];
          }
					for (;m<v_size_;m++) 
            m1[l * v_size_ + m] = 0;
				}
				for (;l<v_size_;l++) 
          for (m=0;m<v_size_;m++)
				  	m1[l * v_size_ + m] = 0;
        
        // 2) Assign a m2
        // IMPLEMENT THIS
        for (m=0;m<block_col_1;m++){
					for (v=0;v<block_col_2;v++){
            int index2 = (j+m) * (num_matrix2) + (k+v);
            m2[m * v_size_ + v] = input_mat[index2];
          }
					for (;v<v_size_;v++){
            m2[m * v_size_ + v] = 0;
          }
				}
				for (;m<v_size_;m++) 
          for (v=0;v<v_size_;v++)
					  m2[m * v_size_ + v] = 0;
            
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
  int input_height = inputs[0].size();
  int input_width = inputs[0][0].size();

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];
  
  //loop variables
  int z,r,i,j,x,y;

  //rearranging inputs into new_inputs matrix
	for (z=0;z<input_channel;z++){
		for (i=0;i<input_height - conv_height + 1;i++){  
      for (j=0;j<input_width - conv_width + 1;j++){
			  for (x=0;x<conv_height;x++){
          for (y=0;y<conv_width;y++){
				    new_inputs[z * (conv_height * conv_width) + x * (conv_width) + y][i * (input_width - conv_width + 1) + j] = inputs[z][i+x][j+y];
          }
        }
      }  
		}
	}
  
  //rearranging convolutional weights into our new_wights matrix
  for (r=0;r<conv_channel;r++){
		for (z=0;z<input_channel;z++){
			for (i=0;i<conv_height;i++){ 
        for (j=0;j<conv_width;j++){
				  new_weights[r][z * (conv_height * conv_width) + (i * conv_width + j)] = cnn_weights[r][z][i][j];
			  }
      }
		}
	}
  
}
