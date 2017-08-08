#include "CTensor.h"
#include "timer.h"
#include <cstring>
#include <cmath>



void unarycosts_pixelwise_euclidean(CTensor<float>& leftImg, CTensor<float>& rightImg, int x_size, int y_size, float* Disparity)
{
  for( int y = 0; y < y_size; ++y){
    for( int x = 0; x < x_size; ++x){

  	  float ed_temp = sqrtf((leftImg(x, y, 0) - rightImg(x, y, 0)) * (leftImg(x, y, 0) - rightImg(x, y, 0)) +
     	              	    (leftImg(x, y, 1) - rightImg(x, y, 1)) * (leftImg(x, y, 1) - rightImg(x, y, 1)) +
        	             	(leftImg(x, y, 2) - rightImg(x, y, 2)) * (leftImg(x, y, 2) - rightImg(x, y, 2)));

	  Disparity[y * x_size + x] = ed_temp;	
	}
  }
}




void unarycosts_L1(CTensor<float>& leftImg, CTensor<float>& rightImg, int x_size, int y_size, float* Disparity)
{


  for( int y = 0; y < y_size; ++y){
    for( int x = 0; x < x_size; ++x){
      int l1_init = 37485;  // max difference between two kernels
      int l1_temp = 0;

      for(int sC = 50; sC >= 0 ; sC--){  // max matching distance is 50(only from right side)
        for(int i = -3; i < 4; i++){
          for(int j = -3; j < 4; j++){
            if(0 <= x + i < x_size && 0 <= y + j < y_size && 0 <= x + i + sC < x_size)
            {  
			  l1_temp += fabs(leftImg(x+i, y+j, 0) - rightImg(x+sC+i, y+j, 0)) + 
              			 fabs(leftImg(x+i, y+j, 1) - rightImg(x+sC+i, y+j, 1)) + 
              			 fabs(leftImg(x+i, y+j, 2) - rightImg(x+sC+i, y+j, 2)); 



           }
          } 
        }
    
        if(l1_temp <= l1_init)
          l1_init = l1_temp;

        Disparity[(y * x_size + x) * 51 + sC] = l1_temp;
        l1_temp = 0;
      } 
    }
  }
}


void unarycosts_L2(CTensor<float>& leftImg, CTensor<float>& rightImg, int x_size, int y_size, float* Disparity)
{

  for( int y = 0; y < y_size; ++y){
    for( int x = 0; x < x_size; ++x){

      int l2_init = 255 * 255 * 3 * 49 ;  // max difference between two kernels
      int l2_temp = 0;

      for(int sC = 0; sC <51; sC++){  // max matching distance is 50(only from right side)
        for(int i = -3; i < 4; i++){
          for(int j = -3; j < 4; j++){
            if(0 <= x + i < x_size && 0 <= y + j < y_size && 0 <= x + i + sC < x_size)
            {  
			  l2_temp += (leftImg(x+i ,y+j, 0) - rightImg(x+sC+i, y+j, 0)) * (leftImg(x+i ,y+j, 0) - rightImg(x+sC+i, y+j, 0)) +
						 (leftImg(x+i ,y+j, 1) - rightImg(x+sC+i, y+j, 1)) * (leftImg(x+i ,y+j, 1) - rightImg(x+sC+i, y+j, 1)) +
						 (leftImg(x+i ,y+j, 2) - rightImg(x+sC+i, y+j, 2)) * (leftImg(x+i ,y+j, 2) - rightImg(x+sC+i, y+j, 2));
            }
          } 
        }
    
        if(l2_temp <= l2_init)
          l2_init = l2_temp;

        Disparity[(y * x_size + x) * 51 + sC] = l2_temp;
        l2_temp = 0;
      } 
    }
  }

}

void unarycosts_NCC(CTensor<float>& leftImg, CTensor<float>& rightImg, int x_size, int y_size, float* Disparity)
{

  for( int y = 0; y < y_size; ++y){
    for( int x = 0; x < x_size; ++x){
      float ncc_init = 0;  // min difference bewteen two kernels

      for(int sC = 50; sC >= 0; sC--){  // max matching distance is 50(only from right side)
        int count = 0;
        float sum_left_x = 0;
        float sum_left_y = 0;
        float sum_left_z = 0;

        float sum_right_x = 0;
        float sum_right_y = 0;
        float sum_right_z = 0;
 
        for(int i = -3; i < 4; i++)
          for(int j = -3; j < 4; j++)
            if(0 <= x + i < x_size && 0 <= y + j < y_size){
              count ++;     
              sum_left_x += leftImg(x+i,y+j,0);
              sum_left_y += leftImg(x+i,y+j,1);
              sum_left_z += leftImg(x+i,y+j,2); 

              sum_right_x += rightImg(x+sC+i, y+j, 0);
              sum_right_y += rightImg(x+sC+i, y+j, 1);
              sum_right_z += rightImg(x+sC+i, y+j, 2);       
           }
  
        float average_left_x = sum_left_x / count;
        float average_left_y = sum_left_y / count;
        float average_left_z = sum_left_z / count;
  
        float average_right_x = sum_right_x / count;
        float average_right_y = sum_right_y / count;
        float average_right_z = sum_right_z / count;

        float ncc_temp1 = 0;
        float ncc_temp2 = 0;
        float ncc_temp  = 0;

        for(int i = -3; i < 4; i++){
          for(int j = -3; j < 4; j++){
            if(0 <= x + i < x_size && 0 <= y + j < y_size){
              ncc_temp1 += ((leftImg(x+i,y+j,0) - average_left_x)*(rightImg(x+sC+i, y+j, 0) - average_right_x) +
                            (leftImg(x+i,y+j,1) - average_left_y)*(rightImg(x+sC+i, y+j, 1) - average_right_y) +
                            (leftImg(x+i,y+j,2) - average_left_z)*(rightImg(x+sC+i, y+j, 2) - average_right_z));
  
              ncc_temp2 += sqrt(((leftImg(x+i,y+j,0) - average_left_x) * (leftImg(x+i,y+j,0) - average_left_x) +
                                 (leftImg(x+i,y+j,1) - average_left_y) * (leftImg(x+i,y+j,1) - average_left_y) +
                                 (leftImg(x+i,y+j,2) - average_left_z) * (leftImg(x+i,y+j,2) - average_left_z)) *
                                ((rightImg(x+sC+i, y+j, 0) - average_right_x) * (rightImg(x+sC+i, y+j, 0) - average_right_x) +
                                 (rightImg(x+sC+i, y+j, 1) - average_right_y) * (rightImg(x+sC+i, y+j, 1) - average_right_y) +
                                 (rightImg(x+sC+i, y+j, 2) - average_right_z) * (rightImg(x+sC+i, y+j, 2) - average_right_z)));
              
              ncc_temp = ncc_temp1 / ncc_temp2;  
            }
          }
        }

        if(ncc_temp >= ncc_init)
            ncc_init = ncc_temp;

        Disparity[(y * x_size + x) * 51 + sC] = 10000*(1 - ncc_temp);
        ncc_temp = 0;
      }
    }
  }
}




void belief_propagation(int x_size, int y_size, float* MpqsF0, float* MpqsB0, float* MpqsF2, float* MpqsB2, float* MpqsF4, float* MpqsB4, float* MpqsF6, float* MpqsB6, float* MpqsF9, float* MpqsB9,
                                                           float* MpqsF11, float* MpqsB11, float* MpqsF13, float* MpqsB13, float* MpqsF15, float* MpqsB15, float* Disparity, float* result)
{

  int potts = 0;
  int lambda0 = 1500, lambda2 = 1000, lambda4 = 1000, lambda6 = 1000, lambda9 = 1000, lambda11 = 1000, lambda13 = 1000, lambda15 = 1000;





  //forward pass 0 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50 ; j++)
      MpqsF0[y * x_size * 51 + j] = 0.0f;
    for(int q = 1; q < x_size; q++)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        MpqsF0[(y * x_size + q) * 51 + j] = Disparity[(y * x_size + q-1) * 51] + MpqsF0[(y * x_size + q-1) * 51] + lambda0 * potts;
        for(int i = 1; i <= 50; i++)
        {
          if(i == j)
            potts = 0;
          else
            potts = 1;
          float costf0 = Disparity[(y * x_size + q-1) * 51 + i] + MpqsF0[(y * x_size + q-1) * 51 + i] + lambda0 * potts;
          if(costf0 < MpqsF0[(y * x_size + q) * 51 + j])
            MpqsF0[(y * x_size + q) * 51 + j] = costf0;
        }
      }
    }  
    
  //backward pass 0 degree
    for(int j = 0; j <= 50; j++)
      MpqsB0[((y+1) * x_size - 1) * 51 + j] = 0.0f;
    for(int q = x_size - 2; q >= 0; q--)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        MpqsB0[(y * x_size + q) * 51 + j] = Disparity[(y * x_size + q+1) * 51] + MpqsB0[(y * x_size + q+1) * 51] + lambda0 * potts;
        for(int i = 1; i <= 50; i++)
        {
          if(i == j)
            potts = 0;
          else
            potts = 1;
          float costb0 = Disparity[(y * x_size + q+1) * 51 + i] + MpqsB0[(y * x_size + q+1) * 51 + i] + lambda0 * potts;
          if(costb0 < MpqsB0[(y * x_size + q) * 51 + j])
            MpqsB0[(y * x_size + q) * 51 + j] = costb0;
        }
      }
    }
  }



  //forward pass 22.5 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50 ; j++){
      MpqsF2[y * x_size * 51 + j] = 0.0f;
      MpqsF2[(y * x_size + 1) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= (x_size-1) * 51 ; j++){
    MpqsF2[((y_size - 2) * x_size) * 51 + j] = 0.0f;
    MpqsF2[((y_size - 3) * x_size) * 51 + j] = 0.0f;
  }
  

  for( int x = 0; x < x_size; ++x){  
    for(int q = y_size - 3; q >= 0; q--)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q+1) * x_size + x-2 < 0 || (q+1) * x_size + x-2 >= y_size * x_size) 
          MpqsF2[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsF2[(q * x_size + x) * 51 + j] = Disparity[((q+1) * x_size + x-2) * 51] + MpqsF2[((q+1) * x_size + x-2) * 51] + lambda2 * potts;       
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costf2 = Disparity[((q+1) * x_size + x-2) * 51 + i] + MpqsF2[((q+1) * x_size + x-2) * 51 + i] + lambda2 * potts;
            if(costf2 < MpqsF2[(q * x_size + x) * 51 + j])
              MpqsF2[(q * x_size + x) * 51 + j] = costf2;
          }
	    }
      }
    }  
  }


  //backward pass 22.5 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50; j++){
      MpqsB2[((y+1) * x_size - 1) * 51 + j] = 0.0f;
      MpqsB2[((y+1) * x_size - 2) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= 2*(x_size-1)*51; j++)
    MpqsB2[j] = 0.0f;
  
  for( int x = 0; x < x_size; ++x){
    for(int q = 2; q < y_size; q++)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q-1) * x_size + x+2 < 0 || (q-1) * x_size + x+2 >= y_size * x_size)
          MpqsB2[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsB2[(q * x_size + x) * 51 + j] = Disparity[((q-1) * x_size + x+2) * 51] + MpqsB2[((q-1) * x_size + x+2) * 51] + lambda2 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costb2 = Disparity[((q-1) * x_size + x+2) * 51 + i] + MpqsB2[((q-1) * x_size + x+2) * 51 + i] + lambda2 * potts;
            if(costb2 < MpqsB2[(q * x_size + x) * 51 + j])
              MpqsB2[(q * x_size + x) * 51 + j] = costb2;
          }
        }
      }
    }
  }  


  

  

  //forward pass 45 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50 ; j++){
      MpqsF4[y * x_size * 51 + j] = 0.0f;
      MpqsF4[(y * x_size + 1) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= (x_size-1) * 51 ; j++){
    MpqsF4[((y_size - 2) * x_size) * 51 + j] = 0.0f;
    MpqsF4[((y_size - 3) * x_size) * 51 + j] = 0.0f;
  }

  for( int x = 0; x < x_size; ++x){
    for(int q = y_size - 3; q >= 0; q--)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q+1) * x_size + x-1 < 0 || (q+1) * x_size + x-1 >= y_size * x_size) 
          MpqsF4[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsF4[(q * x_size + x) * 51 + j] = Disparity[((q+1) * x_size + x-1) * 51] + MpqsF4[((q+1) * x_size + x-1) * 51] + lambda4 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costf4 = Disparity[((q+1) * x_size + x-1) * 51 + i] + MpqsF4[((q+1) * x_size + x-1) * 51 + i] + lambda4 * potts;
            if(costf4 < MpqsF4[(q * x_size + x) * 51 + j])
              MpqsF4[(q * x_size + x) * 51 + j] = costf4;
          }
        }
      }  
    }  
  }


  //backward pass 45 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50; j++){
      MpqsB4[((y+1) * x_size - 1) * 51 + j] = 0.0f;
      MpqsB4[((y+1) * x_size - 2) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= 2*(x_size-1)*51; j++)
    MpqsB4[j] = 0.0f;

  for( int x = 0; x < x_size; ++x){
    for(int q = 2; q < y_size; q++)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q-1) * x_size + x+1 < 0 || (q-1) * x_size + x+1 >= y_size * x_size)
          MpqsB4[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsB4[(q * x_size + x) * 51 + j] = Disparity[((q-1) * x_size + x+1) * 51] + MpqsB4[((q-1) * x_size + x+1) * 51] + lambda4 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costb4 = Disparity[((q-1) * x_size + x+1) * 51 + i] + MpqsB4[((q-1) * x_size + x+1) * 51 + i] + lambda4 * potts;
            if(costb4 < MpqsB4[(q * x_size + x) * 51 + j])
              MpqsB4[(q * x_size + x) * 51 + j] = costb4;
          }
        }
      }
    }
  }




  //forward pass 67.5 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50 ; j++){
      MpqsF6[y * x_size * 51 + j] = 0.0f;
      MpqsF6[(y * x_size + 1) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= (x_size-1) * 51 ; j++){
    MpqsF6[((y_size - 2) * x_size) * 51 + j] = 0.0f;
    MpqsF6[((y_size - 3) * x_size) * 51 + j] = 0.0f;
  }

  for( int x = 0; x < x_size; ++x){ 
    for(int q = y_size - 3; q >= 0; q--)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q+2) * x_size + x-1 < 0 || (q+2) * x_size + x-1 >= y_size * x_size) 
          MpqsF6[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsF6[(q * x_size + x) * 51 + j] = Disparity[((q+2) * x_size + x-1) * 51] + MpqsF6[((q+2) * x_size + x-1) * 51] + lambda6 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costf6 = Disparity[((q+2) * x_size + x-1) * 51 + i] + MpqsF6[((q+2) * x_size + x-1) * 51 + i] + lambda6 * potts;
            if(costf6 < MpqsF6[(q * x_size + x) * 51 + j])
              MpqsF6[(q * x_size + x) * 51 + j] = costf6;
          }
        }
      }   
    }   
  }

  //backward pass 67.5 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50; j++){
      MpqsB6[((y+1) * x_size - 1) * 51 + j] = 0.0f;
      MpqsB6[((y+1) * x_size - 2) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= 2*(x_size-1)*51; j++)
    MpqsB6[j] = 0.0f;
 
  for( int x = 0; x < x_size; ++x){
    for(int q = 2; q < y_size; q++)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q-2) * x_size + x+1 < 0 || (q-2) * x_size + x+1 >= y_size * x_size)
          MpqsB6[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsB6[(q * x_size + x) * 51 + j] = Disparity[((q-2) * x_size + x+1) * 51] + MpqsB6[((q-2) * x_size + x+1) * 51] + lambda6 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costb6 = Disparity[((q-2) * x_size + x+1) * 51 + i] + MpqsB6[((q-2) * x_size + x+1) * 51 + i] + lambda6 * potts;
            if(costb6 < MpqsB6[(q * x_size + x) * 51 + j])
              MpqsB6[(q * x_size + x) * 51 + j] = costb6;
          }
        }
      }
    }
  }






  //forward pass 90 degree
  for(int j = 0; j <= (x_size-1) * 51 ; j++){
    MpqsF9[((y_size - 2) * x_size) * 51 + j] = 0.0f;
  }

  for( int x = 0; x < x_size; ++x){
    for(int q = y_size - 2; q >= 0; q--)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        MpqsF9[(q * x_size + x) * 51 + j] = Disparity[((q+1) * x_size + x) * 51] + MpqsF9[((q+1) * x_size + x) * 51] + lambda9 * potts;
        for(int i = 1; i <= 50; i++)
        {
          if(i == j)
            potts = 0;
          else
            potts = 1;
          float costf9 = Disparity[((q+1) * x_size + x) * 51 + i] + MpqsF9[((q+1) * x_size + x) * 51 + i] + lambda9 * potts;
          if(costf9 < MpqsF9[((q+1) * x_size + x) * 51 + j])
            MpqsF9[((q+1) * x_size + x) * 51 + j] = costf9;
        }
      }
    }  
  }

  //backward pass 90 degree
  for(int j = 0; j <= (x_size-1)*51; j++)
    MpqsB9[j] = 0.0f;

  for( int x = 0; x < x_size; ++x){
    for(int q = 1; q < y_size; q++)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        MpqsB9[(q * x_size + x) * 51 + j] = Disparity[((q-1) * x_size + x) * 51] + MpqsB9[((q-1) * x_size + x) * 51] + lambda9 * potts;
        for(int i = 1; i <= 50; i++)
        {
          if(i == j)
            potts = 0;
          else
            potts = 1;
          float costb9 = Disparity[((q-1) * x_size + x) * 51 + i] + MpqsB9[((q-1) * x_size + x) * 51 + i] + lambda9 *potts;
          if(costb9 < MpqsB9[((q-1) * x_size + x) * 51 + j])
            MpqsB9[((q-1) * x_size + x) * 51 + j] = costb9;
        }
      }
    }
  }




  //forward pass 112.5 degree


  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50; j++){
      MpqsF11[((y+1) * x_size - 1) * 51 + j] = 0.0f;
      MpqsF11[((y+1) * x_size - 2) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= (x_size-1) * 51 ; j++){
    MpqsF11[((y_size - 2) * x_size) * 51 + j] = 0.0f;
    MpqsF11[((y_size - 3) * x_size) * 51 + j] = 0.0f;
  }


  for( int x = 0; x < x_size; ++x){
    for(int q = y_size - 3; q >= 0; q--)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q+2) * x_size + x+1 < 0 || (q+2) * x_size + x+1 >= y_size * x_size) 
          MpqsF11[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsF11[(q * x_size + x) * 51 + j] = Disparity[((q+2) * x_size + x+1) * 51] + MpqsF11[((q+2) * x_size + x+1) * 51] + lambda11 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costf11 = Disparity[((q+2) * x_size + x+1) * 51 + i] + MpqsF11[((q+2) * x_size + x+1) * 51 + i] + lambda11 * potts;
            if(costf11 < MpqsF11[(q * x_size + x) * 51 + j])
              MpqsF11[(q * x_size + x) * 51 + j] = costf11;
          }
        }
      }   
    }  
  }

  //backward pass 112.5 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50 ; j++){
      MpqsB11[y * x_size * 51 + j] = 0.0f;
      MpqsB11[(y * x_size + 1) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= 2*(x_size-1)*51; j++)
    MpqsB11[j] = 0.0f;

  for( int x = 0; x < x_size; ++x){
    for(int q = 2; q < y_size; q++)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q-2) * x_size + x-1 < 0 || (q-2) * x_size + x-1 >= y_size * x_size)
          MpqsB11[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsB11[(q * x_size + x) * 51 + j] = Disparity[((q-2) * x_size + x-1) * 51] + MpqsB11[((q-2) * x_size + x-1) * 51] + lambda11 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costb11 = Disparity[((q-2) * x_size + x-1) * 51 + i] + MpqsB11[((q-2) * x_size + x-1) * 51 + i] + lambda11 * potts;
            if(costb11 < MpqsB11[(q * x_size + x) * 51 + j])
              MpqsB11[(q * x_size + x) * 51 + j] = costb11;
          }
        }
      }
    }
  }


  //forward pass 135 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50; j++){
      MpqsF13[((y+1) * x_size - 1) * 51 + j] = 0.0f;
      MpqsF13[((y+1) * x_size - 2) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= (x_size-1) * 51 ; j++){
    MpqsF13[((y_size - 2) * x_size) * 51 + j] = 0.0f;
    MpqsF13[((y_size - 3) * x_size) * 51 + j] = 0.0f;
  }

  for( int x = 0; x < x_size; ++x){
    for(int q = y_size - 3; q >= 0; q--)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q+1) * x_size + x+1 < 0 || (q+1) * x_size + x+1 >= y_size * x_size) 
          MpqsF13[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsF13[(q * x_size + x) * 51 + j] = Disparity[((q+1) * x_size + x+1) * 51] + MpqsF13[((q+1) * x_size + x+1) * 51] + lambda13 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costf13 = Disparity[((q+1) * x_size + x+1) * 51 + i] + MpqsF13[((q+1) * x_size + x+1) * 51 + i] + lambda13 * potts;
            if(costf13 < MpqsF13[(q * x_size + x) * 51 + j]) 
              MpqsF13[(q * x_size + x) * 51 + j] = costf13;
          }
        }
      }  
    }  
  }

              

  //backward pass 135 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50 ; j++){
      MpqsB13[y * x_size * 51 + j] = 0.0f;
      MpqsB13[(y * x_size + 1) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= 2*(x_size-1)*51; j++)
    MpqsB13[j] = 0.0f;

  for( int x = 0; x < x_size; ++x){
    for(int q = 2; q < y_size; q++)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q-1) * x_size + x-1 < 0 || (q-1) * x_size + x-1 >= y_size * x_size)
          MpqsB13[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsB13[(q * x_size + x) * 51 + j] = Disparity[((q-1) * x_size + x-1) * 51] + MpqsB13[((q-1) * x_size + x-1) * 51] + lambda13 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costb13 = Disparity[((q-1) * x_size + x-1) * 51 + i] + MpqsB13[((q-1) * x_size + x-1) * 51 + i] + lambda13 * potts;
            if(costb13 < MpqsB13[(q * x_size + x) * 51 + j])
              MpqsB13[(q * x_size + x) * 51 + j] = costb13;
          }
        }
      }
    }
  }



 


  //forward pass 157.5 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50; j++){
      MpqsF15[((y+1) * x_size - 1) * 51 + j] = 0.0f;
      MpqsF15[((y+1) * x_size - 2) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= (x_size-1) * 51 ; j++){
    MpqsF15[((y_size - 2) * x_size) * 51 + j] = 0.0f;
    MpqsF15[((y_size - 3) * x_size) * 51 + j] = 0.0f;
  }
 
  for( int x = 0; x < x_size; ++x){
    for(int q = y_size - 3; q >= 0; q--)
    {
      for(int j = 0; j <= 50 ; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q+1) * x_size + x+2 < 0 || (q+1) * x_size + x+2 >= y_size * x_size) 
          MpqsF15[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsF15[(q * x_size + x) * 51 + j] = Disparity[((q+1) * x_size + x+2) * 51] + MpqsF15[((q+1) * x_size + x+2) * 51] + lambda15 * potts;
          for(int i = 1; i <= 50; i++)
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costf15 = Disparity[((q+1) * x_size + x+2) * 51 + i] + MpqsF15[((q+1) * x_size + x+2) * 51 + i] + lambda15 * potts;
            if(costf15 < MpqsF15[(q * x_size + x) * 51 + j])
              MpqsF15[(q * x_size + x) * 51 + j] = costf15;
          }
        }
      }  
    }  
  }


  //backward pass 157.5 degree
  for( int y = 0; y < y_size; ++y){
    for(int j = 0; j <= 50 ; j++){
      MpqsB15[y * x_size * 51 + j] = 0.0f;
      MpqsB15[(y * x_size + 1) * 51 + j] = 0.0f;
    }
  }
  for(int j = 0; j <= 2*(x_size-1)*51; j++)
    MpqsB15[j] = 0.0f;

  for( int x = 0; x < x_size; ++x){
    for(int q = 2; q < y_size; q++)
    {
      for(int j = 0; j <= 50; j++)
      {
        if(j == 0)
          potts = 0;
        else
          potts = 1;
        if((q-1) * x_size + x-2 < 0 || (q-1) * x_size + x-2 >= y_size * x_size)
          MpqsB15[(q * x_size + x) * 51 + j] = 0;
        else{
          MpqsB15[(q * x_size + x) * 51 + j] = Disparity[((q-1) * x_size + x-2) * 51] + MpqsB15[((q-1) * x_size + x-2) * 51] + lambda15 * potts;
          for(int i = 1; i <= 50; i++)
    
          {
            if(i == j)
              potts = 0;
            else
              potts = 1;
            float costb15 = Disparity[((q-1) * x_size + x-2) * 51 + i] + MpqsB15[((q-1) * x_size + x-2) * 51 + i] + lambda15 * potts;
            if(costb15 < MpqsB15[(q * x_size + x) * 51 + j])
              MpqsB15[(q * x_size + x) * 51 + j] = costb15;
          }
        }
      }
    }
  }



  //decision
  int minIndex = 0;

  for( int y = 0; y < y_size; ++y){
    for(int q = 0; q < x_size; q++)
    {
      minIndex = 0;

      float minCost = Disparity[(y * x_size + q) * 51] + MpqsF0[(y * x_size + q) * 51] + MpqsB0[(y * x_size + q) * 51] + MpqsF2[(y * x_size + q) * 51] + MpqsB2[(y * x_size + q) * 51]
                                                       + MpqsF4[(y * x_size + q) * 51] + MpqsB4[(y * x_size + q) * 51] + MpqsF6[(y * x_size + q) * 51] + MpqsB6[(y * x_size + q) * 51]
                                                       + MpqsF9[(y * x_size + q) * 51] + MpqsB9[(y * x_size + q) * 51] + MpqsF11[(y * x_size + q) * 51] + MpqsB11[(y * x_size + q) * 51]
                                                       + MpqsF13[(y * x_size + q) * 51] + MpqsB13[(y * x_size + q) * 51] + MpqsF15[(y * x_size + q) * 51] + MpqsB15[(y * x_size + q) * 51];

      for(int i = 1; i <= 50; i++)
      { 
        float cost = Disparity[(y * x_size + q) * 51 + i] + MpqsF0[(y * x_size + q) * 51 + i] + MpqsB0[(y * x_size + q) * 51 + i] + MpqsF2[(y * x_size + q) * 51 + i] + MpqsB2[(y * x_size + q) * 51 + i]
                                                          + MpqsF4[(y * x_size + q) * 51 + i] + MpqsB4[(y * x_size + q) * 51 + i] + MpqsF6[(y * x_size + q) * 51 + i] + MpqsB6[(y * x_size + q) * 51 + i]
                                                          + MpqsF9[(y * x_size + q) * 51 + i] + MpqsB9[(y * x_size + q) * 51 + i] + MpqsF11[(y * x_size + q) * 51 + i] + MpqsB11[(y * x_size + q) * 51 + i]
                                                          + MpqsF13[(y * x_size + q) * 51 + i] + MpqsB13[(y * x_size + q) * 51 + i] + MpqsF15[(y * x_size + q) * 51 + i] + MpqsB15[(y * x_size + q) * 51 + i];
        if(cost < minCost)
        {
          minCost = cost;
          minIndex = i;
        }
      }

      result[y * x_size + q] = minIndex;

    }
  } 
} 


void imgconv(int x_size, int y_size, float* result, CTensor<float>& resultImg)
{
  for( int y = 0; y < y_size; ++y)
    for( int x = 0; x < x_size; ++x){

  int sC = result[y * x_size + x];
  float a = sC * 5;
  if(a > 255 || a < 0)
    a = fmax(0, fmin(a, 255));

  resultImg(x,y,0) = a;
  resultImg(x,y,1) = a;
  resultImg(x,y,2) = a;

}

}


int main(int argc, char** argv)
{
  
  /*-----------------------------------------------------------------------
   *  Read rectified left and right input image and put them into
   *  Color CMatrices
   *-----------------------------------------------------------------------*/
  CTensor<float> leftImg;
  leftImg.readFromPPM("couchR.ppm");

  CTensor<float> rightImg;
  rightImg.readFromPPM("couchL.ppm");

  CTensor<float> resultImg;
  resultImg.readFromPPM("couchL.ppm");


  float* Disparity = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* imgStore0 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize());
  float* imgStore1 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize());
  float* imgStore2 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize());

  float* MpqsF0 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB0 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsF2 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB2 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsF4 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB4 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsF6 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB6 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsF9 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB9 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsF11 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB11 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsF13 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB13 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsF15 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* MpqsB15 = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize()*51);

  float* result = (float*) malloc(sizeof(float)*leftImg.xSize()*leftImg.ySize());

  // Which block matching method do we choose?
  int input;
  do {
    std::cout << "\n"
              << "Choose:\n\n"
              << " (1): Sum of absolute differences(L1)\n"
              << " (2): Sum of squared differences(L2)\n"
              << " (3): Normalized Cross Correlation\n"
              << "\n"
              << "Our choice [1-3]: ";
    std::cin >> input;
  } while (input < 1 or input > 3);

  switch (input) {
   // 
	case 1: { unarycosts_L1(leftImg, rightImg, leftImg.xSize(), leftImg.ySize(), Disparity);   break; }    
	case 2: { unarycosts_L2(leftImg, rightImg, leftImg.xSize(), leftImg.ySize(), Disparity);   break; }
    case 3: { unarycosts_NCC(leftImg, rightImg, leftImg.xSize(), leftImg.ySize(), Disparity);   break; }
    default: throw std::runtime_error("Invalid choice");
  }
  
 
  timer::start("CPU processing");
  belief_propagation(leftImg.xSize(), leftImg.ySize(), MpqsF0, MpqsB0, MpqsF2, MpqsB2, MpqsF4, MpqsB4, MpqsF6, MpqsB6, MpqsF9, MpqsB9, 
                                                       MpqsF11, MpqsB11, MpqsF13, MpqsB13, MpqsF15, MpqsB15, Disparity, result);
  
  imgconv(leftImg.xSize(), leftImg.ySize(), result, resultImg);

  resultImg.writeToPPM("couch.ppm");
  timer::stop("CPU processing"); 
  timer::printToScreen();



  free(Disparity);
  free(MpqsF0);
  free(MpqsB0);
  free(MpqsF2);
  free(MpqsB2);
  free(MpqsF4);
  free(MpqsB4);
  free(MpqsF6);
  free(MpqsB6);
  free(MpqsF9);
  free(MpqsB9);
  free(MpqsF11);
  free(MpqsB11);
  free(MpqsF13);
  free(MpqsB13);
  free(MpqsF15);
  free(MpqsB15);
  free(result);

  return 0;
}
