
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <string.h>
// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image);
void topStencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image);
void bottomStencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image);
void midStencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image);

void send_recv(int ny, int height, float send_buf[], float recv_buf[], float img[], MPI_Status status, int src, int des, int end);

void init_image(const int nx, const int ny, float *restrict image, float *restrict tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *restrict image);
double wtime(void);

int main(int argc, char *argv[]) {

  int flag;
  int size;
  int rank;
  MPI_Status status;
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);
  // float *out_image = malloc(sizeof(float)*nx*ny);
  float out_image[ny*nx];

  MPI_Init(&argc, &argv);

  MPI_Initialized(&flag);
  if (flag != 1 ) {
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Check usage
  // if (argc != 4) {
  //   fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
  //   exit(EXIT_FAILURE);
  // }

  // Allocate the image
  float *image = malloc(sizeof(float)*nx*ny);
  float *tmp_image = malloc(sizeof(float)*nx*ny);
 
  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // float *send_buf1= malloc(sizeof(float)*ny);
  float up_buf[ny]; 
  float down_buf[ny]; 

  // 2. try to run the stencil by one master and 4 workers

  // double gtic = MPI_Wtime();
  int numCores = size;
  int height = nx/numCores;
  // float *localImg = malloc(sizeof(float)*ny*(nx/numCores+2));
  float localImg[ny*(nx/numCores+2)];
  float localTmp1[ny*(nx/numCores+2)];
  float localTmp2[ny*(nx/numCores+2)];
  
  float localTmp[ny*(nx/numCores+1)];
  float localImg0[ny*(nx/numCores+1)];

  // float localImg1[ny*(nx/numCores+2)];
  // float localImg2[ny*(nx/numCores+2)];
  

  int start = rank * height;

  if (size == 1){
    double tic = wtime();
    for (int i =0; i < niters; i++){
      stencil(nx, ny, image, tmp_image);
      stencil(nx, ny, tmp_image, image);
    }
    double toc = wtime();
    printf("------------------------------------\n");
    printf(" runtime for one worker: %lf s\n", toc-tic);
    printf("------------------------------------\n");
  }

  else if (rank == 0) {

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < ny; ++j) {
        localImg0[j+i*ny] = image[j+i*ny];
      }
    }
    double tic = wtime();

    for (int i =0; i < niters; i++){
      
      // send neighbours wanted rows
      for (int j = 0; j < ny; ++j) {
        down_buf[j] = localImg0[j+(height-1)*ny];
      }
      MPI_Ssend(&down_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);

      //get rows from neighbour rank
      MPI_Recv(&up_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localImg0[j+height*ny] = up_buf[j];
      }
      // send_recv(ny, height, send_buf1, send_buf2, localImg, status, 1, 1, end);
      topStencil(height+1, ny, localImg0, localTmp);

      //send neighbours wanted rows
      for (int j = 0; j < ny; ++j) {
        down_buf[j] = localTmp[j+(height-1)*ny];
      }
      MPI_Ssend(&down_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);

      MPI_Recv(&up_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      //get rows from neighbour rank
      for (int j = 0; j < ny; ++j) {
        localTmp[j+height*ny] = up_buf[j];
      }
      // send_recv(ny, height, send_buf1, send_buf2, localTmp, status, 1, 1, end);
      topStencil(height+1, ny, localTmp, localImg0);
    }

    double toc = wtime();
    printf("------------------------------------\n");
    printf(" runtime rank0: %lf s\n", toc-tic);
    printf("------------------------------------\n");
    output_image("stencil0.pgm", height+1, ny, localImg0);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < ny; ++j) {
        out_image[j+i*ny] = localImg0[j+i*ny];
      }
    }
    printf ("%f to %f from rank %d\n", localImg0[0], localImg0[(ny-1)+(height)*ny], 0);

    free(localImg0);

    MPI_Recv(localImg0, ny*(nx/numCores+1), MPI_FLOAT, size-1, 0, MPI_COMM_WORLD, &status);
    printf ("%f to %f from rank %d\n", localImg0[ny], localImg0[(ny-1)+height*ny], size-1);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < ny; ++j) {
          out_image[j+((size-1)*height+i)*ny] = localImg0[j+(i+1)*ny];
        }
    }
    printf("%f to %f given by rank %d\n", out_image[((size-1)*height)*ny], out_image[ny-1+((size-1)*height+height-1)*ny], size-1);

    printf ("%f \n", out_image[((size-1)*height)*ny]);
    for (int src = 1; src < size-1; src++){

      MPI_Recv(localImg, ny*(nx/numCores+2), MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
      printf ("%f to %f from rank %d\n", localImg[ny], localImg[(ny-1)+(height)*ny], src);
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < ny; ++j) {
          out_image[j+(src*height+i)*ny] = localImg[j+(i+1)*ny];
        }
        
      }
      printf("%f to %f given by rank %d\n", out_image[(src*height)*ny], out_image[ny-1+(src*height+height-1)*ny], src);

    }
    // printf ("%f to %f \n", out_image[0], out_image[(ny-1)/2]);

    output_image(OUTPUT_FILE, nx, ny, out_image);
    // free(out_image);
  }

  else if (rank==size-1){
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < ny; ++j) {
        localImg0[j+(i+1)*ny] = image[j+(start+i)*ny];
      }
    }
    double tic = wtime();
    if(rank % 2 == 0){
      printf("even\n");

      for (int i =0; i < niters; i++){
        
        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localImg0[j+ny];
        }
        MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

        //receive from lower rank its last row
        MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localImg0[j] = down_buf[j];
        }
        
        bottomStencil(height+1, ny, localImg0, localTmp);

        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localTmp[j+ny];
        }
        MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
        //receive from lower rank its last row
        MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localTmp[j] = down_buf[j];
        }

        bottomStencil(height+1, ny, localTmp, localImg0);

      }
    double toc = wtime();
    
    printf("------------------------------------\n");
    printf(" runtime rank %d: %lf s\n", rank, toc-tic);
    printf("------------------------------------\n");
    MPI_Ssend(localImg0, ny*(nx/numCores+1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    output_image("stencil3.pgm", height+1, ny, localImg0);

    } 
    else if (rank % 2 == 1){
      printf("odd\n");
      double tic = wtime();
      for (int i =0; i < niters; i++){
        //receive from lower rank its last row
        MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localImg0[j] = down_buf[j];
        }
        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localImg0[j+ny];
        }
        MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

        bottomStencil(height+1, ny, localImg0, localTmp);

        //receive from lower rank its last row
        MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localTmp[j] = down_buf[j];
        }
        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localTmp[j+ny];
        }
        MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

        bottomStencil(height+1, ny, localTmp, localImg0);

      }
    double toc = wtime();
    printf("------------------------------------\n");
    printf(" runtime rank %d: %lf s\n", rank, toc-tic);
    printf("------------------------------------\n");
    MPI_Ssend(localImg0, ny*(nx/numCores+1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    output_image("stencil3.pgm", height+1, ny, localImg0);

    }
  }

  else if(rank % 2 == 1 && rank != size-1){
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < ny; ++j) {
        localImg[j+(i+1)*ny] = image[j+(start+i)*ny];
      }
    }
    double tic = wtime();
    for (int i =0; i < niters; i++){

      //receive from lower rank its last row
      MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localImg[j] = down_buf[j];
      }
      //send lower rank the first row
      for (int j = 0; j < ny; ++j) {
        up_buf[j] = localImg[j+ny];
      }
      MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

      //receive from higher rank its first row
      MPI_Recv(&up_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localImg[j+(height+1)*ny] = up_buf[j];
      }
      //send higher rank the last row
      for (int j = 0; j < ny; ++j) {
        down_buf[j] = localImg[j+height*ny];
      }
      MPI_Ssend(&down_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);

      midStencil(height+2, ny, localImg, localTmp1);

      //receive from lower rank its last row
      MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localTmp1[j] = down_buf[j];
      }
      //send lower rank the first row
      for (int j = 0; j < ny; ++j) {
        up_buf[j] = localTmp1[j+ny];
      }
      MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

      //receive from higher rank its first row
      MPI_Recv(&up_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localTmp1[j+(height+1)*ny] = up_buf[j];
      }
      //send higher rank the last row
      for (int j = 0; j < ny; ++j) {
        down_buf[j] = localTmp1[j+height*ny];
      }
      MPI_Ssend(&down_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
      
      midStencil(height+2, ny, localTmp1, localImg);
    }
    double toc = wtime();

    printf("------------------------------------\n");
    printf(" runtime rank1: %lf s\n", toc-tic);
    printf("------------------------------------\n");
    output_image("stencil1.pgm", height+2, ny, localImg);

    MPI_Ssend(localImg, ny*(nx/numCores+2), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }

  else if (rank % 2 == 0 && rank != 0 && rank != size-1){
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < ny; ++j) {
        localImg[j+(i+1)*ny] = image[j+(start+i)*ny];
      }
    }
    double tic = wtime();
    for (int i =0; i < niters; i++){

      //send higher rank the last row
      for (int j = 0; j < ny; ++j) {
        down_buf[j] = localImg[j+height*ny];
      }
      MPI_Ssend(&down_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
      //receive from higher rank its first row
      MPI_Recv(&up_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localImg[j+(height+1)*ny] = up_buf[j];
      }
      
      //send first row to lower rank
      for (int j = 0; j < ny; ++j) {
        up_buf[j] = localImg[j+ny];
      }
      MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
      //receive from lower rank its last row
      MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localImg[j] = down_buf[j];
      }      
      midStencil(height+2, ny, localImg, localTmp2);

      //send higher rank the last row
      for (int j = 0; j < ny; ++j) {
        down_buf[j] = localTmp2[j+height*ny];
      }
      MPI_Ssend(&down_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
      //receive from higher rank its first row
      MPI_Recv(&up_buf, ny, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localTmp2[j+(height+1)*ny] = up_buf[j];
      }
      //send first row to lower rank
      for (int j = 0; j < ny; ++j) {
        up_buf[j] = localTmp2[j+ny];
      }
      MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
      //receive from lower rank its last row
      MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < ny; ++j) {
        localTmp2[j] = down_buf[j];
      }      
      midStencil(height+2, ny, localTmp2, localImg);
      
    }
    double toc = wtime();

    printf("------------------------------------\n");
    printf(" runtime rank2: %lf s\n", toc-tic);
    printf("------------------------------------\n");
    output_image("stencil2.pgm", height+2, ny, localImg);

    MPI_Ssend(localImg, ny*(nx/numCores+2), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
void send_recv(int ny, int height, float send_buf[], float recv_buf[], float img[], MPI_Status status, int src, int des, int end){
  for (int j = 0; j < ny; ++j) {
    send_buf[j] = img[j+end*ny];
  }
  MPI_Ssend(&send_buf, ny, MPI_FLOAT, des, 0, MPI_COMM_WORLD);
  MPI_Recv(&recv_buf, ny, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
  //get rows from neighbour rank
  for (int j = 0; j < ny; ++j) {
    img[j+height*ny] = recv_buf[j];
  }
}

void topStencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image){
  //top-left
  tmp_image[0] = image[0] * 0.6f;
  tmp_image[0] += image[ny] * 0.1f;
  tmp_image[0] += image[1] * 0.1f;
  //top-right
  tmp_image[ny-1] = image[ny-1] * 0.6f;
  tmp_image[ny-1] += image[ny-1+ny] * 0.1f;
  tmp_image[ny-1] += image[ny-2] * 0.1f;
  //top edge
  for (int i = 1; i < ny-1; ++i){
    tmp_image[i] = image[i] * 0.6f;
    tmp_image[i] += image[i+ny] * 0.1f;
    tmp_image[i] += image[i-1] * 0.1f;
    tmp_image[i] += image[i+1] * 0.1f;
  }
  //side
  for (int i = 1; i < nx-1; ++i){
    const int lcpx = i*ny;
    tmp_image[lcpx] = image[lcpx] * 0.6f;
    tmp_image[lcpx] += image[lcpx - ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + 1] * 0.1f;
    const int rcpx = i*ny+ny-1;
    tmp_image[rcpx] = image[rcpx] * 0.6f;
    tmp_image[rcpx] += image[rcpx + ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - 1] * 0.1f;
  }
  //middle
  for (int i = 1; i < nx-1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      const int px = j+i*ny;
      tmp_image[px] = image[px] * 0.6f;
      tmp_image[px] += image[px - ny] * 0.1f;
      tmp_image[px] += image[px + ny] * 0.1f;
      tmp_image[px] += image[px - 1] * 0.1f;
      tmp_image[px] += image[px + 1] * 0.1f;
    }
  }
}
void bottomStencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image){
  //bottom-right
  tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f;
  tmp_image[nx*ny-1] += image[nx*ny-1-ny] * 0.1f;
  tmp_image[nx*ny-1] += image[nx*ny-2] * 0.1f;
  //bottom-left
  tmp_image[(nx-1)*ny] = image[(nx-1)*ny] * 0.6f;
  tmp_image[(nx-1)*ny] += image[(nx-1)*ny+1] * 0.1f;
  tmp_image[(nx-1)*ny] += image[(nx-1)*ny-ny] * 0.1f;
  //bottom edge
  for (int i = 1; i < ny-1; ++i){
    const int lrpx = i+(nx-1)*ny;
    tmp_image[lrpx] = image[lrpx] * 0.6f;
    tmp_image[lrpx] += image[lrpx-ny] * 0.1f;
    tmp_image[lrpx] += image[lrpx+1] * 0.1f;
    tmp_image[lrpx] += image[lrpx-1] * 0.1f;
  }
  //side
  for (int i = 1; i < nx-1; ++i){
    const int lcpx = i*ny;
    tmp_image[lcpx] = image[lcpx] * 0.6f;
    tmp_image[lcpx] += image[lcpx - ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + 1] * 0.1f;
    const int rcpx = i*ny+ny-1;
    tmp_image[rcpx] = image[rcpx] * 0.6f;
    tmp_image[rcpx] += image[rcpx + ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - 1] * 0.1f;
  }
  //middle
  for (int i = 1; i < nx-1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      const int px = j+i*ny;
      tmp_image[px] = image[px] * 0.6f;
      tmp_image[px] += image[px - ny] * 0.1f;
      tmp_image[px] += image[px + ny] * 0.1f;
      tmp_image[px] += image[px - 1] * 0.1f;
      tmp_image[px] += image[px + 1] * 0.1f;
    }
  }
}
void midStencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image){
  //side
  for (int i = 1; i < nx-1; ++i){
    const int lcpx = i*ny;
    tmp_image[lcpx] = image[lcpx] * 0.6f;
    tmp_image[lcpx] += image[lcpx - ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + 1] * 0.1f;
    const int rcpx = i*ny+ny-1;
    tmp_image[rcpx] = image[rcpx] * 0.6f;
    tmp_image[rcpx] += image[rcpx + ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - 1] * 0.1f;
  }
  //middle
  for (int i = 1; i < nx-1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      const int px = j+i*ny;
      tmp_image[px] = image[px] * 0.6f;
      tmp_image[px] += image[px - ny] * 0.1f;
      tmp_image[px] += image[px + ny] * 0.1f;
      tmp_image[px] += image[px - 1] * 0.1f;
      tmp_image[px] += image[px + 1] * 0.1f;
    }
  }
}


void stencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image) {
  //top-left
  tmp_image[0] = image[0] * 0.6f;
  tmp_image[0] += image[ny] * 0.1f;
  tmp_image[0] += image[1] * 0.1f;
  //bottom-right
  tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f;
  tmp_image[nx*ny-1] += image[nx*ny-1-ny] * 0.1f;
  tmp_image[nx*ny-1] += image[nx*ny-2] * 0.1f;
  //top-right
  tmp_image[ny-1] = image[ny-1] * 0.6f;
  tmp_image[ny-1] += image[ny-1+ny] * 0.1f;
  tmp_image[ny-1] += image[ny-2] * 0.1f;
  //bottom-left
  tmp_image[(nx-1)*ny] = image[(nx-1)*ny] * 0.6f;
  tmp_image[(nx-1)*ny] += image[(nx-1)*ny+1] * 0.1f;
  tmp_image[(nx-1)*ny] += image[(nx-1)*ny-ny] * 0.1f;
  for (int i = 1; i < nx-1; ++i){
    const int lcpx = i*ny;
    tmp_image[lcpx] = image[lcpx] * 0.6f;
    tmp_image[lcpx] += image[lcpx - ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + ny] * 0.1f;
    tmp_image[lcpx] += image[lcpx + 1] * 0.1f;
    const int rcpx = i*ny+ny-1;
    tmp_image[rcpx] = image[rcpx] * 0.6f;
    tmp_image[rcpx] += image[rcpx + ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - ny] * 0.1f;
    tmp_image[rcpx] += image[rcpx - 1] * 0.1f;
  }
  for (int i = 1; i < ny-1; ++i){
  tmp_image[i] = image[i] * 0.6f;
  tmp_image[i] += image[i+ny] * 0.1f;
  tmp_image[i] += image[i-1] * 0.1f;
  tmp_image[i] += image[i+1] * 0.1f;
  const int lrpx = i+(nx-1)*ny;
  tmp_image[lrpx] = image[lrpx] * 0.6f;
  tmp_image[lrpx] += image[lrpx-ny] * 0.1f;
  tmp_image[lrpx] += image[lrpx+1] * 0.1f;
  tmp_image[lrpx] += image[lrpx-1] * 0.1f;
  }  
  for (int i = 1; i < nx-1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      const int px = j+i*ny;
      tmp_image[px] = image[px] * 0.6f;
      tmp_image[px] += image[px - ny] * 0.1f;
      tmp_image[px] += image[px + ny] * 0.1f;
      tmp_image[px] += image[px - 1] * 0.1f;
      tmp_image[px] += image[px + 1] * 0.1f;
    }
  }
 
}

// Create the input image
void init_image(const int nx, const int ny, float *restrict image, float *restrict tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *restrict image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0f;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
