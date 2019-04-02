
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void init_image(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float * restrict image);
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float * image = malloc(sizeof(float)*nx*ny);
  float * tmp_image = malloc(sizeof(float)*nx*ny);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Call the stencil kernel
  double tic = wtime();
  #pragma omp simd
  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, image, tmp_image);
    stencil(nx, ny, tmp_image, image);
  }
  double toc = wtime();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);
}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  // #pragma omp simd

  float tmp = 0.0f;
  for (int i = 1; i < nx-1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      const int px = j+i*ny;
      tmp = image[px] * 0.6f;
      tmp += image[px - ny] * 0.1f;
      tmp += image[px + ny] * 0.1f;
      tmp += image[px - 1] * 0.1f;
      tmp += image[px + 1] * 0.1f;
      tmp_image[px] = tmp;
   }
  }
  //corner cases
  tmp = image[0] * 0.6f;
  tmp += image[ny] * 0.1f;
  tmp += image[1] * 0.1f;
  tmp_image[0] = tmp;
  //tmp_image[0] = image[0] * 0.6f + image[ny] * 0.1f + image[1] * 0.1f; //top-left
  const int br = nx*ny-1;
  tmp = image[br] * 0.6f;
  tmp += image[br-ny] * 0.1f;
  tmp += image[br-1] * 0.1f;
  tmp_image[br] = tmp;
  // tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f +  image[(nx-1)*ny-1] * 0.1f + image[nx*ny-2] * 0.1f; //bottom-right
  tmp = image[ny-1] * 0.6f;
  tmp += image[ny-1+ny] * 0.1f;
  tmp += image[ny-2] * 0.1f;
  tmp_image[ny-1] = tmp;
  // tmp_image[ny-1] = image[ny-1] * 0.6f + image[ny-1+ny] * 0.1f + image[ny-2] * 0.1f; //top-right
  const int bl = (nx-1)*ny;
  tmp = image[bl] * 0.6f;
  tmp += image[bl+1] * 0.1f;
  tmp += image[bl-ny] * 0.1f;
  tmp_image[bl] = tmp;
  // tmp_image[(nx-1)*ny] = image[(nx-1)*ny] * 0.6f + image[(nx-1)*ny+1] * 0.1f + image[(nx-2)*ny] * 0.1f; //bottom-left

  //edge cases
  // #pragma omp simd  
  for (int i = 1; i < nx-1; ++i){
    const int px = i*ny;
    tmp = image[px] * 0.6f;
    tmp += image[px - ny] * 0.1f;
    tmp += image[px + ny] * 0.1f;
    tmp += image[px + 1] * 0.1f;
    tmp_image[px] = tmp;
    // tmp_image[px] = image[px] * 0.6f + image[px+1]*0.1f + image[px-ny] * 0.1f + image[px+ny] * 0.1f;
    tmp = image[ny-1+px] * 0.6f;
    tmp += image[ny-1+px+ny] * 0.1f;
    tmp += image[px-1] * 0.1f;
    tmp += image[ny-2+px] * 0.1f;
    tmp_image[ny-1+px] = tmp;
    // tmp_image[ny-1+px] = image[ny-1+px] * 0.6f + image[ny-1+px+ny] * 0.1f + image[px-1] * 0.1f + image[ny-2+px] * 0.1f;
    tmp = image[i] * 0.6f;
    tmp += image[i+ny] * 0.1f;
    tmp += image[i-1] * 0.1f;
    tmp += image[i+1] * 0.1f;
    tmp_image[i] = tmp;
    // tmp_image[i] = image[i] * 0.6f + image[i+ny] * 0.1f + image[i-1] * 0.1f + image[i+1] * 0.1f;
    const int lrpx = i+(nx-1)*ny;
    tmp = image[lrpx] * 0.6f;
    tmp += image[lrpx-ny] * 0.1f;
    tmp += image[lrpx+1] * 0.1f;
    tmp += image[lrpx-1] * 0.1f;
    tmp_image[lrpx] = tmp;
    // tmp_image[lrpx] = image[lrpx] * 0.6f + image[lrpx-ny] * 0.1f + image[lrpx+1] * 0.1f + image[lrpx-1] * 0.1f;
   }
  // #pragma omp simd 
  //  for (int j = 1; j < ny-1; ++j){
  //    tmp_image[j] = image[j] * 0.6f + image[j+ny] * 0.1f + image[j-1] * 0.1f + image[j+1] * 0.1f;
  //    const int lrpx = j+(nx-1)*ny;
  //    tmp_image[lrpx] = image[lrpx] * 0.6f + image[lrpx-ny] * 0.1f + image[lrpx+1] * 0.1f + image[lrpx-1] * 0.1f;
  //  }
}

// Create the input image
void init_image(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0f;
      tmp_image[j+i*ny] = 0.0f;
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
void output_image(const char * file_name, const int nx, const int ny, float * restrict image) {

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
