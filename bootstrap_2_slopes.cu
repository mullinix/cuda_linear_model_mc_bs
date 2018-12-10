#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <gsl/gsl_cdf.h> // for calculating std. normal prob,inverse
#include "mystats.h"

// gpu random functions includes
#include <curand.h>
#include <curand_kernel.h>
#include <unistd.h> // for time(NULL) call

#define MAX_SAMPLE 100

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

float my_ahat(float xin[], int n);// calculate "a" for BCa acceleration
void jack_knife(float xin[], float yin[], int n1, int n2, float jack_theta[]);
	       
__device__ void calc_BLUE_slope_intercept(float xin[], float yin[], int n, 
			      float *slope_out, float *intercept_out);
__device__ void calc_mean(float xin[], int n, float* mean);

void calc_BLUE_slope_intercept_host(float xin[], float yin[], int n, 
			      float *slope_out, float *intercept_out);

// this GPU kernel function is used to initialize the random states 
// source: http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html (accessed: 6/5/2017)
__global__ void init_rand_kernel(unsigned int seed, curandState_t* states) {
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

// main GPU kernel
__global__ void mc_bs_2slopes_kernel(curandState_t* states, float d_x[], 
                      float d_y[], int n1, int n2, int B, float d_slope[]){
					
	int idx_glob,i,elt_idx,ranval;
	float boot_x[MAX_SAMPLE],boot_y[MAX_SAMPLE];
	float slope1,slope2,discard;

	int tot_threads=blockDim.x*gridDim.x;
    
    int tot_samples=n1+n2;

	// global index corresponds to bootstrap iterate
	idx_glob = blockIdx.x*blockDim.x+threadIdx.x;

	for(elt_idx=idx_glob;elt_idx<B;elt_idx+=tot_threads){
        // initialize the boot array for sampling w/replacement
        for(i=0;i<tot_samples;i++){
            boot_x[i]=d_x[i];
            boot_y[i]=d_y[i];
        }
		// randomly sample WITHOUT replacement store boot populations
		for(i=tot_samples-1;i>0;i--){
			ranval=curand(&states[blockIdx.x])%(i+1);
            discard=boot_x[i];
			boot_x[i]=boot_x[ranval];
            boot_x[ranval]=discard;
            discard=boot_y[i];
			boot_y[i]=boot_y[ranval];
            boot_y[ranval]=discard;
		}
		// calculate BLUE slope and intercept
		calc_BLUE_slope_intercept(boot_x,boot_y,n1,&slope1,&discard);
		// calculate BLUE slope and intercept
		calc_BLUE_slope_intercept(&boot_x[n1],&boot_y[n1],n2,&slope2,&discard);
		// store results
		d_slope[elt_idx]=fabs(slope1-slope2);
	}
}

int main(int argc, char *argv[]){
	int Nbs;
	int thds_per_block = (1<<8);
	int num_blocks = (1<<12);
	int i,npts1,npts2,npts;
	
	int hist_bins=100;
	int hist_counts[hist_bins],bin;
	float hist_max[hist_bins],bin_width;
	
	float alpha,mean,tmp,SE_median,SE_lower,SE_upper;
	float *h_x,*d_x,*h_y,*d_y;
    float *h_slope,*d_slope,slope,slope1,slope2,intercept;
	float lower_percentile,upper_percentile,middle;
	float BCa_alpha1,BCa_alpha2,z0,lower_BCa,upper_BCa,p_bias;
	float z_lower,z_upper;
	float *jack_knife_array,ahat;
	
	FILE * ipt1_fptr;
    FILE * ipt2_fptr;
	FILE * opt_fptr;
	
	char ipt1_fname[255],ipt2_fname[255];
	char opt_fname[255];
	char readin[255];
	
	curandState_t* states;
	float exectime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// input checking
	if(argc!=4){
		printf("Usage: $ %s <file_1_in> <file_2_in> <alpha>\n",argv[0]);
		return -1;
	}
	
	sprintf(ipt1_fname,"%s",argv[1]);
    sprintf(ipt2_fname,"%s",argv[2]);
	
	alpha=strtod(argv[3],NULL);
    
    npts1=0; npts2=0;
	//read in data
	ipt1_fptr = fopen(ipt1_fname,"r");
	if(ipt1_fptr==NULL){
		printf("Could not open file %s!\n",ipt1_fname);
		return(-07071);
	}
    ipt2_fptr = fopen(ipt2_fname,"r");
	if(ipt2_fptr==NULL){
		printf("Could not open file %s!\n",ipt2_fname);
		return(-07071);
	}
	// count number of points we will be reading
	while(fgets(readin,255,ipt1_fptr) != NULL){
		npts1++;
	}
	printf("File 1 -- number of points read in: %d\n",npts1);
	rewind(ipt1_fptr);
    while(fgets(readin,255,ipt2_fptr) != NULL){
		npts2++;
	}
	printf("File 2 -- number of points read in: %d\n",npts2);
	rewind(ipt2_fptr);
	
	if(npts1<3 || npts2<3){
		printf("Not enough data to work with, exiting!\n");
		return 0;
	}
	
	// if the power set has less than a million, use full BS (no MC)
	printf("The power set is");
    npts=(npts1<npts2)?npts1:npts2;
	if(npts<8){
		Nbs = pow(npts,npts)-npts;
		//printf(" not too large. Running full Bootstrap.");
	}else{
		Nbs = (1<<20);
		//printf(" too large. Running Monte-Carlo Bootstrap.");
	}
	printf("\nThere will be %d BS iterations.\n",Nbs);
	
	// malloc data
	// allocate bootstrap results array
	h_slope=(typeof(h_slope))malloc(Nbs*sizeof(*h_slope));
	if(h_slope==NULL){
		printf("Could not allocate host bootstrap memory in %s!\n",argv[0]);
		return -314;
	}
	// host values, read in from file
	h_x=(typeof(h_x))malloc((npts1+npts2)*sizeof(*h_x));
	h_y=(typeof(h_y))malloc((npts1+npts2)*sizeof(*h_y));

	// jack_knife_array used in BCa
	jack_knife_array=(typeof(jack_knife_array))malloc((npts1+npts2)*sizeof(*jack_knife_array));

	if(h_x==NULL || h_y==NULL || jack_knife_array==NULL){
		printf("Could not allocate host data memory in %s!\n",argv[0]);
		return -314;
	}
	// read data into arrays
	for(i=0;i<npts1;i++){
		fscanf(ipt1_fptr,"%f,%f\n",&h_x[i],&h_y[i]);
	}
	fclose(ipt1_fptr);
    for(i=npts1;i<(npts1+npts2);i++){
		fscanf(ipt2_fptr,"%f,%f\n",&h_x[i],&h_y[i]);
	}
	fclose(ipt2_fptr);
	
	// malloc on device
	cudaMalloc((void**) &d_x, (npts1+npts2) * sizeof(*d_x));
	checkCUDAError("cudaMalloc d_x");
	cudaMalloc((void**) &d_y, (npts1+npts2) * sizeof(*d_y));
	checkCUDAError("cudaMalloc d_y");
	cudaMalloc((void**) &states, num_blocks * sizeof(curandState_t));
	checkCUDAError("cudaMalloc states");
	cudaMalloc((void**) &d_slope, Nbs * sizeof(*d_slope));
	checkCUDAError("cudaMalloc d_slope");

	// store data
	cudaMemcpy(d_x,h_x,(npts1+npts2)*sizeof(*h_x),cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy d_x to device");
	cudaMemcpy(d_y,h_y,(npts1+npts2)*sizeof(*h_y),cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy d_y to device"); 
	
    // initialize random number generator on all blocks
    // start timing 
    cudaEventRecord(start, 0);
    init_rand_kernel<<<num_blocks,thds_per_block>>>(time(0),states);
    // block until the device has completed
    cudaThreadSynchronize();
    //calculate elapsed time:
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    //Computes the elapsed time between two events (in milliseconds)
    cudaEventElapsedTime(&exectime, start, stop);
    checkCUDAError("cuda_init_rand");
    printf("CUDA: init_rand time: %.5e\n",exectime*1e-3);
    
    // run the bootstrap
    // start timing 
    cudaEventRecord(start, 0);
    mc_bs_2slopes_kernel<<<num_blocks,thds_per_block>>>(states, d_x, d_y, npts1, npts2, Nbs, d_slope);
    // block until the device has completed
    cudaThreadSynchronize();
    //calculate elapsed time:
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    //Computes the elapsed time between two events (in milliseconds)
    cudaEventElapsedTime(&exectime, start, stop);
    printf("CUDA: mc_bs_slope time: %.5e\n",exectime*1e-3);
    checkCUDAError("mc_bs_slope_kernel");

	// transfer results to host
	// start timing 
	cudaEventRecord(start, 0);
	// copy device memory to host
	cudaMemcpy(h_slope,d_slope,Nbs*sizeof(*d_slope),cudaMemcpyDeviceToHost);
	//calculate elapsed time:
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	checkCUDAError("cudaMemcpy d_slope,d_s to host");

	printf("CUDA: cudaMemcpy_bs time: %.5e\n",exectime*1e-3);
	    
	/* evaluate SLOPE bootstrap results */
	printf("--- Slope ---\n");
	// BCa percentile procedure
	// create jack-knife array of estimates for the slope
	jack_knife(h_x,h_y,npts1,npts2,jack_knife_array);
	// calculate ahat
	ahat=my_ahat(jack_knife_array,npts);
	// sort to find median (also makes finding p_bias easy)
	middle=my_median(h_slope,Nbs,0);
	// calculate slope to find p_bias
	calc_BLUE_slope_intercept_host(h_x,h_y,npts1,&slope1,&intercept);
    calc_BLUE_slope_intercept_host(&h_x[npts1],&h_y[npts1],npts2,&slope2,&intercept);
    slope=fabs(slope1-slope2);
	printf("slope: %f, min: %f\n",slope,h_slope[0]);
	// find #{theta*<theta}
	for(i=0;i<Nbs && h_slope[i]<slope;i++);
	// this is the probability used to find z0 from std normal
	printf("#{theta*<theta}: %d\n",i);
	p_bias = (float)i/Nbs;
	printf("p_bias: %f\n", p_bias);
    printf("p-value: %f\n",1.0f-p_bias);
	z0=gsl_cdf_ugaussian_Pinv(p_bias);
	printf("z0: %f\n", z0);
	// lower and upper x-values associated with alpha/2 area 
	//     under left and right tails (respectively) of std normal
	z_lower=gsl_cdf_ugaussian_Pinv(alpha*0.5);
	z_upper=gsl_cdf_ugaussian_Qinv(alpha*0.5);
	printf("zinvs: (%f,%f)\n", z_lower,z_upper);
	// finally, these are the alpha values associated with the BCa percentiles
	BCa_alpha1=gsl_cdf_ugaussian_P(z0+(z0+z_lower)/(1-ahat*(z0+z_lower)));
	BCa_alpha2=gsl_cdf_ugaussian_P(z0+(z0+z_upper)/(1-ahat*(z0+z_upper)));
	printf("BCa_quantiles: (%f,%f)\n", BCa_alpha1,BCa_alpha2);
	// calculate BCa confidence intervals
	lower_BCa=h_slope[(int)(BCa_alpha1*Nbs)];
	upper_BCa=h_slope[(int)(BCa_alpha2*Nbs)];	
	// make histogram
	bin_width=(h_slope[Nbs-1]-h_slope[0])/hist_bins;
	printf("bin size: %.7e (%.7e-%.7e)/%d\n",bin_width,h_slope[Nbs-1],h_slope[0],hist_bins);
	for(bin=0,i=0;bin<hist_bins;bin++){
		hist_max[bin]=h_slope[0]+bin_width*(bin+1);
		//printf("bin max: %.7e, slope[%d]: %e\n",hist_max[bin],i,h_slope[i]);
		for(hist_counts[bin]=0;h_slope[i]<=hist_max[bin] && i<Nbs;i++,hist_counts[bin]++);
	}
	
	// non-bias-corrected percentiles
	lower_percentile=h_slope[(int)((alpha*0.5)*Nbs)];
	upper_percentile=h_slope[(int)((1.0-alpha*0.5)*Nbs)];
	
	// SE(median)
	SE_median=0;
	for(i=0;i<Nbs;i++){
		tmp=h_slope[i]-middle;
		SE_median+=tmp*tmp;
	}
	SE_median/=(Nbs-1);
	SE_median=sqrt(SE_median);
	SE_lower=middle-SE_median;
	SE_upper=middle+SE_median;
		
	// write percentile data to file
	sprintf(opt_fname,"%s-vs-%s-slope-CI.dat",ipt1_fname,ipt2_fname);
	opt_fptr=fopen(opt_fname,"w");
	if(opt_fptr==NULL){
		printf("Could not open %s for writing!\n",opt_fname);
		return -1414;
	}
	mean=my_mean(h_slope,Nbs);
	fprintf(opt_fptr,"%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%d,%d\n",
			lower_percentile,upper_percentile,lower_BCa,upper_BCa,
			SE_lower,SE_upper,middle,mean,npts,Nbs);
	fclose(opt_fptr);
	
	// write histogram data to file
	sprintf(opt_fname,"%s-vs-%s-slope-histogram.dat",ipt1_fname,ipt2_fname);
	opt_fptr=fopen(opt_fname,"w");
	if(opt_fptr==NULL){
		printf("Could not open %s for writing!\n",opt_fname);
		return -1414;
	}
	for(bin=0;bin<hist_bins;bin++){
		fprintf(opt_fptr,"%.7e,%d\n",hist_max[bin],hist_counts[bin]);
	}
	fclose(opt_fptr);
	
	printf("percentile:\t(%.7e,%.7e)\n BCa\t\t(%.7e,%.7e)\nMedian: %.7e, Mean: %.7e\n",
	        lower_percentile,upper_percentile,lower_BCa,upper_BCa,middle,mean);
	
		
	free(h_slope);
	free(h_x);
	free(h_y);
	free(jack_knife_array);
	
	cudaFree(d_slope);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(states);	
	
    return 0;
}

float my_ahat(float xin[], int n){
	// calculate ahat for BCa
	// skewness estimator
	float top=0;
	float bot=0;
	float ahat;
	int i;
	float meen=my_mean(xin,n);
	float tmp,tmp1;
	
	for(i=0;i<n;i++){
		tmp=meen-xin[i];
		tmp1=tmp*tmp;
		bot+=tmp1;
		top+=tmp1*tmp;
	}
	bot=bot*bot*bot;
	bot=6*sqrt(bot);
	ahat=top/bot;
	
	return ahat;
}

void jack_knife(float xin[], float yin[], int n1, int n2, float jack_theta[]){
	int i,j;
	float popA[n1+n2],popB[n1+n2];
    float slope1,slope2,discard;
	for(i=0;i<(n1+n2);i++){
        for(j=0;j<(n1+n2)-1;j++){
            if(j!=i){// so long as we aren't equal, set populations
                popA[j]=xin[j];
                popB[j]=yin[j];
            }
        }
        popA[i]=xin[(n1+n2)-1];
        popB[i]=yin[(n1+n2)-1];
        calc_BLUE_slope_intercept_host(popA,popB,n1-1,&slope1,&discard);
        calc_BLUE_slope_intercept_host(&popA[n1-1],&popB[n1-1],n2-1,&slope2,&discard);
		// calculate func for this iteration
		jack_theta[i]=slope1-slope2;
	}
}

__device__ void calc_mean(float xin[], int n, float* mean){
	int i;
	(*mean)=0;
	for(i=0;i<n;i++){
		(*mean)+=xin[i];
	}
	(*mean)/=n;
}

__device__ void calc_BLUE_slope_intercept(float xin[], float yin[], int n, 
			      float *slope_out, float *intercept_out){
	int i;
	float Sx,Sy,Sx2,Sxy;
	Sx=0; Sy=0;
	Sx2=0; Sxy=0;
	for(i=0;i<n;i++){
		Sx +=xin[i];
		Sy +=yin[i];
		Sx2+=xin[i]*xin[i];
		Sxy+=xin[i]*yin[i];
	}
	(*slope_out)=(n*Sxy-Sx*Sy)/(n*Sx2-Sx*Sx);
	(*intercept_out)=(Sy/n)-(*slope_out)*(Sx/n);
}
void calc_BLUE_slope_intercept_host(float xin[], float yin[], int n, 
			      float *slope_out, float *intercept_out){
	int i;
	float Sx,Sy,Sx2,Sxy;
	Sx=0; Sy=0;
	Sx2=0; Sxy=0;
	for(i=0;i<n;i++){
		Sx +=xin[i];
		Sy +=yin[i];
		Sx2+=xin[i]*xin[i];
		Sxy+=xin[i]*yin[i];
	}
	(*slope_out)=(n*Sxy-Sx*Sy)/(n*Sx2-Sx*Sx);
	(*intercept_out)=(Sy/n)-(*slope_out)*(Sx/n);
}

void checkCUDAError(const char *msg){
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err){
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
