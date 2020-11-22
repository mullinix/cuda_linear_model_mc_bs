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

#define DEBUG 0
#define SLOPE 0
#define INTERCEPT 1
#define PEARSON_RHO 2

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

float my_ahat(float xin[], int n);// calculate "a" for BCa acceleration
void jack_knife(float xin[], float yin[], int n, int variable, float jack_theta[], 
	        float (*func)(float *, float *, int, int));
float jack_knife_wrapper(float xin[], float yin[], int n, int variable);
int stats_from_bs(
	float x[],
	float y[],
	float alpha,
	int npts,
	int histbins,
	int Nbs,
	int res_idx,
	float BS_data[],
	float *median,
	float *BCa_upper,
	float *BCa_lower,
	float *percentile_upper,
	float *percentile_lower,
	float *SE_upper,
	float *SE_lower,
	float hist_max[],
	int hist_counts[],
	float (*jk_func)(float *, float *, int, int)
	);
	       
__host__ __device__ void calc_BLUE_slope_intercept(float xin[], float yin[], int n, float results[]);

// this GPU kernel function is used to initialize the random states 
// source: http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html (accessed: 6/5/2017)
__global__ void init_rand_kernel(unsigned int seed, curandState_t* states) {
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

// main GPU kernel
__global__ void mc_bs_slope_kernel(
	curandState_t* states,
	float d_x[], float d_y[],
	int n, int B,
	float d_slope[], float d_intercept[], float d_pearson_rho[]){
					
	extern __shared__ float shared[];
        float *boot_x = &shared[0];
        float *boot_y = &shared[n];
        int idx_glob,i,elt_idx,ranval;
	float results[3];

	int tot_threads=blockDim.x*gridDim.x;

	// global index corresponds to bootstrap iterate
	idx_glob = blockIdx.x*blockDim.x+threadIdx.x;

	for(elt_idx=idx_glob;elt_idx<B;elt_idx+=tot_threads){
		// randomly sample, store boot populations
		for(i=0;i<n;i++){
			ranval=curand(&states[blockIdx.x])%n;
			boot_x[i]=d_x[ranval];
			boot_y[i]=d_y[ranval];
		}
		// calculate BLUE slope and intercept
		calc_BLUE_slope_intercept(boot_x,boot_y,n,results);
		// store results
		d_slope[elt_idx]=results[SLOPE];
		d_intercept[elt_idx]=results[INTERCEPT];
		d_pearson_rho[elt_idx]=results[PEARSON_RHO];
	}
}
// run full bootstrap!
__global__ void full_bs_slope_kernel(
	float d_x[], float d_y[],
	int n, int B,
	float d_slope[], float d_intercept[], float d_pearson_rho[]){
					
	extern __shared__ float shared[];
        float *boot_x = &shared[0];
        float *boot_y = &shared[n];
        int idx_glob,i,elt_idx,pop_idx,divided;
	unsigned long long int skip;
	float results[3];
	
	int tot_threads=blockDim.x*gridDim.x;

	// global index corresponds to bootstrap iterate
	idx_glob = blockIdx.x*blockDim.x+threadIdx.x;
	
	skip=pow(n,n)-1;
	skip/=(n-1);

	for(elt_idx=idx_glob;elt_idx<B;elt_idx+=tot_threads){
		// we skip over guaranteed singularities 
		divided = (elt_idx+1)+(elt_idx+1)*(skip+1)/skip/skip;

		// Grab a bootstrap sample (not random!)
		for(i=0;i<n;i++){
			pop_idx = divided%n;
			boot_x[i]=d_x[pop_idx];
			boot_y[i]=d_y[pop_idx];
			divided/=n;
		}
		// calculate BLUE slope and intercept
		calc_BLUE_slope_intercept(boot_x,boot_y,n,results);
		// store results
		d_slope[elt_idx]=results[SLOPE];
		d_intercept[elt_idx]=results[INTERCEPT];
		d_pearson_rho[elt_idx]=results[PEARSON_RHO];
	}
}

int main(int argc, char *argv[]){
	int Nbs;
	int thds_per_block = (1<<8);
	int num_blocks = (1<<12);
	int bin,i,npts=0;
	
	int hist_bins=100;
	int hist_counts[hist_bins];
	float hist_max[hist_bins];
	
	float *h_x,*h_y,*h_slope,*h_intercept,*h_pearson_rho;
	float *d_x,*d_y,*d_slope,*d_intercept,*d_pearson_rho;

	float mean,SE_lower,SE_upper;
	float lower_percentile,upper_percentile,middle,lower_BCa,upper_BCa;
	
	FILE * ipt_fptr;
	FILE * opt_fptr;
	
	char ipt_fname[255];
	char opt_fname[255];
	char readin[255];
	float alpha;
	
	curandState_t* states;
	float exectime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// input checking
	if(argc!=3){
		printf("Usage: $ %s <filein> <alpha>\n",argv[0]);
		return -1;
	}
	
	sprintf(ipt_fname,"%s",argv[1]);
	
	alpha=strtod(argv[2],NULL);

	//read in data
	ipt_fptr = fopen(ipt_fname,"r");
	if(ipt_fptr==NULL){
		printf("Could not open file %s!\n",ipt_fname);
		return(-07071);
	}
	// count number of points we will be reading
	while(fgets(readin,255,ipt_fptr) != NULL){
		npts++;
	}
	printf("Number of points read in: %d\n",npts);
	rewind(ipt_fptr);
	
	if(npts<3){
		printf("Not enough data to work with, exiting!\n");
		return 0;
	}
	
	// if the power set has less than a million, use full BS (no MC)
	printf("The power set is");
	if(npts<8){
		Nbs = pow(npts,npts)-npts;
		printf(" not too large. Running full Bootstrap.");
	}else{
		Nbs = (1<<20);
		printf(" too large. Running Monte-Carlo Bootstrap.");
	}
	printf("\nThere will be %d BS iterations.\n",Nbs);
	
	// malloc data
	// allocate bootstrap results array
	h_slope=(typeof(h_slope))malloc(Nbs*sizeof(*h_slope));
	if(h_slope==NULL){
		printf("Could not allocate host bootstrap memory in %s!\n",argv[0]);
		return -314;
	}
	// allocate bootstrap results array
	h_intercept=(typeof(h_intercept))malloc(Nbs*sizeof(*h_intercept));
	if(h_intercept==NULL){
		printf("Could not allocate host bootstrap memory in %s!\n",argv[0]);
		return -314;
	}
	// allocate bootstrap results array
	h_pearson_rho=(typeof(h_pearson_rho))malloc(Nbs*sizeof(*h_pearson_rho));
	if(h_pearson_rho==NULL){
		printf("Could not allocate host bootstrap memory in %s!\n",argv[0]);
		return -314;
	}
	// host values, read in from file
	h_x=(typeof(h_x))malloc(npts*sizeof(*h_x));
	h_y=(typeof(h_y))malloc(npts*sizeof(*h_y));

	// read data into arrays
	for(i=0;i<npts;i++){
		fscanf(ipt_fptr,"%f,%f\n",&h_x[i],&h_y[i]);
	}
	fclose(ipt_fptr);
	
	// malloc on device
	cudaMalloc((void**) &d_x, npts * sizeof(*d_x));
	checkCUDAError("cudaMalloc d_x");
	cudaMalloc((void**) &d_y, npts * sizeof(*d_y));
	checkCUDAError("cudaMalloc d_y");
	cudaMalloc((void**) &states, num_blocks * sizeof(curandState_t));
	checkCUDAError("cudaMalloc states");
	cudaMalloc((void**) &d_slope, Nbs * sizeof(*d_slope));
	checkCUDAError("cudaMalloc d_slope");
	cudaMalloc((void**) &d_intercept, Nbs * sizeof(*d_intercept));
	checkCUDAError("cudaMalloc d_intercept");
	cudaMalloc((void**) &d_pearson_rho, Nbs * sizeof(*d_pearson_rho));
	checkCUDAError("cudaMalloc d_pearson_rho");

	// store data
	cudaMemcpy(d_x,h_x,npts*sizeof(*h_x),cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy d_x to device");
	cudaMemcpy(d_y,h_y,npts*sizeof(*h_y),cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy d_y to device"); 

        int block_memory_size = 2*npts*sizeof(*h_x);
	
	if(Nbs==(1<<20)){ // we perform MC-BS
		// initialize random number generator on all blocks
		// start timing 
		cudaEventRecord(start, 0);
		init_rand_kernel<<<num_blocks,thds_per_block>>>(time(0),states);
		// block until the device has completed
		cudaDeviceSynchronize();
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
		mc_bs_slope_kernel<<<num_blocks,thds_per_block,block_memory_size>>>(states, d_x, d_y, npts, Nbs, d_slope, d_intercept, d_pearson_rho);
		// block until the device has completed
		cudaDeviceSynchronize();
		//calculate elapsed time:
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		//Computes the elapsed time between two events (in milliseconds)
		cudaEventElapsedTime(&exectime, start, stop);
		printf("CUDA: mc_bs_slope time: %.5e\n",exectime*1e-3);
		checkCUDAError("mc_bs_slope_kernel");
	}else{ // we perform full BS
		// run the bootstrap
		// start timing 
		cudaEventRecord(start, 0);
		full_bs_slope_kernel<<<num_blocks,thds_per_block,block_memory_size>>>(d_x, d_y, npts, Nbs, d_slope, d_intercept, d_pearson_rho);
		// block until the device has completed
		cudaDeviceSynchronize();
		//calculate elapsed time:
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		//Computes the elapsed time between two events (in milliseconds)
		cudaEventElapsedTime(&exectime, start, stop);
		printf("CUDA: full bs_slope time: %.5e\n",exectime*1e-3);
		checkCUDAError("full bs_slope_kernel");
	}

	// transfer results to host
	// start timing 
	cudaEventRecord(start, 0);
	// copy device memory to host
	cudaMemcpy(h_slope,d_slope,Nbs*sizeof(*d_slope),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_intercept,d_intercept,Nbs*sizeof(*d_intercept),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pearson_rho,d_pearson_rho,Nbs*sizeof(*d_pearson_rho),cudaMemcpyDeviceToHost);
	//calculate elapsed time:
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	checkCUDAError("cudaMemcpy d_slope,d_intercept,d_pearson_rho to host");

        cudaEventElapsedTime(&exectime, start, stop);
	printf("CUDA: cudaMemcpy_bs time: %.5e\n",exectime*1e-3);
	
	/* evaluate SLOPE bootstrap results */
	printf("--- Slope BS Results ---\n");
	sprintf(opt_fname,"%s-CI.dat",ipt_fname);
	opt_fptr=fopen(opt_fname,"w");
	if(opt_fptr==NULL){
		printf("Could not open %s for writing!\n",opt_fname);
		return -1414;
	}
	fprintf(opt_fptr, "statistic,lower_pctile,upper_pctile,lower_BCa,upper_BCa,lower_SE,upper_SE,median,mean,data_size,bootstrap_samples\n");
	fclose(opt_fptr);

	// BCa percentile procedure
	float *BS_map[3] = {h_slope, h_intercept, h_pearson_rho};
	const char *stat_map[3] = {"slope", "intercept", "pearson_rho"};
	float results[3];
	calc_BLUE_slope_intercept(h_x,h_y,npts,results);
	for(int variable=0; variable<3; variable++){
		stats_from_bs(
			//inputs
			h_x, h_y, alpha, npts, hist_bins, Nbs, variable, BS_map[variable], 
			//scalar outputs
			&middle, &upper_BCa, &lower_BCa, &upper_percentile, &lower_percentile, &SE_upper, &SE_lower,
			//histogram outputs
			hist_max, hist_counts,
			// jackknife function pointer
			jack_knife_wrapper
		);

		// write percentile data to file
		sprintf(opt_fname,"%s-CI.dat",ipt_fname);
		opt_fptr=fopen(opt_fname,"a");
		if(opt_fptr==NULL){
			printf("Could not open %s for writing!\n",opt_fname);
			return -1414;
		}
		mean=my_mean(BS_map[variable],Nbs);
		fprintf(opt_fptr,"%s,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%d,%d\n",
				stat_map[variable],lower_percentile,upper_percentile,lower_BCa,upper_BCa,
				SE_lower,SE_upper,middle,mean,npts,Nbs);
		fclose(opt_fptr);
		
		// write histogram data to file
		sprintf(opt_fname,"%s-%s-histogram.dat",ipt_fname,stat_map[variable]);
		opt_fptr=fopen(opt_fname,"w");
		if(opt_fptr==NULL){
			printf("Could not open %s for writing!\n",opt_fname);
			return -1414;
		}
		for(bin=0;bin<hist_bins;bin++){
			fprintf(opt_fptr,"%.7e,%d\n",hist_max[bin],hist_counts[bin]);
		}
		fclose(opt_fptr);
		
		printf("BLUE %s: %.7e\n", stat_map[variable], results[variable]);
		printf("Bootstrap\n");
		printf(" - Median: %.7e\n", middle);
		printf(" - %d%% CI (BCa): (%.7e,%.7e)\n", (int)((1.0-alpha)*100+0.5), lower_BCa, upper_BCa);
		printf(" - Standard Error (Median): (%.7e,%.7e)\n", SE_lower, SE_upper);
		printf(" - Mean: %.7e\n", mean);

		if(DEBUG)
		    printf(" - Percentile %d%% CI: (%.7e,%.7e)\n", (int)((1.0-alpha)*100+0.5), lower_percentile, upper_percentile);
	}
		
	
		
	free(h_slope);
	free(h_intercept);
	free(h_pearson_rho);
	free(h_x);
	free(h_y);

	
	cudaFree(d_slope);
	cudaFree(d_intercept);
	cudaFree(d_pearson_rho);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(states);	
	
    return 0;
}

int stats_from_bs(
	float x[],
	float y[],
	float alpha,
	int npts,
	int hist_bins,
	int Nbs,
	int res_idx,
	float BS_data[],
	float *median,
	float *BCa_upper,
	float *BCa_lower,
	float *percentile_upper,
	float *percentile_lower,
	float *SE_upper,
	float *SE_lower,
	float hist_max[],
	int hist_counts[],
	float (*jk_func)(float *, float *, int, int)
	){
	/*
	stats_from_bs
	
	Args:
	    x: independent variable data
	    y: dependent variable data
	    alpha: confidence level
	    npts: number of data points in x,y
	    hist_bins: number of histogram bins to compute
	    Nbs: size of bootstrap array
	    res_idx: which result index to use (slope=SLOPE, intercept=INTERCEPT, pearson_rho=PEARSON_RHO)
	    BS_data: bootstrap data array
	    jk_func: function pointer for jack-knife function

	Outputs:
	    median, BCa_upper/lower, SE_upper/lower, percentile_upper/lower: output data pointers
	    hist_max: array where we store the histogram bin RHS value
	    hist_counts: array where we store the histogram counts
	    returns: 0
	    
	*/
	float lower_percentile,upper_percentile,middle,lower_SE,upper_SE,lower_BCa,upper_BCa;
	float ahat,BCa_alpha1,BCa_alpha2,z0,p_bias,z_lower,z_upper;
	float *jack_knife_array;
	float results[3];
	int bin,i;
	float bin_width;
	float tmp,SE_median;
	// jack_knife_array used in BCa
	jack_knife_array=(typeof(jack_knife_array))malloc(npts*sizeof(*jack_knife_array));
	if(jack_knife_array==NULL){
		printf("Could not allocate jackknife memory!\n");
		return -314;
	}
	// create jack-knife array of estimates for the slope
	jack_knife(x,y,npts,res_idx,jack_knife_array,jk_func);
	// calculate ahat
	ahat=my_ahat(jack_knife_array,npts);
	// sort to find median (also makes finding p_bias easy)
	middle=my_median(BS_data,Nbs,0);
	// calculate slope to find p_bias
	calc_BLUE_slope_intercept(x,y,npts,results);
	// find #{theta*<theta}
	for(i=0;i<Nbs && BS_data[i]<results[res_idx];i++);
	// this is the probability used to find z0 from std normal
	if(DEBUG)
            printf("#{theta*<theta}: %d\n",i);
	p_bias = (float)i/Nbs;
	if(DEBUG)
	    printf("p_bias: %f\n", p_bias);
	z0=gsl_cdf_ugaussian_Pinv(p_bias);
	if(DEBUG)
	    printf("z0: %f\n", z0);
	// lower and upper x-values associated with alpha/2 area 
	//     under left and right tails (respectively) of std normal
	z_lower=gsl_cdf_ugaussian_Pinv(alpha*0.5);
	z_upper=gsl_cdf_ugaussian_Qinv(alpha*0.5);
	if(DEBUG)
	    printf("zinvs: (%f,%f)\n", z_lower,z_upper);
	// finally, these are the alpha values associated with the BCa percentiles
	BCa_alpha1=gsl_cdf_ugaussian_P(z0+(z0+z_lower)/(1-ahat*(z0+z_lower)));
	BCa_alpha2=gsl_cdf_ugaussian_P(z0+(z0+z_upper)/(1-ahat*(z0+z_upper)));
	if(DEBUG)
	    printf("BCa_quantiles: (%f,%f)\n", BCa_alpha1,BCa_alpha2);
	// calculate BCa confidence intervals
	lower_BCa=BS_data[(int)(BCa_alpha1*Nbs)];
	upper_BCa=BS_data[(int)(BCa_alpha2*Nbs)];	
	// make histogram
	bin_width=(BS_data[Nbs-1]-BS_data[0])/hist_bins;
	if(DEBUG)
	    printf("bin size: %.7e (%.7e-%.7e)/%d\n",bin_width,BS_data[Nbs-1],BS_data[0],hist_bins);
	for(bin=0,i=0;bin<hist_bins;bin++){
		hist_max[bin]=BS_data[0]+bin_width*(bin+1);
	        if(DEBUG)
		    printf("bin max: %.7e, data[%d]: %e\n",hist_max[bin],i,BS_data[i]);
		for(hist_counts[bin]=0;BS_data[i]<=hist_max[bin] && i<Nbs;i++,hist_counts[bin]++);
	}
	
	// non-bias-corrected percentiles
	lower_percentile=BS_data[(int)((alpha*0.5)*Nbs)];
	upper_percentile=BS_data[(int)((1.0-alpha*0.5)*Nbs)];
	
	// SE(median)
	SE_median=0;
	for(i=0;i<Nbs;i++){
		tmp=BS_data[i]-middle;
		SE_median+=tmp*tmp;
	}
	SE_median/=(Nbs-1);
	SE_median=sqrt(SE_median);
	lower_SE=middle-SE_median;
	upper_SE=middle+SE_median;
	// assign output values
	*median = middle;
	*BCa_lower = lower_BCa;
	*BCa_upper = upper_BCa;
	*SE_lower = lower_SE;
	*SE_upper = upper_SE;
	*percentile_lower = lower_percentile;
	*percentile_upper = upper_percentile;
	// cleanup malloc'd memory
	free(jack_knife_array);
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

void jack_knife(float xin[], float yin[], int n, int variable, float jack_theta[], 
	       float (*func)(float *, float *, int, int)){
	int i,j;
	float popA[n],popB[n];
	for(i=0;i<n;i++){
		for(j=0;j<n-1;j++){
			if(j!=i){// so long as we aren't equal, set populations
				popA[j]=xin[j];
				popB[j]=yin[j];
			}
		}
		popA[i]=xin[n-1];
		popB[i]=yin[n-1];
		// calculate func for this iteration
		jack_theta[i]=func(popA,popB,n-1,variable);
	}
}

float jack_knife_wrapper(float xin[], float yin[], int n, int variable){
	float results[3];
	calc_BLUE_slope_intercept(xin,yin,n,results);
	return results[variable];
}

__host__ __device__ void calc_BLUE_slope_intercept(
	float xin[], float yin[], int n, float results[]
	){
	int i;
	float Sx,Sy,Sx2,Sxy,slope,intercept,pearson_rho;
	float sd_x,sd_y,x_bar,y_bar,x,y;
	Sx=0; Sy=0;
	Sx2=0; Sxy=0;
	sd_x=0; sd_y=0;
	x_bar=0; y_bar=0;
	x=0; y=0;
	for(i=0;i<n;i++){
		Sx +=xin[i];
		Sy +=yin[i];
		Sx2+=xin[i]*xin[i];
		Sxy+=xin[i]*yin[i];
	}
	x_bar=Sx/n; y_bar=Sy/n;
	for(i=0;i<n;i++){
		x=xin[i]-x_bar;
		y=yin[i]-y_bar;
		sd_x+=x*x;
		sd_y+=y*y;
	}
	sd_x = sqrt(sd_x/(n-1));
	sd_y = sqrt(sd_y/(n-1));
	slope=(n*Sxy-Sx*Sy)/(n*Sx2-Sx*Sx);
	intercept=(Sy/n)-slope*(Sx/n);
	pearson_rho=(Sxy-n*x_bar*y_bar)/((n-1)*sd_x*sd_y);
	results[SLOPE] = slope;
	results[INTERCEPT] = intercept;
	results[PEARSON_RHO] = pearson_rho;
}

void checkCUDAError(const char *msg){
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err){
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
