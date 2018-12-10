#include <stdio.h>
#include <stdlib.h>

float my_mean(float datain[], int datasize);
float my_var(float datain[], int datasize);
float my_median(float datain[], int datasize, int issorted);
float my_min(float datain[], int datasize, int issorted);
float my_max(float datain[], int datasize, int issorted);

int cmp_func(const void *a, const void *b);

float my_mean(float datain[], int datasize){
	int i;
	float sum=0;
	for(i=0;i<datasize;i++)
		sum+=datain[i];
	return(sum/datasize);
}

float my_var(float datain[], int datasize){
	int i;
	float sum=0;
	float diff=0;
	float meen=my_mean(datain,datasize);
	for(i=0;i<datasize;i++){
		diff=datain[i]-meen;
		sum+=diff*diff;
	}
	return(sum/(datasize-1));/* subtract 1 to remove bias */
}

float my_median(float datain[], int datasize, int issorted){
	int mid=datasize/2;
	if(issorted==0)	// sort the array
		qsort(datain,datasize,sizeof(*datain),cmp_func);
	if(datasize%2==1)
	  return(datain[mid]);
	return(0.5*(datain[mid]+datain[mid+1]));
}

float my_min(float datain[], int datasize, int issorted){
	float minval=1e30; int i;
	if(issorted==1)
		return(datain[0]);
	for(i=0;i<datasize;i++)
		minval=(datain[i]<minval) ? datain[i]:minval;
	return(minval);
}

float my_max(float datain[], int datasize, int issorted){
	float maxval=-1e30; int i;
	if(issorted==1)
		return(datain[datasize-1]);
	for(i=0;i<datasize;i++)
		maxval=(datain[i]>maxval) ? datain[i]:maxval;
	return(maxval);
}

int cmp_func(const void *a, const void *b){
	float aa = *(float*)a;
	float bb = *(float*)b;
	if( aa<bb )
	  return(-1);
	if( aa>bb )
	  return(1);
	return(0);
}

