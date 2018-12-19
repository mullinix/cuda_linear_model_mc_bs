linboot: lin-bootstrap.cu
	nvcc -o linboot lin-bootstrap.cu -lm -lgsl -lgslcblas
