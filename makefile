boot2slopes: bootstrap_2_slopes.cu
	nvcc -o boot2slopes bootstrap_2_slopes.cu -lm -lgsl -lgslcblas
linboot: lin-bootstrap.cu
	nvcc -o linboot lin-bootstrap.cu -lm -lgsl -lgslcblas
