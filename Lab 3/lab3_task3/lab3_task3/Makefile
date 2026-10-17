# On CAAS first run: module load openmpi/4.1.5-gcc-11.2.0-ux65npg
wsn: main.c task1.c task2.c wsn.h
	mpicc -O2 -Wall -fopenmp -o wsn main.c task1.c task2.c

clean:
	rm -f wsn
