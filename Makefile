
all:
	gcc -O3 -fopenmp -Wall -Wextra -Wpedantic $(wildcard src/*.c) -lm -lpthread -lSDL2 -lSDL2_ttf -lmpfr

