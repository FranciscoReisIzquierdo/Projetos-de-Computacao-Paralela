TARGET = k_means

SRC = src
INC = includes
BIN = bin
THREADS = 12


SOURCE = $(wildcard $(SRC)/*.c)
OBJECT = $(patsubst %,$(BIN)/%, $(notdir $(SOURCE:.c=.o)))

CC = gcc

CFLAGS = -std=gnu99 -O2 -funroll-loops -ftree-vectorize -msse4 -I$(INC)

$(BIN)/$(TARGET) : $(OBJECT)
		$(CC) -o $@ $^ -fopenmp

$(BIN)/%.o : $(SRC)/%.c
		$(CC) $(CFLAGS) -c -fopenmp $< -o $@

.PHONY : help run clean

runseq : $(BIN)/$(TARGET)
		$(BIN)/$(TARGET) -fopenmp 10000000 $(CP_CLUSTERS)
		
runpar : $(BIN)/$(TARGET)
		$(BIN)/$(TARGET) -fopenmp 10000000 $(CP_CLUSTERS) $(THREADS)

clean :
		rm -f $(OBJECT) $(BIN)/$(TARGET)

help :
		@echo "src: $(SOURCE)"
