TARGET = k_means

SRC = src
INC = includes
BIN = bin


SOURCE = $(wildcard $(SRC)/*.c)
OBJECT = $(patsubst %,$(BIN)/%, $(notdir $(SOURCE:.c=.o)))

CC = gcc

CFLAGS = -std=gnu99 -O2 -funroll-loops -ftree-vectorize -msse4 -I$(INC)

$(BIN)/$(TARGET) : $(OBJECT)
		$(CC) -o $@ $^

$(BIN)/%.o : $(SRC)/%.c
		$(CC) $(CFLAGS) -c $< -o $@

.PHONY : help run clean

run : $(BIN)/$(TARGET)
		$(BIN)/$(TARGET)

clean :
		rm -f $(OBJECT) $(BIN)/$(TARGET)

help :
		@echo "src: $(SOURCE)"
