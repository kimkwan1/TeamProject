CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm -lpthread

OBJS = mt64.o libIsing.o 
SRCS = mt64.c libIsing.c

# $(OBJS): $(SRCS)
# 	$(CC) $(CFLAGS) -c $< $(LDFLAGS)
%.o: %.c
	$(CC) $(CFLAGS) -c $< $(LDFLAGS)

main: $(OBJS)
#	$(CC) $(CFLAGS) -c get_Ising_all_dataset.c $(LDFLAGS)
	$(CC) $(CFLAGS) $(OBJS) get_Ising_all_dataset.o -o main $(LDFLAGS)

clean:
	rm $(OBJS)
	rm main

.PHONY: main
