package main

import (
	"fmt"
	"math"
	"sync"
)

func isPrime(number int) bool {
	if number <= 1 {
		return false
	}
	for i := 2; i <= int(math.Sqrt(float64(number))); i++ {
		if number%i == 0 {
			return false
		}
	}
	return true
}

func findPrimes(start, end int, wg *sync.WaitGroup, primesChan chan<- int) {
	defer wg.Done()
	for number := start; number <= end; number++ {
		if isPrime(number) {
			primesChan <- number
		}
	}
}

func main() {
	const start = 1
	const end = 10000
	const numGoroutines = 10
	var wg sync.WaitGroup
	primesChan := make(chan int, 100)

	// Divide the range among Goroutines
	step := (end - start + 1) / numGoroutines
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go findPrimes(start+i*step, start+(i+1)*step-1, &wg, primesChan)
	}

	// Close channel when all Goroutines are done
	go func() {
		wg.Wait()
		close(primesChan)
	}()

	// Collect and print prime numbers
	for prime := range primesChan {
		fmt.Println(prime)
	}
}
