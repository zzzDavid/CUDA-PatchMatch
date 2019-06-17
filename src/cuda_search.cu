#include "cuda_search.h"


__global__ void sayHelloWorld() {
	printf("HelloWorld! GPU \n");
	//cout << "HelloWorld! GPU" << endl;     //不能使用cout, std命名不能使用到GPU上
}


int cuda_full_search(unsigned char *out, unsigned char *in0, unsigned char *in1, int width, int height) {

	// do something here
	printf("HelloWorld! CPU \n");

	// call kernel function
	sayHelloWorld << <1, 10 >> > ();   //调用GPU上执行的函数，调用10个GPU线程

	// explicitly free resources of current thread
	cudaDeviceReset();

}

