#include "cuda_search.h"


__global__ void sayHelloWorld() {
	printf("HelloWorld! GPU \n");
	//cout << "HelloWorld! GPU" << endl;     //����ʹ��cout, std��������ʹ�õ�GPU��
}


int cuda_full_search(unsigned char *out, unsigned char *in0, unsigned char *in1, int width, int height) {

	// do something here
	printf("HelloWorld! CPU \n");

	// call kernel function
	sayHelloWorld << <1, 10 >> > ();   //����GPU��ִ�еĺ���������10��GPU�߳�

	// explicitly free resources of current thread
	cudaDeviceReset();

}

