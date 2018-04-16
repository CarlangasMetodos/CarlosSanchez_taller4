#Archivo makefiles

transformada.txt: fourier
	./fourier 

fourier:Lagrange_punto3.cpp datos.txt
	g++ Lagrange_punto3.cpp

