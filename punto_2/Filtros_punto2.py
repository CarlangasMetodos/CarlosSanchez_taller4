"""
Segundo Punto - Filtro
Filtro de altas y bajas
"""
import argparse
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray


#Calcula el indice de la grilla del kernel
def calculate_index(odd_int):
    return (odd_int-1)/2

#Crea el kernel para hacer la convolucion, se toma gaussiana con valor de sigma=1, se tienen en cuenta los tipos de filtro 

def compute_kernel(x_line, y_line, filter_type, sigma=1,):
    x_grid, y_grid = np.meshgrid(x_line, y_line)
    non_normalized = np.exp((-1/(2*(sigma**2))) * (x_grid**2 + y_grid**2))
    normalized_low = non_normalized / np.sum(non_normalized)
    if filter_type == "bajo":
        return normalized_low
    else:
        non_normalized = 1 - normalized_low
        return non_normalized / np.sum(non_normalized)

#Trae la imagen como una png en escala de grises y retorna su arreglo

def image_preprocessing(file_path):
    img = mpimg.imread(file_path)
    img_gray = rgb2gray(img)
    return img_gray

"""
#Metodo que calcula la transformada de Fourier en una dimension

N=len(ndarray)
g=ndarray
n=[]
k=np.linspace(0,N-1,N)
Fourier=[]

for i in range (N):
	n.append(datos[i])

#for i in range(N):
exponencial=np.exp(-1j*2*np.pi*k*n/N)
mult=exponencial*g
Fourier.append(mult)
"""

#Metodo para hacer la transformada de Fourier en una dimension, a partir de la multiplicacion entre dos arreglos 

def fourier_transform_1d(ndarray):
    N = ndarray.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-1j * 2* np.pi * k * n / N)
    return np.dot(M, ndarray)

#Calcula la transformada inversa de Fourier en una dimension, a partir de la interpretacion previa

def inv_fourier_transform_1d(ndarray):
    """Realiza la transformada inversa de Fourier"""
    N = ndarray.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(1j * 2 * np.pi * k * np.divide(n, N))
    return np.divide(np.dot(M, ndarray), N)


#Calcula la transformada de Fourier en dos dimensiones

def fourier_transform_2d(img_np):
	#Devuelve un arreglo de ceros con las mismas caracteristicas que el dado
    output = np.zeros_like(img_np, dtype="complex")
	#Enumerate cuenta y guarda los datos del arreglo

    for counter, single_ndarray in enumerate(img_np):
        complex_row = fourier_transform_1d(single_ndarray)
        output[counter] = complex_row

    for counter, single_ndarray in enumerate(output.T):
        complex_column = fourier_transform_1d(single_ndarray)
        output[:, counter] = complex_column

    return output

#Hace la transformada inversa de Fourier en dos dimensiones
def inv_fourier_transform_2d(img_np):
    output = np.zeros_like(img_np, dtype="complex")
    for counter, single_ndarray in enumerate(img_np):
        complex_row = inv_fourier_transform_1d(single_ndarray)
        output[counter] = complex_row

    for counter, single_ndarray in enumerate(output.T):
        complex_column = inv_fourier_transform_1d(single_ndarray)
        output[:, counter] = complex_column

    return output


#Funcion main que hace todo, incluido el filtro que se pida, para una funcion principal de python

def main(file_path, filter_type):

    # Creamos la grilla para evaluar nuestras funciones
    index = calculate_index(5)
    x_range = np.arange(-index, index + 1, 1)
    y_range = np.arange(-index, index + 1, 1)
    kernel = compute_kernel(x_range, y_range,filter_type, sigma=3)

    # Procesamiento de imagenes
    img_np = image_preprocessing(file_path)
    embedded_kernel = np.zeros_like(img_np)
    embedded_kernel[0:kernel.shape[0], 0: kernel.shape[1]] = kernel

    # FT
    fftimage = fourier_transform_2d(img_np)
    fftkernel = fourier_transform_2d(embedded_kernel)
    fftblurimage = np.multiply(fftimage, fftkernel)

    # De vuelta por inversa
    blurimage = inv_fourier_transform_2d(fftblurimage)

    #Aplica el filtro, segun lo que se pida

    if filter_type == "bajo":
        f_name = "bajo.png"
    else:
        f_name = "alto.png"

    #Guarda la imagen
    mpimg.imsave(f_name, blurimage.astype("float"))

#Especifica que comandos vamos a usar en el programa, en este caso la ruta de imagen y el tipo de filtro que se quiere aplicar (filter_type)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("file_path", help="Ruta del archivo")
    PARSER.add_argument("filter_type", help="Tipo del filtro")
    ARGS = PARSER.parse_args()
    # print("types", ARGS.file_path, type(ARGS.file_path))
    # print("types", ARGS.n_pixel_kernel, type(ARGS.n_pixel_kernel))
    main(ARGS.file_path, ARGS.n_pixel_kernel)
else:
    main("imagen.png", 3)



