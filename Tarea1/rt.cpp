// rt: un lanzador de rayos minimalista
 // g++ -O3 -fopenmp rt.cpp -o rt
#include <math.h>
#include <stdlib.h>
#include <stdio.h>  
#include <omp.h>
#include <stdio.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
#include <cstdlib>
#include <algorithm>

inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}


class Vector 
{
public:        
	double x, y, z; // coordenadas x,y,z 
  
	// Constructor del vector, parametros por default en cero
	Vector(double x_= 0, double y_= 0, double z_= 0){ x=x_; y=y_; z=z_; }
  
	// operador para suma y resta de vectores
	Vector operator+(const Vector &b) const { return Vector(x + b.x, y + b.y, z + b.z); }
	Vector operator-(const Vector &b) const { return Vector(x - b.x, y - b.y, z - b.z); }
	// operator multiplicacion vector y escalar 
	Vector operator*(double b) const { return Vector(x * b, y * b, z * b); }

	// operator % para producto cruz
	Vector operator%(Vector&b){return Vector(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);}
	
	// producto punto con vector b
	double dot(const Vector &b) const { return x * b.x + y * b.y + z * b.z; }

	// producto elemento a elemento (Hadamard product)
	Vector mult(const Vector &b) const { return Vector(x * b.x, y * b.y, z * b.z); }
	
	// normalizar vector 
	Vector& normalize(){ return *this = *this * (1.0 / sqrt(x * x + y * y + z * z)); }

	double magnitud() const {
        return sqrt(magnitud2());
        }

    double magnitud2() const {
        return x*x + y*y + z*z;
        }

};


typedef Vector Point;
typedef Vector Color;

inline  Vector cross(const Vector &u, const Vector &v) {
    return Vector(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}

void coordinateSystem(const Vector &n, Vector &s, Vector &t) {
	if (std::abs(n.x) > std::abs(n.y)) {
		float invLen = 1.0f / std::sqrt(n.x * n.x + n.z * n.z);
		t = Vector(n.z * invLen, 0.0f, -n.x * invLen);
	} else {
		float invLen = 1.0f / std::sqrt(n.y * n.y + n.z * n.z);
		t = Vector(0.0f, n.z * invLen, -n.y * invLen);
	}
	s = cross(t, n);
	}


class Ray 
{ 
public:
	Point o;
	Vector d; // origen y direcccion del rayo
	Ray(Point o_, Vector d_) : o(o_), d(d_) {} // constructor
};

class Sphere 
{
public:
	double r;	// radio de la esfera
	Point p;	// posicion
	Color c;	// color  

	Sphere(double r_, Point p_, Color c_): r(r_), p(p_), c(c_) {}
  
	// PROYECTO 1
	// determina si el rayo intersecta a esta esfera
	double intersect(const Ray &ray) const {
		Vector oc = ray.o-p;

		double a = ray.d.dot(ray.d);
		double b =  oc.dot(ray.d);
		double c = oc.dot(oc)-r*r;
		double discriminant= b*b - a*c;
		// regresar distancia si hay intersección
		// regresar 0.0 si no hay interseccion
		if (discriminant<0) {
			return 0.0;
		}
		else{
			double tpositivo = -b + sqrt(discriminant);
			double tnegativo = -b - sqrt(discriminant);
			double t;
			if (tpositivo > 0.0 && tnegativo > 0.0 )
			t = (tpositivo < tnegativo) ? tpositivo : tnegativo;
			else 
			t = (tpositivo > tnegativo) ? tpositivo : tnegativo;
			if (t < 0.0) return 0.0;
			else return t;
		}
	}
};

Sphere spheres[] = {
	//Escena: radio, posicion, color 
	Sphere(1e5,  Point(-1e5 - 49, 0, 0),   Color(.75, .25, .25)), // pared izq
	Sphere(1e5,  Point(1e5 + 49, 0, 0),    Color(.25, .25, .75)), // pared der
	Sphere(1e5,  Point(0, 0, -1e5 - 81.6), Color(.75, .75, .75)), // pared detras
	Sphere(1e5,  Point(0, -1e5 - 40.8, 0), Color(.75, .75, .75)), // suelo
	Sphere(1e5,  Point(0, 1e5 + 40.8, 0),  Color(.75, .75, .75)), // techo
	Sphere(16.5, Point(-23, -24.3, -34.6), Color(.999, .999, .999)), // esfera abajo-izq
	Sphere(16.5, Point(23, -24.3, -3.6),   Color(.999, .999, .999) ), // esfera abajo-der
	Sphere(10.5, Point(0, 24.3, 0),        Color(1, 1, 1)) // esfera arriba	
};

// limita el valor de x a [0,1]
inline double clamp(const double x) { 
	if(x < 0.0)
		return 0.0;
	else if(x > 1.0)
		return 1.0;
	return x;
}

// convierte un valor de color en [0,1] a un entero en [0,255]
inline int toDisplayValue(const double x) {
	return int( pow( clamp(x), 1.0/2.2 ) * 255 + .5); 
}

// PROYECTO 1
// calcular la intersección del rayo r con todas las esferas
// regresar true si hubo una intersección, falso de otro modo
// almacenar en t la distancia sobre el rayo en que sucede la interseccion
// almacenar en id el indice de spheres[] de la esfera cuya interseccion es mas cercana
inline bool intersect(const Ray &r, double &t, int &id) {
	int NS=8;
	double aux[NS];


	for (int i=0;i<NS;i++){
        aux[i]=spheres[i].intersect(r);
		if ( aux[i]>0){
			t= aux[i];
			id=i;
		};

    };

	for (int i=0;i<NS;i++){
		if ( t>aux[i] && aux[i]>0.0001 ){
			t= aux[i];
			id=i;
		};
    };
	if (t>0){
		//printf("%f  %d\t ",t,id);
		return true;
		};

	return false;
}
void normalizat(double &t,const double &min,const double &max){
	t=(t-min)/(max-min);
}

// Calcula el valor de color para el rayo dado
//Color shade(const Ray &r,double &min,double &max) {//Use este Shade para encontrar el minimo y el maximo de la escena.
Color shade(const Ray &r) {
	double t;
	int id = 0;

	double min=144.676098;
	//double min=0.0;
	double max=304.110891;
	// determinar que esfera (id) y a que distancia (t) el rayo intersecta
	if (!intersect(r, t, id)){
		return Color();}	// el rayo no intersecto objeto, return Vector() == negro

	const Sphere &obj = spheres[id];

	/*if (t<min)
		min=t;   //Lineas de codigo para calcular el minimo y maximo
	if (t>max)
		max=t; */

	// PROYECTO 1
	// determinar coordenadas del punto de interseccion
	Point x; //Linea de codigo para el calculo de las coordenadas 
	x=r.d*t+r.o;
	normalizat(t,min,max);
	//printf("%f,",t);
	//h=sqrt(x.x*x.x+x.y*x.y+x.z*x.z)/103.5; //esta linea no se utilizo
	
	// determinar la dirección normal en el punto de interseccion
	Vector n;
	n=x-obj.p;
	n.normalize();

	//Imagen2.jpg Es necesario descomentar la siguiente linea de codigo para obtener lo equivalente a esta imangen.
	//Color colorvalue(obj.c+n);// Para pintar las esferas de acuerdo a la normal en su punto intersectado

	//Imagen.jpg Es necesario descomentar la siguiente linea de codigo para obtener lo equivalente a esta imangen.
	//Color colorvalue(obj.c);// Para pintar las esferas de acuerdo a su color 

	//Imagen3.jpg Es necesario descomentar la siguiente linea de codigo para obtener lo equivalente a esta imangen.
	//Color colorvalue((1/310.0)*t,(1/310.0)*t,(1/310.0)*t); //Para pintarlo deacuerdo a la distancia
	Color colorvalue(t,t,t);
	//printf("%f,\t,%f\n",t,x.magnitud());
	//Posible variante para la Imagen3.jpg
	//Color colorvalue(1*h,1*h,1*h);//Para pintarlo deacuerdo a la distancia, es redundante y no funciona bien, hay que descomentar h para probarlo

	return colorvalue; 
}


int main(int argc, char *argv[]) {
	double time_spent = 0.0;
	//double min=144.676098;
	//double min=0.0;
	//double max=304.110891;
    clock_t begin = clock();
 
	int w = 1024, h = 768; // image resolution
  
	// fija la posicion de la camara y la dirección en que mira
	Ray camera( Point(0, 11.2, 214), Vector(0, -0.042612, -1).normalize() );

	// parametros de la camara
	Vector cx = Vector( w * 0.5095 / h, 0., 0.); 
	Vector cy = (cx % camera.d).normalize() * 0.5095;
  
	// auxiliar para valor de pixel y matriz para almacenar la imagen
	Color *pixelColors = new Color[w * h];

	// PROYECTO 1
	// usar openmp para paralelizar el ciclo: cada hilo computara un renglon (ciclo interior),
	int NUM_THREADS=omp_get_max_threads();
	fprintf(stderr," \r Vamos a trabajar con %d hilos ",NUM_THREADS);
	omp_set_num_threads(NUM_THREADS); //Lineas de Codigo para paralelizar
	#pragma omp parallel for schedule(dynamic,10)//schedule(dynamic,1) //Con la paralelizacion se reduce en un 60 % aproxiamadamente el tiempo de ejecucion.

	for(int y = 0; y < h; y++) 
	{ 
		// recorre todos los pixeles de la imagen
		fprintf(stderr,"\r%5.2f%%",100.*y/(h-1));
		for(int x = 0; x < w; x++ ) {

			int idx = (h - y - 1) * w + x; // index en 1D para una imagen 2D x,y son invertidos

			Color pixelValue = Color(); // pixelValue en negro por ahora
			// para el pixel actual, computar la dirección que un rayo debe tener

			Vector cameraRayDir = cx * ( double(x)/w - .5) + cy * ( double(y)/h - .5) + camera.d;
			// computar el color del pixel para el punto que intersectó el rayo desde la camara

			pixelValue = shade( Ray(camera.o, cameraRayDir.normalize()));
			//pixelValue = shade( Ray(camera.o, cameraRayDir.normalize()),min,max );//Use este min max para encontrar el min y maximo de la escena
			// limitar los tres valores de color del pixel a [0,1] 
			pixelColors[idx] = Color(clamp(pixelValue.x), clamp(pixelValue.y), clamp(pixelValue.z));
		}
		
	
	
	}


	fprintf(stderr,"\n");
	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("The elapsed time is %f seconds \n", time_spent);
	//printf("El minimo t para la escena  %f es y el maximo es t para la escena es %f",min,max);

	// PROYECTO 1
	// Investigar formato ppm
	FILE *f = fopen("distancia2.ppm", "w");
	// escribe cabecera del archivo ppm, ancho, alto y valor maximo de color
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
	for (int p = 0; p < w * h; p++) 
	{ // escribe todos los valores de los pixeles
    		fprintf(f,"%d %d %d \n", toDisplayValue(pixelColors[p].x), toDisplayValue(pixelColors[p].y), 
				toDisplayValue(pixelColors[p].z));
  	}
  	fclose(f);

  	delete[] pixelColors;

	return 0;
}
