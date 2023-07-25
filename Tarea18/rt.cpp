// rt: un lanzador de rayos minimalista
 // g++ -O3 -fopenmp rt.cpp -o rt
#include <math.h>
#include <stdlib.h>
#include <stdio.h>  
#include <omp.h>
#include <utility>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
//#define NUM_THREADS 12
#include <cstdlib>
#include <cmath>
#include<algorithm>
using namespace std;
#include "material.h"



class Sphere 
{
public:
	double r;	// radio de la esfera
	Point p;	// posicion 
	material *m;	// Material de la sefera ****Proyecto 2*******

	Sphere(double r_, Point p_,material* m_): r(r_), p(p_), m(m_) {}

  
	double intersect(const Ray &ray) const {
		Vector oc = ray.o-p;

		double a = ray.d.dot(ray.d);
		double b =  oc.dot(ray.d);
		double c = oc.dot(oc)-r*r;
		double discriminant= b*b - a*c;
		// regresar distancia si hay intersección
		// regresar 0.0 si no hay interseccion
		if (discriminant<0.0) {
			return 0.0;
		}
		else{
			double tpositivo = -b + sqrt(discriminant);
			double tnegativo = -b - sqrt(discriminant);
			double t;
			if (tpositivo > 0.0001 && tnegativo > 0.0001 )
			{
			t = (tpositivo < tnegativo) ? tpositivo : tnegativo;
			}
			else if(tpositivo >0.0001 && tnegativo < 0.00001) 
			{
			t = tpositivo;
			}
			else if(tpositivo < 0.0001 && tnegativo > 0.00001)
			{
			t = tnegativo;
			}
			else
			{
				t=0;
			}
			return t;

		}
	
	}
};

Luz  Esferaluminoza(Color(10.0, 10.0, 10.0));
Conductor EsAbaIz(Color(1.66058,0.88143,0.531467),Color(9.2282,6.27077,4.83803));//Aluminio
//MicroFasetC EsAbaIz(Color(1.66058,0.88143,0.531467),Color(9.2282,6.27077,4.83803),0.3);//Aluminio
//MicroFasetC EsAbaDer(Color(1.66058,0.88143,0.531467),Color(9.2282,6.27077,4.83803),0.3);//Aluminio
MicroFasetC EsAbaDer(Color(0.143245,0.377423,1.43919),Color(3.98478,2.3847,1.60434),0.1);//Oro
//MicroFasetC EsAbaIz(Color(0.143245,0.377423,1.43919),Color(3.98478,2.3847,1.60434),0.3);//Oro
//Conductor Espeq(Color(0.143245,0.377423,1.43919),Color(3.98478,2.3847,1.60434));//Oro
//Conductor Esferacristal(Color(0.208183,0.919438,1.110241),Color(3.92198,2.45627,2.14157));//Cobre
//Luz  Esferaluminoza(Color(1.0, 1.0, 1.0));
//double etam=1.5,etai=1.00029;
Dielectrico Espeq(1.5,1.0);

Difuso ParIzq(Color(.75, .25, .25));//roja
//Abedo ParDer(Color(.25, .75, .25),0.5);//verde
Difuso ParDer(Color(.25, .25, .75));//azul
//Abedo ParedAt(Color(1.0, 1.0, 1.0),0.5);//blanco
//Abedo Suelo(Color(1.0, 1.0, 1.0),0.5);//blanco
//Abedo Techo(Color(1.0, 1.0, 1.0),0.5);//blanco
Difuso ParedAt(Color(.25, .75, .25));//verde
Difuso Suelo(Color(.25, .75, .75));//verde bajito
Difuso Techo(Color(.75, .75, .25));//amarillo
//Difuso EsAbaIz(Color(.2, .3, .4));//(.2, .3, .4)
//OrenNayar EsAbaIz(Color(.4, .3, .2),0.5);
//OrenNayar EsAbaDer(Color(.2, .3, .4),0.8);


Sphere spheres[] = {
	//Escena: radio, posicion ,material    //Fara facilitarme las cosas quite el Color, y se lo agregue en el material.
        Sphere(1e5,  Point(-1e5 - 49, 0, 0),     &ParIzq), // pared izq
        Sphere(1e5,  Point(1e5 + 49, 0, 0),      &ParDer), // pared der
        Sphere(1e5,  Point(0, 0, -1e5 - 81.6),   &ParedAt), // pared detras
        Sphere(1e5,  Point(0, -1e5 - 40.8, 0),   &Suelo), // suelo
        Sphere(1e5,  Point(0, 1e5 + 40.8, 0),    &Techo), // techo
        Sphere(18.5, Point(-23, -22.3, -34.6),   &EsAbaIz), // esfera abajo-izq
		//Sphere(16.5, Point(-23, -24.3, -34.6),   &Esferaluminoza), // esfera abajo-izq
        //Sphere(18.5, Point(23, -22.3, -30.6),     &EsAbaDer), // esfera abajo-der// Para observar las dos fuentes luminosas hay que comentar esta linea
		//Sphere(12.5, Point(23, -28.3, -30.6),     &EsAbaDer), // esfera abajo-der// Para observar las dos fuentes luminosas hay que comentar esta linea
		Sphere(16.5, Point(23, -24.3, -3.6),     &EsAbaDer), // esfera abajo-der // Para observar las dos fuentes luminosas hay que descomentar esta linea
        Sphere(10.5, Point(0, 24.3, 0),          &Esferaluminoza), // esfera arriba // esfera iluminada
		Sphere(7.5, Point(-23.0, -33.2, 30.0),          &Espeq)
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


inline bool intersect(const Ray &r, double &t, int &id) {
	double aux;
	double limite =  100000000000;
	t = limite;
	int NS=9;

	for (int i=0;i < NS;i++) {
		aux = spheres[i].intersect(r);

		if (aux && aux > 0.001 && aux< t) {
			t = aux;
			id = i;
		}
	}
	if (t < limite)
		return true;
	else 
		return false;
}

// inline bool intersect(const Ray &r, double &t, int &id) {
// 	int NS=8;
// 	double aux[NS];


// 	for (int i=0;i<NS;i++){
//         aux[i]=spheres[i].intersect(r);
// 		if ( aux[i]>0.001){
// 			t= aux[i];
// 			id=i;
// 		};
//     };

// 	for (int i=0;i<NS;i++){
// 		if ( t>aux[i] && aux[i]>0.001 ){
// 			t= aux[i];
// 			id=i;
// 		};
//     };
// 	if (t>0){
// 		//printf("%f  %d\t ",t,id);
// 		return true;
// 		};

// 	return false;
// }

Color shade(const Ray &r,int prof) { //Agregamos la profundidad para hacer una funcion recursiva, esto nos permite lanzar un segundo rayo desde
	registro rec;
	double t; 						 						 
	int id = 0;						 

	if (prof <= 0) // Si ya se ha llegado 
        return Color();

	if (!intersect(r, t, id)){
		return Color();
		}	// el rayo no intersecto objeto, return Vector() == negro

	const Sphere &obj = spheres[id];

	rec.x=r.d*t+r.o; 
	rec.n=(rec.x-obj.p).normalize();
	rec.t=t;

    Color emite = obj.m->Emite(rec.x);

	Ray rebota;
	Color attenuation;

    if (!obj.m->Rebota(r, rebota,rec)) 
        return emite;   
	
	double pdf=obj.m->PDF(rebota,rec);
	attenuation=obj.m->BDRF(r,rebota,rec);

	double Coseno=rec.n.dot(rebota.d);
	
	return  emite+ attenuation.mult(shade(rebota, prof-1))*(fabs(Coseno)/pdf);
}

// Color shade(const Ray &r) { /// path traicing interativo sesgo
	
// 	double t; 						 						 
// 	int id = 0;	
// 	Color attenuation;
// 	Ray rebota=r;
// 	registro rec;
// 	Color emite;
// 	double q=0.1;
// 	double continueprob=1.0-q;
// 	double Coseno;
// 	double pdf;

// 	Color troughpout(1.0,1.0,1.0);
// 	if (!intersect(r, t, id)){
// 		return Color();
// 		}
// 	do{			 

// 	const Sphere &obj = spheres[id];

// 	rec.x=rebota.d*t+rebota.o; 
// 	rec.n=(rec.x-obj.p).normalize();
// 	rec.t=t;
	
//     emite = obj.m->Emite(rec.x);

// 	// if (emite.x > 0.0 && emite.y > 0.0 && emite.z > 0.0) {
// 	//  	break; 
// 	//  }

//     if (!obj.m->Rebota(r, rebota,rec)) {
// 		break; 
// 	}
// 		obj.m->Rebota(r, rebota,rec);
// 		pdf=obj.m->PDF(rebota,rec);
// 		attenuation=obj.m->BDRF(r,rebota,rec);	
// 		Coseno=rec.n.dot(rebota.d);	
	
// 		troughpout=troughpout*attenuation*(fabs(Coseno)/(continueprob*pdf));//

// 	if (random_double()<q) 
// 		break;
	
// 	if (!intersect(rebota, t, id)){
// 		break;
// 		}

// 	}while(true) ;
	
// 	return  emite*troughpout;
// }





int main(int argc, char *argv[]) {
	double time_spent = 0.0;
	double muestras = 512.0;
	double invMuestras=1.0/(muestras);
	int prof=10;
    clock_t begin = clock();

 
	int w = 1024, h = 768; // image resolution
  
	// fija la posicion de la camara y la dirección en que mira
	Ray camera( Point(0, 11.2, 214), Vector(0, -0.042612, -1).normalize() );

	// parametros de la camara
	Vector cx = Vector( w * 0.5095 / h, 0., 0.); 
	Vector cy = (cx % camera.d).normalize() * 0.5095;
  
	// auxiliar para valor de pixel y matriz para almacenar la imagen
	Color *pixelColors = new Color[w * h];

	int NUM_THREADS=omp_get_max_threads();
	fprintf(stderr," \r Vamos a trabajar con %d hilos \n",NUM_THREADS);
	omp_set_num_threads(NUM_THREADS); //Lineas de Codigo para paralelizar
	#pragma omp parallel
	{ //for // schedule(dynamic,1) //Con la paralelizacion se reduce en un 60 % aproxiamadamente el tiempo de ejecucion.
	#pragma omp for //schedule(dynamic,1)
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
			for (int i=0; i<muestras;i++)
			{
				pixelValue = pixelValue + shade( Ray(camera.o, cameraRayDir.normalize()),prof)*invMuestras;
			// limitar los tres valores de color del pixel a [0,1] 
			}
			//pixelValue = pixelValue;

			pixelColors[idx] = Color(clamp(pixelValue.x), clamp(pixelValue.y), clamp(pixelValue.z));
		}
		
	
	
	}

}
	fprintf(stderr,"\n");
	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("The elapsed time is %f seconds", time_spent);

	// PROYECTO 1
	// Investigar formato ppm
	FILE *f = fopen("Dielectrica.ppm", "w");
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
// Color puntual2(const Ray &r,registro &rec,const int &id) { 


// 	const Sphere &obj = spheres[id];


// 	Point luz(0, 24.3,0);// Creamos la fuente puntual, en este caso es la direccion a la que vamos a muestrear, como este punto ya esta en coordenadas

//     Color attenuation;
// 	Vector x1=(rec.x-luz); //Creamos la direccion en la que va nuestra el rayo de luz, en la ecuacion Le(x',-wi) x'
// 	Ray rebota(luz,x1); 

// 	Color emite(5000,5000,5000);


// 	double ta;
// 	int id2=0;
// 	double dist=0;
// 	if (!intersect(rebota, ta, id2)){//lanzamos nuevamente el rayo pero ahora desde la luz hacia la esfera. Si no toca nada regresamos negro.
// 		return Color();
// 	}

// 	const Sphere &obj2 = spheres[id2];
// 	Point x2=rebota.d*ta+rebota.o;



// 	if(obj2.m->Rebota(r,rebota,rec)){//Si la esfera que tocamos la primera vez es igual a la esfera que tocamos la segunda vez regresamos color si no no regresamos nada
		
// 	//return emite*(1.0/dist);
// 	return Color();
// 		}else{
// 			dist=(luz-x2).magnitud();
// 			return emite*(1.0/dist);
// }
		
    


    

// }

// Color Area(const Ray &r,registro &rec,const int &id) { //,const registro &rec

// 	const Sphere &obj = spheres[id];
// 	rec.n=(rec.x-obj.p).normalize();

// 	double radio=10.5; 
// 	Vector luz(0, 24.3,0);
//     double costmax;

//     Vector w=luz-rec.x;
//     Vector re=random_asolido(w,radio,costmax).normalize();
// 	w.normalize();

// 	Vector s; 
// 	Vector ti;
// 	coordinateSystem(w,s,ti);
// 	Vector dir(s*re.x+ti*re.y+w*re.z);
// 	dir.normalize();

//     Ray rebota(rec.x,dir);

// 	double pw=1.0 / (2.0 * pi * (1.0 - costmax));
	
// 	double ta; 
// 	int id2;
// 		if (!intersect(rebota, ta, id2)){
// 		return Color();
// 		}

// 	const Sphere &obj2 = spheres[id2];

// 	Color emite2 = obj2.m->Emite(rec.x);

//     if (!obj2.m->Rebota(r, rebota,rec)) 
//     return emite2*(1/pw); 
// 	else return Color();

// }