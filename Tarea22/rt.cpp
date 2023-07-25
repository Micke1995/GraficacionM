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
#include "volumen.h"


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
			double tol=0.003;
			if (tpositivo > tol && tnegativo > tol )
			{
			t = (tpositivo < tnegativo) ? tpositivo : tnegativo;
			}
			else if(tpositivo >tol && tnegativo < tol) 
			{
			t = tpositivo;
			}
			else if(tpositivo < tol && tnegativo > tol)
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

Luz  Esferaluminoza(Color(10.0, 10.0, 10.0));///Area
//Luz  Esferaluminoza(Color(50.0, 50.0, 50.0));//Puntual
Conductor EsAbaIz(Color(1.66058,0.88143,0.531467),Color(9.2282,6.27077,4.83803));//Aluminio
//MicroFasetC EsAbaIz(Color(1.66058,0.88143,0.531467),Color(9.2282,6.27077,4.83803),0.3);//Aluminio
//MicroFasetC EsAbaDer(Color(1.66058,0.88143,0.531467),Color(9.2282,6.27077,4.83803),0.3);//Aluminio
//MicroFasetC EsAbaDer(Color(0.143245,0.377423,1.43919),Color(3.98478,2.3847,1.60434),0.1);//Oro
//MicroFasetC EsAbaIz(Color(0.143245,0.377423,1.43919),Color(3.98478,2.3847,1.60434),0.3);//Oro
//Conductor Espeq(Color(0.143245,0.377423,1.43919),Color(3.98478,2.3847,1.60434));//Oro
//Conductor Esferacristal(Color(0.208183,0.919438,1.110241),Color(3.92198,2.45627,2.14157));//Cobre
//Luz  Esferaluminoza(Color(1.0, 1.0, 1.0));
//double etam=1.5,etai=1.00029;
//Dielectrico Espeq(1.5,1.0);
Dielectrico EsAbaDer(1.5,1.0);

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
//Difuso EsAbaDer(Color(.2, .3, .4));


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
        Sphere(10.5, Point(0, 24.3, 0),          &Esferaluminoza) // esfera arriba // esfera iluminada
		//Sphere(0.007, Point(0, 24.3, 0),          &Esferaluminoza) // esfera puntual
		//Sphere(7.5, Point(-23.0, -33.2, 30.0),          &Espeq)
};

Sphere Luces[]={Sphere(10.5, Point(0, 24.3, 0), &Esferaluminoza)};
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
	if(x != x)return 0.0;
	return int( pow( clamp(x), 1.0/2.2 ) * 255 + .5); 
}


inline bool intersect(const Ray &r, double &t, int &id) {
	double aux;
	double limite =  100000000000;
	t = limite;
	int NS=8;

	for (int i=0;i < NS;i++) {
		aux = spheres[i].intersect(r);

		if (aux && aux > 0.0000001 && aux< t) {
			t = aux;
			id = i;
		}
	}
	if (t < limite)
		return true;
	else 
		return false;
}
Color puntual(const Ray &r,registro &rec,const int &id) { 


	const Sphere &obj = spheres[id];


	const Sphere &luz = Luces[0];

	Vector x1=(rec.x-luz.p); //Creamos la direccion en la que va nuestra el rayo de luz, en la ecuacion Le(x',-wi) x'

	Ray rebota(luz.p,x1); 

	double ta;
	int id2;
	double dist;
	if (!intersect(rebota, ta, id2)){//lanzamos nuevamente el rayo pero ahora desde la luz hacia la esfera. Si no toca nada regresamos negro.
		return Color();
	}
	//printf("%d,%d\n",id,id2);
	const Sphere &obj2 = spheres[id2];// esfera a la que llego 
	Point x2=rebota.d*ta+rebota.o;

	if(!obj2.m->Rebota(r,rebota,rec)){//Si la esfera rebota el rayo devuelve true, si es un emisor devuelve false
		//printf("%d,%d,%f\n",id,id2,ta);
		dist=(luz.p-x2).magnitud();
		Color emite = obj2.m->Emite(x2);
		return emite*(1.0/(dist));//emite*(1.0/(dist));
	
		}else{

		return Color();
}
		
}

Color PT(const Ray &r,int prof) { //Agregamos la profundidad para hacer una funcion recursiva, esto nos permite lanzar un segundo rayo desde
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
	
	return  emite+ attenuation.mult(PT(rebota, prof-1))*(fabs(Coseno)/pdf);
}

Color Area(const Ray &r,registro &rec,const int &id) { //,const registro &rec

	const Sphere &obj = spheres[id];
	rec.n=(rec.x-obj.p).normalize();

	double radio=10.5; 
	Vector luz(0, 24.3,0);
    double costmax;

    Vector w=luz-rec.x;
    Vector re=random_asolido(w,radio,costmax).normalize();
	w.normalize();

	Vector s; 
	Vector ti;
	coordinateSystem(w,s,ti);
	Vector dir(s*re.x+ti*re.y+w*re.z);
	dir.normalize();

    Ray rebota(rec.x,dir);

	double pw=1.0 / (2.0 * pi * (1.0 - costmax));
	
	double ta; 
	int id2;
		if (!intersect(rebota, ta, id2)){
		return Color();
		}

	const Sphere &obj2 = spheres[id2];

	Color emite2 = obj2.m->Emite(rec.x);

    if (!obj2.m->Rebota(r, rebota,rec)) 
    return emite2*(1/pw) ;//
	else return Color();
	// return Color(1,1,1);
	// else return  Color(0,0,0);

}
// double EquiAngularSampling(Ray& ray, Vector3& lightPos, double t_MAX,double sigmaT, double& pdf,double& temp ){
double muestraEquiAngular(const Ray &r,const int &id, double t_MAX, double& pdf,double& temp ){
    // Considerando a=0 y b=t, la muestra EquiAngular esta dada de la siguiente manera
	double t_sample;

    
	Vector luz(0, 24.3,0);
    // get coord of closest point to light along (infinite) ray
    double delta = (luz - r.o).dot(r.d);
    
    // get distance this point is from light
    double D = (r.o + r.d * delta - luz).magnitud2();
    
    // get angle of endpoints
    double thetaA = - atan(delta / D);
    double thetaB = atan((t_MAX - delta) / D);
    
    // take sample
    double xi = random_double();
    double t = D * tan(thetaA *(1 - xi)+ thetaB * xi);
    
//    // debug
    temp = (D*D + t*t);
    
    t_sample = delta + t;
	
    pdf = D/((thetaB - thetaA)*(D*D + t*t));
    
    return t_sample;
};

double distanceSampling(double b,double sigmaT, double& pdf){
    double t_sample;
    
    t_sample = - log(1.0 - random_double() * (1.0 - exp(- sigmaT * b))) / sigmaT;
    pdf = sigmaT * exp( - sigmaT * t_sample) / (1.0 - exp( - sigmaT * b));
	

    return t_sample;
};




Volume VolumenHomogeneo(0.0001,0.0009);

Color shade(const Ray &r, int prof,int paso){

	registro rec;
	double t; 						 						 
	int id = 0;						 

	// if (prof <= 0) // Si ya se ha llegado 
    //     return Color();

	if (!intersect(r, t, id)){
		return Color();
		}	// el rayo no intersecto objeto, return Vector() == negro
   
    Color Li(0,0,0);
  
	const Sphere &obj = spheres[id];

	rec.x=r.d*t+r.o; 
	rec.n=(rec.x-obj.p).normalize();
	rec.t=t;
	


    double transmittance = VolumenHomogeneo.TrasmitanciaHomogenea(r.o, rec.x);
	Color Background=PT(r,prof);
    Li = Li + Background* transmittance;// Color(0,0,0)* transmittance;//
	return Background;
    registro rec_trozo;
    double stepSize = rec.t / paso;
	double pdf,temp;

    rec_trozo.t = 0.0;
    rec_trozo.x = r.o;
    double tr = 0.0;
	Vector luz(0, 24.3, 0);
	
	double st=VolumenHomogeneo.getSigmaT();
	double sigmaS = VolumenHomogeneo.getSigmaS();
	double pf = VolumenHomogeneo.funcionfase();

    for(int i = 0 ; i < paso - 1; i++){
        //rec_trozo.t += stepSize;
		rec_trozo.t = muestraEquiAngular(r,id,rec.t,pdf,temp);
		//rec_trozo.t = distanceSampling(rec.t,st,pdf);
        
		rec_trozo.x = r.o + r.d * rec_trozo.t;
        tr = VolumenHomogeneo.TrasmitanciaHomogenea(rec_trozo.t);//Extincion para Distance Sampling y EquiAngular Volumetric


		//tr = VolumenHomogeneo.TrasmitanciaHomogenea(r.o,rec_trozo.x);//Extincion Para Fuerza Bruta
        double tL = VolumenHomogeneo.TrasmitanciaHomogenea(rec_trozo.x, luz);//;
    	Color Le= Area(r, rec_trozo,id);
		//Color Le=  puntual(r, rec_trozo,id);

        Li =  Li + Le * tr * sigmaS * pf * tL*(1/(pdf*paso));//Li Para Distance Sampling y EquiAngular Volumetric

		//Li =  Li + Le * tr * sigmaS * pf * tL*stepSize;//Li Para Fuerza Bruta
    
    }

   return Li;
}






int main(int argc, char *argv[]) {
	double time_spent = 0.0;
	double muestras = 64.0;
	double invMuestras=1.0/muestras;
	int paso=20;
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
				pixelValue = pixelValue + shade( Ray(camera.o, cameraRayDir.normalize()),prof,paso)*invMuestras;
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
