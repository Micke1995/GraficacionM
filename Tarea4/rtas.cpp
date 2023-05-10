// rt: un lanzador de rayos minimalista
 // g++ -O3 -fopenmp rt.cpp -o rt
#include <math.h>
#include <stdlib.h>
#include <stdio.h>  
#include <omp.h>
#include <stdio.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
#define NUM_THREADS 4
#include <cstdlib>

double pi=3.141592653589793238; //Creamos el valor de pi para facilitarnos varais cosas.

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

	//operador para multiplicacion vector vector
	Vector operator*(const Vector &b) const { return Vector(x * b.x , y * b.y , z * b.z); }

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

void coordinateSystem(const Vector &n, Vector &s, Vector &t) { //Esta es la funcion para crear el sistema de coordenadas locales
	if (std::abs(n.x) > std::abs(n.y)) {
		float invLen = 1.0f / std::sqrt(n.x * n.x + n.z * n.z);
		t = Vector(n.z * invLen, 0.0f, -n.x * invLen);
	} else {
		float invLen = 1.0f / std::sqrt(n.y * n.y + n.z * n.z);
		t = Vector(0.0f, n.z * invLen, -n.y * invLen);
	}
	s = cross(t, n);
	}
void ortonormalSystem(Vector &w,Vector &u,Vector &v){
    w=w.normalize();
    Vector a = (fabs(w.x) > 0.9) ? Vector(0,1,0) : Vector(1,0,0);
    v = (cross(w, a)).normalize();
    u = cross(w, v);
}


inline Vector random_esfera(double ro=1.0) { //Esta funcion crea direcciones aleatorias esfericas.
    auto r1 = random_double();
    auto r2 = random_double();

    auto x = ro*cos(2*pi*r1)*2*sqrt(r2*(1-r2));
    auto y = ro*sin(2*pi*r1)*2*sqrt(r2*(1-r2));
    auto z = ro*(1 - 2*r2);

    return Vector(x, y, z);
}


inline Vector random_hemisferio() {//Esta funcion crea direcciones aleatorias en un hemisferio.
    auto r1 = random_double();
    auto r2 = random_double();

    auto theta=acos(1-r1);
    auto phi=2*pi*r2;

    auto x = sin(theta)*cos(phi);
    auto y = sin(theta)*sin(phi);
    auto z = 1 - r1;

    return Vector(x, y, z);
}

inline Vector random_coseno(double &theta) {//Esta funcion crea direcciones aleatorias con distribucion coseno hemisferico.
    auto r1 = random_double();
    auto r2 = random_double();

    auto phi = 2*pi*r1;

    auto z = sqrt(1-r2);
    auto x = cos(phi)*sqrt(r2);
    auto y = sin(phi)*sqrt(r2);


    return Vector(x, y, z);
}

inline Vector random_asolido(const Point &w,const double &r,double &arg,double &thetamax){
    auto r1 = random_double();
    auto r2 = random_double();

    arg=(r/w.magnitud());
    thetamax=asin(arg);

    double theta=acos(1-r1+r1*sqrt(1-arg*arg));
    double phi=2*phi*r2;

    auto x = sin(theta)*cos(phi);
    auto y = sin(theta)*sin(phi);
    //auto y = cos(theta);
    auto z = 1 - r1;


    return Vector(x, y, z);


}

class Ray 
{ 
public:
	Point o;
	Vector d; // origen y direcccion del rayo
	Ray(Point o_, Vector d_) : o(o_), d(d_) {} // constructor
};

//Agregue esta clase por que me facilito muchas cosas, primero quite el color en la clase esfera por que era redundante//
///Despues use Poliformismo para que diferenciar entre si es la esfera es un emisor de luz o si es un abedo //
//Si es un emisor de luz regresa directamente el valor del este emisor, si no regresa falso y en la funcion shade vuelve a lanzar un rayo
// Si esta segunda vez toca la luz regresa la aproximacion que hacemos con Montecarlo.
class material {
    public:
        virtual Color Emite(const Point& p) const {
            return Color(0,0,0);
        }

        virtual bool Rebota(
            const Ray& r_in, Color& atenuacion) const = 0;
};

class Luz : public material {
    public:

        Luz(Color c) : emit(c) {}

        virtual bool Rebota(const Ray& r_in, Color& atenuacion) const override {
            return false;
        }

        virtual Color Emite(const Point& p) const override {
            return emit;
        }

    public:
        Color emit;
};

class Abedo : public material {
    public:
        Abedo(const Color& a) : albedo(a) {}

        virtual bool Rebota(
            const Ray& r_in, Color& atenuacion) const override {
            atenuacion = albedo*(1.0/pi);
            return true;
        }
    public:
        Color albedo;
};

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
		if (discriminant<0) {
			return 0.0;
		}
		else{			
			double t0 = -b + sqrt(discriminant);
			double t1 = -b - sqrt(discriminant);
			double t = (t0 < t1) ? t0 : t1;
			if (t < 0) return 0;
			else return t;
		}
	}
};

Luz  Esferaluminoza(Color(10, 10, 10));
Abedo ParIzq(Color(.75, .25, .25));
Abedo ParDer(Color(.25, .25, .75));
Abedo ParedAt(Color(.25, .75, .25));
Abedo Suelo(Color(.25, .75, .75));
Abedo Techo(Color(.75, .75, .25));
Abedo EsAbaIz(Color(.2, .3, .4));
Abedo EsAbaDer(Color(.4, .3, .2));


Sphere spheres[] = {
	//Escena: radio, posicion ,material    //Fara facilitarme las cosas quite el Color, y se lo agregue en el material.
        Sphere(1e5,  Point(-1e5 - 49, 0, 0),     &ParIzq), // pared izq
        Sphere(1e5,  Point(1e5 + 49, 0, 0),      &ParDer), // pared der
        Sphere(1e5,  Point(0, 0, -1e5 - 81.6),   &ParedAt), // pared detras
        Sphere(1e5,  Point(0, -1e5 - 40.8, 0),   &Suelo), // suelo
        Sphere(1e5,  Point(0, 1e5 + 40.8, 0),    &Techo), // techo
        Sphere(16.5, Point(-23, -24.3, -34.6),   &EsAbaIz), // esfera abajo-izq
        Sphere(16.5, Point(23, -24.3, -3.6),     &EsAbaDer), // esfera abajo-der// Para observar las dos fuentes luminosas hay que comentar esta linea
		//Sphere(16.5, Point(23, -24.3, -3.6),     &Esferaluminoza), // esfera abajo-der // Para observar las dos fuentes luminosas hay que descomentar esta linea
        Sphere(10.5, Point(0, 24.3, 0),          &Esferaluminoza) // esfera arriba // esfera iluminada
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
		if ( t>aux[i] && aux[i]>0.001 ){
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

// Calcula el valor de color para el rayo dado
Color shade(const Ray &r,int prof) { //Agregamos la profundidad para hacer una funcion recursiva, esto nos permite lanzar un segundo rayo desde
	double t; 						 // donde intersecta nuestro primer rayo si es que intersecta, si consideramos la profunidad mayor que 2
	double h;						 // y si pensamos que en lugar de un segundo rayo es un rebote del rayo original, podriamos decir que con una
	int id = 0;						 // profundidad n podemos calcular como el rayo rebota n veces.
	// determinar que esfera (id) y a que distancia (t) el rayo intersecta

	if (prof <= 0) // Si ya se ha llegado 
        return Color();

	if (!intersect(r, t, id)){
		return Color();}	// el rayo no intersecto objeto, return Vector() == negro

	const Sphere &obj = spheres[id];

	// PROYECTO 2
	Point x=r.d*t+r.o; //Linea de codigo para el calculo de las coordenadas
	// determinar la dirección normal en el punto de interseccion
	Vector n=(x-obj.p).normalize();

	double radio=10.5;
	//Point re=random_esfera(ro); 
	Vector luz(0, 24.3,0);
    double arg,thetamax;

    Vector w=luz-x;
    double Distancia=(x-w).magnitud2();
    
    Point re=random_asolido(w,radio,arg,thetamax);

    Vector v;
    Vector u;
    ortonormalSystem(w,v,u);

    Vector dir=u*re.x+v*re.y+w*re.z;
    double cosenoluz=fabs(dir.dot(x)/(x.magnitud(),dir.magnitud()));
    
    Ray rebota(x,dir.normalize());
    Color attenuation;
    Color emite = obj.m->Emite(x);
    
	double Coseno=n.dot(dir);

    if (!obj.m->Rebota(r, attenuation)) 
        return emite;  
   
	//double pdf=Distancia/(4.0*pi*pi*(1-arg)*radio*cosenoluz);
    double pdf=Distancia/(4.0*pi*pi*(1-cos(thetamax))*radio*cosenoluz);//
    //double pdf=Distancia;
    //printf("%f \t",pdf);

    return attenuation.mult(shade(rebota, prof-1))*(Coseno/pdf); 
}


int main(int argc, char *argv[]) {
	double time_spent = 0.0;
	int muestras=32;
	int prof=2;
    clock_t begin = clock();
	//sleep(3);
 
	int w = 1024, h = 768; // image resolution
  
	// fija la posicion de la camara y la dirección en que mira
	Ray camera( Point(0, 11.2, 214), Vector(0, -0.042612, -1).normalize() );

	// parametros de la camara
	Vector cx = Vector( w * 0.5095 / h, 0., 0.); 
	Vector cy = (cx % camera.d).normalize() * 0.5095;
  
	// auxiliar para valor de pixel y matriz para almacenar la imagen
	Color *pixelColors = new Color[w * h];

	omp_set_num_threads(NUM_THREADS); //Lineas de Codigo para paralelizar
	#pragma omp parallel for //schedule(dynamic,1) //Con la paralelizacion se reduce en un 60 % aproxiamadamente el tiempo de ejecucion.

	for(int y = 0; y < h; y++) 
	{ 
		// recorre todos los pixeles de la imagen
		fprintf(stderr,"\r%5.2f%%",100.*y/(h-1));
		for(int x = 0; x < w; x++ ) {

			int idx = (h - y - 1) * w + x; // index en 1D para una imagen 2D x,y son invertidos

			Color pixelValue = Color(); // pixelValue en negro por ahora
			// para el pixel actual, computar la dirección que un rayo debe tener
			for (int i=0; i<muestras;i++)
			{
                
			Vector cameraRayDir = cx * ( double(x)/w - .5) + cy * ( double(y)/h - .5) + camera.d;
			// computar el color del pixel para el punto que intersectó el rayo desde la camara

			pixelValue = pixelValue + shade( Ray(camera.o, cameraRayDir.normalize()) ,prof);
			// limitar los tres valores de color del pixel a [0,1] 
			}
			//pixelValue = pixelValue;

			pixelColors[idx] = Color(clamp(pixelValue.x), clamp(pixelValue.y), clamp(pixelValue.z));
		}
		
	
	
	}


	fprintf(stderr,"\n");
	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("The elapsed time is %f seconds", time_spent);

	// PROYECTO 1
	// Investigar formato ppm
	FILE *f = fopen("MuestreoAnguloSolido.ppm", "w");
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
