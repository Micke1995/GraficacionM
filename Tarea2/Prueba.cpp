#include <math.h>
#include <stdlib.h>
#include <stdio.h>  
#include <cstdlib>
#include <time.h>


double pi=3.14159265358979323846;

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
	//Vector operator*(const Vector &b) const { return Vector(x * b.x , y * b.y , z * b.z); }

	// operator % para producto cruz
	Vector operator%(Vector&b){return Vector(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);}
	
	// producto punto con vector b
	double dot(const Vector &b) const { return x * b.x + y * b.y + z * b.z; }

	// producto elemento a elemento (Hadamard product)
	Vector mult(const Vector &b) const { return Vector(x * b.x, y * b.y, z * b.z); }
	
	// normalizar vector 
	Vector& normalize(){ return *this = *this * (1.0 / sqrt(x * x + y * y + z * z)); }

};


typedef Vector Point;
typedef Vector Color;

inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline Vector random_esfera() {
    auto r1 = random_double();
    auto r2 = random_double();

    auto x = cos(2*pi*r1)*2*sqrt(r2*(1-r2));
    auto y = sin(2*pi*r1)*2*sqrt(r2*(1-r2));
    auto z = 1 - 2*r2;

    return Vector(x, y, z);
}


inline Vector random_hemisferio() {
    auto r1 = random_double();
    auto r2 = random_double();

    auto theta=acos(1-r1);
    auto phi=2*pi*r2;

    auto x = sin(theta)*cos(phi);
    auto y = sin(theta)*sin(phi);
    auto z = 1 - r1;

    return Vector(x, y, z);
}

inline Vector random_coseno() {
    auto r1 = random_double();
    auto r2 = random_double();

    auto phi = 2*pi*r1;

    auto z = sqrt(1-r2);
    auto x = cos(phi)*sqrt(r2);
    auto y = sin(phi)*sqrt(r2);


    return Vector(x, y, z);
}



int main(){
    srand(time(NULL));
    //printf("hola mundo");
    Vector Direccion;
    for (int i=0;i<100;i++){
    Direccion=random_hemisferio();
    printf("%f,%f,%f,\n",Direccion.x,Direccion.y,Direccion.z);
    };

};

/*int main() {
    for (int i = 0; i < 100; i++) {
        auto r1 = random_double();
        auto r2 = random_double();
        auto x = cos(2*pi*r1)*2*sqrt(r2*(1-r2));
        auto y = sin(2*pi*r1)*2*sqrt(r2*(1-r2));
        auto z = 1 - 2*r2;
        printf("%f,%f,%f,\n",x,y,z);
    }
}*/