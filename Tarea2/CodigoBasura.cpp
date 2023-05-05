#include <math.h>
#include <stdlib.h>
#include <stdio.h>  

struct hit_record {
    Point p;
    Vector normal;
    Color *mat_ptr;
    double t;
};

struct hit_record {
    Point p;
    Vector normal;
    Color *mat_ptr;
    double t;
};

class material {
    public:
        virtual Color emitted(
            const Ray& r_in, const hit_record& rec, double u, double v, const Point& p) const {
            return Color(0,0,0);
        }
};

class luz : public material {
    public:
        //diffuse_light(shared_ptr<texture> a) : emit(a) {}
        luz(Color c) : albedo(c) {}

        virtual Color emitted(
            const Ray& r_in, const hit_record& rec, double u, double v, const Point& p) const override {
            if (rec.t>1)
                return Color(0,0,0);
            return albedo;
        }

    public:
        Color albedo;
};

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
	Color m;	// Material de la sefera

	Sphere(double r_, Point p_, Color c_,Color m_): r(r_), p(p_), c(c_), m(m_) {}
  
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
			return (-b-sqrt(discriminant)/a);
		}
	}
};