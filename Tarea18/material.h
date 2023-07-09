// rt: un lanzador de rayos minimalista
 // g++ -O3 -fopenmp rt.cpp -o rt
#include <math.h>
#include <stdlib.h>
#include <stdio.h>  
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm> 
#include <iterator>  
#include <iostream>
#include <unistd.h> 
#include <time.h>
using namespace std;

std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dis(0.0, 1.0);

#ifndef MATERIAL_H
#define MATERIAL_H

double pi=3.14159265358979323846; //Creamos el valor de pi para facilitarnos varais cosas.
double invpi=1.0/pi;

inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
	//return dis(gen);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}
inline double Clamp(double val, int low, int high) {
    if (val < low) return low;
    else if (val > high) return high;
    else return val;
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
inline Vector unitVector(const Vector &v){
	return  v * (1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z)); 

};
inline Vector LocalGlobal(const Vector &n,const Vector &x){
	Vector s,t;
	coordinateSystem(n,s,t);
	return s*x.x+t*x.y+n*x.z;
	 
}	
//inline Vector sqrtvec(const Vector &v){ return Vector(sqrt(v.x),sqrt(v.y),sqrt(v.z)); }

inline Vector GlobalLocal(const Vector &n,const Vector &x){
	Vector s,t;
	coordinateSystem(n,s,t);
	return Vector (s.dot(x),t.dot(x),n.dot(x));
	 
}

inline Vector random_esfera() { //Esta funcion crea direcciones aleatorias esfericas.
    auto r1 = random_double();
    auto r2 = random_double();

	double theta=acos(1.0-2.0*r1);
	double phi=2.0*pi*r2;

    auto x = cos(phi)*sin(theta);
    auto y = sin(phi)*sin(theta);
    auto z = 1.0 - 2.0*r1;

    return Vector(x, y, z);
}
inline void CarteEsfericas(const Vector &w,double &theta,double &phi, double &r){
	r=sqrt(w.x*w.x+w.y*w.y+w.z*w.z);
	theta=acos(w.z/r);
	phi=atan(w.y/w.x);

}

inline Vector random_hemisferio() {//Esta funcion crea direcciones aleatorias en un hemisferio.
    double r1 = random_double();
    double r2 = random_double();

    double theta=acos(r1);
    double phi=2.0*pi*r2;

    double x = sin(theta)*cos(phi);
    double y = sin(theta)*sin(phi);
    double z = r1;

    return Vector(x, y, z);
}
inline Vector random_asolido(const Point &w,const double &r,double &costheamax){
    double r1 = random_double();
    double r2 = random_double();
	//double r3= spheres[7].r;
    double arg=(r*(1.0/w.magnitud()));

    costheamax=sqrt(1.0 - (arg * arg));

	double z= (1.0-r1)+r1*costheamax;
    //double theta = acos(1.0 - r1 + (r1 * (costheamax)));
	double theta=acos(z);
    double phi=2.0*pi*r2;

	double x = sin(theta) * cos(phi);
    double y = sin(theta) * sin(phi);
    //double z = cos(theta);


    return Vector(x, y, z);
}

inline Vector random_coseno() {//double &theta
    double r1 = random_double();
    double r2 = random_double();

    double phi = 2.0*pi*r2;

    double z = sqrt(1.0-r1);
	double theta = acos(z);

    double x = cos(phi)*sin(theta);
    double y = sin(phi)*sin(theta);

    return Vector(x, y, z);
}
inline void random_parametroscoseno(double &theta,double &phi) {
    double r1 = random_double();
    double r2 = random_double();
    phi = 2.0*pi*r2;
	theta = acos(sqrt(1.0-r1));
}
inline Vector random_media(const double &alpha) {//Esta funcion crea direcciones aleatorias con distribucion coseno hemisferico.
    double r1 = random_double();
    double r2 = random_double();

    double phi = 2.0*pi*r2;
	double theta = atan(sqrt(-(alpha*alpha)*log10(1.0-r1)));

	double z = cos(theta);
    double x = cos(phi)*sin(theta);
    double y = sin(phi)*sin(theta);

    return Vector(x, y, z);
}


inline Vector esfCarte(double  &thetao,double &phi) {
    double x = cos(phi)*sin(thetao);
    double y = sin(phi)*sin(thetao);
	double z = cos(z);

    return Vector(x, y, z);
}


class Ray 
{ 
public:
	Point o;
	Vector d; // origen y direcccion del rayo
	Ray(){}
	Ray(Point o_, Vector d_) : o(o_), d(d_) {} // constructor
};

struct registro{
	Point n;
	Point x;
	double t;
};


class material {
    public:
        virtual Color Emite(const Point& p) const {
            return Color(0,0,0);
        }
		virtual bool Rebota(const Ray &wi,Ray &wo,const registro &rec) = 0 ;

        virtual Color BDRF(const Ray &wi, Ray &wo,const registro &rec) = 0;

		virtual double PDF(const Ray &wi,const registro &rec)  = 0;

};

class Luz : public material {
    public:

        Luz(Color c) : emit(c) {}
        virtual Color Emite(const Point& p) const override {
            return emit;
        }

        virtual bool Rebota(const Ray &wi,Ray &wo,const registro &rec)override {
            return false;
        }
		virtual Color BDRF(const Ray &wi, Ray &wo,const registro &rec) override{
			return Color();//emit*(1.0/10.0)
		}

		virtual double PDF(const Ray &wi,const registro &rec)  override {
			return 1.0;
			}




    public:
        Color emit;
};

class Difuso : public material {
    public:
        Difuso(const Color &a) : albedo(a) {}

		virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override {//Tarea 5 Material Difuso 
			wo.o = rec.x;
			wo.d = random_coseno(); 
            return true;
        }
		virtual double PDF(const Ray &wo,const registro &rec)  override{
			return (invpi)*wo.d.z;	
		}
		virtual Color BDRF(const Ray& wi,Ray& wo,const registro& rec) override{
			wo.d=LocalGlobal(rec.n,wo.d);
		return albedo*(invpi);	

		}

    public:
        Color albedo;	

};

class OrenNayar : public material {
    public:
        OrenNayar(const Color &a,const double &b) : albedo(a),sigma(b*b) {}

		virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override {//Tarea 5 Material Lambertiano imperfecto 
			wo.o = rec.x;
			wo.d = random_coseno(); 
            return true;
        }
		virtual Color BDRF(const Ray& wi,Ray& wo,const registro& rec) override{
			double ro,phio,thetao;
			CarteEsfericas(wo.d,thetao,phio,ro);

			double ri,thetai,phii;
			CarteEsfericas(wi.d,thetai,phii,ri);

			double A=1.0-sigma/(2.0*(sigma+0.33));
			double cosio = cos(phii-phio);
			double alpha = ( thetai > thetao) ? thetai : thetao;
			double beta = ( thetai < thetao) ? thetai : thetao;
			
			double B;
			if (cosio>0){ //Correccion Oren Nayar
				B=(0.45*sigma)/(sigma+0.09)*cosio*sin(alpha)*tan(beta);	
			}
			wo.d=LocalGlobal(rec.n,wo.d);
			return albedo*(invpi)*(A+B);
		}
		virtual double PDF(const Ray &wo,const registro &rec) override{
			return invpi*wo.d.z;
		}
    public:
        Color albedo;	
		double sigma;	

};

inline double Fresnel(const double& etap,const double& kapap,const double& cost){
	double coste=Clamp(cost,-1.0,1.0);

	double sente = sqrt(1.0-coste*coste);
	double nksin = etap - kapap - sente * sente;		

	double ab = sqrt(nksin*nksin + etap*kapap*4.0);
	double a = sqrt((ab+nksin)*0.5);

	double rpernum=ab+coste*coste-2.0*a*coste;
	double rperdem=ab+coste*coste+2.0*a*coste;
	double rper=rpernum/rperdem;

	double rparnum=ab*coste*coste + sente * sente * sente * sente - 2.0*a*coste *sente * sente;
	double rpardem=ab*coste*coste + sente * sente * sente * sente + 2.0*a*coste *sente * sente;
	double rpar=rper*(rparnum/rpardem);

	return (0.5)*(rper+rpar);
}

Vector refleccion(const Vector& v, const Vector& n) {
    return  ((n*(v.dot(n)*2.0))-v).normalize();
}
class MicroFasetC: public material {
    public:
        MicroFasetC(const Color& a,const Color& b,const double& c) : eta(a*a),kappa(b*b),alpha(c) {}

        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{
			wo.o = rec.x;
			Vector direccion=random_media(alpha);
			wh=LocalGlobal(rec.n,direccion);
			Vector v =(wi.d*(-1.0)).normalize(); 
			Vector aux=refleccion(v,rec.n);
			wo.d = refleccion(v,wh);
			wo.d.normalize();
            return (aux.dot(rec.n) > 0);// (wo.d.dot(rec.n) > 0);//true;//
        }
		virtual Color BDRF(const Ray &wi, Ray &wo,const registro &rec) override{  //Tarea 8 

			Vector v=wi.d*(-1.0);
			double cosv1=v.dot(rec.n);
			double cosv2=wo.d.dot(rec.n);

			double cosih=v.dot(wh);
			double cosh=wh.dot(rec.n);

			double smith=0.0;
			if((v.dot(wh)/cosv1)>0.f && (wo.d.dot(wh)/cosv2)>0.f){
				smith=GSmith(cosv1)*GSmith(cosv2);	
			}

			double R=Fresnel(eta.x,kappa.x,cosih);
			double G=Fresnel(eta.y,kappa.y,cosih);
			double B=Fresnel(eta.z,kappa.z,cosih);

			double denominador=1.0/(fabs(cosv1)*fabs(cosv2));
			

			return Color(R,G,B)*smith*denominador;//Color(1.0,1.0,1.0)*smith*denominador;// No considere Beckman por que se elimina en con la probabilidad, y considero que el
													// Resultado es satisfactorio.
		}

		virtual double PDF(const Ray &wo,const registro &rec) override {
			double cosh=wh.dot(rec.n);
			double pdf=cosh/(fabs(wo.d.dot(wh)));
			return pdf;

		}

		double  D(const double &costh){
				if (costh>0){
				double alpha2=alpha*alpha;
				double costh2 = costh*costh;
				double tang=sqrt(1.0-costh2)/costh;
				//printf("%f\n",tang);
				return exp(-((tang*tang)/alpha2))/(pi*alpha2*costh2*costh2);
				}else return 0.0;
		}
		double GSmith(const double &cosv){
				double a=(cosv/(sqrt(1.0-cosv*cosv)*alpha));
				
				if (a<1.6){
					return (3.535*a+2.181*a*a)/(1+2.276*a+2.577*a*a) ;
					}else return 1.0;
		
}

    private:
        Color eta;
		Color kappa;
		double alpha;
		Vector wh; //Direccion de la faceta
};








class Conductor: public material {
    public:
        Conductor(const Color& a,const Color& b) : eta(a),kappa(b) {}

        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{
			wo.o = rec.x;
			Vector v =(wi.d*(-1.0)).normalize();
			wo.d = refleccion(v,rec.n);
			//wo.d.normalize();
            return (wo.d.dot(rec.n) > 0);// true;//(wo.d.dot(rec.n) > 0);//true;//(wo.d.dot(rec.n) > 0);////}
        }
		virtual Color BDRF(const Ray& wi, Ray& wo,const registro& rec) override{
			double cost = wo.d.dot(rec.n);

			Color etap = eta*eta;
			Color kapap = kappa*kappa;

			double R=Fresnel(etap.x,kapap.x,cost);
			double G=Fresnel(etap.y,kapap.y,cost);
			double B=Fresnel(etap.z,kapap.z,cost);
			
			return Color(R,G,B)*(1.0/cost);
			// return Color(1.0,1.0,1.0);
		}

		virtual double PDF(const Ray &wo,const registro &rec) override {
			return 1.0;

		}
    public:
        Color eta;
		Color kappa;
};


class Dielectrico: public material {////// Tarea 18/////////////////
    public:
        Dielectrico(const double &a,const double &b) : etaT(a),etaI(b) {}
		
        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{	
			normal=rec.n.dot(wi.d*-1.0 ) > 0 ? rec.n: rec.n* -1.0;

			Vector v= (wi.d*-1.0);
			double cosi = v.dot(rec.n);
			
			bool entra = cosi > 0.f;
			if(!entra){
				indicerefrac = etaT/etaI;
				cosi = fabs(cosi);
			}else indicerefrac = etaI/etaT;			

			double sint = indicerefrac * sqrt( 1.0 - cosi * cosi);
			
			wo.o=rec.x; 	

			if (sint >= 1.0){
		 		return false;
			}
			cost = sqrt( 1.0 - sint * sint);
			double rpar,rper;


			// Fresnel= entra ? F(cost,cosi, etaT, etaI) : F(cost,cosi, etaI, etaT);
			Fresnel= entra ? F(cost,cosi, etaI, etaT) : F(cost,cosi, etaT, etaI);

			bool refraccion = random_double() < Fresnel;//dis(gen)< Fresnel;//false;//random_double() < Fresnel;//
			

			if (refraccion) {
				wo.d=refleccion(v,rec.n); //wi.d - normal*2.0*((normal.dot(wi.d)));// refleccion(v,normal); //refleccion(v,normal);  
			 	//wo.d.normalize();
			 	return (wo.d.dot(rec.n) > 0);//true;

			}else{
					
				wo.d = (wi.d * indicerefrac) + normal*(indicerefrac * cosi - cost);//Formulacion PBR si funciona adecuadamente deacuerdo a las consideracciones que  tengo
				wo.d.normalize();
				// wo.d= entra ? Vector(indicerefrac*wi.d.x,indicerefrac*wi.d.y,cosTt):Vector(indicerefrac*wi.d.x,indicerefrac*wi.d.y,-cosTt); // Formulacion sugerencia del profesor no funciona adecuadamente no se por que	
				// wo.d= refraction(wi.d,normal,indicerefrac);// Formulacion Ray Tracing in one weekend no funciona adecuadamente tampoco se por que 
            	return true;
			 }
        }
		virtual Color BDRF(const Ray& wi, Ray& wo,const registro& rec) override{

			if (!refraccion) {
				double cosr=(wi.d*-1.0).dot(rec.n);
				
				return Color( Fresnel,Fresnel,Fresnel)*(1.0/fabs(cosr));//*(1.0/fabs(cosr));//*(1/fabs(cosr));

			}else {
				return Color( 1.0-Fresnel,1.0-Fresnel,1.0-Fresnel)*(1.0/(indicerefrac*indicerefrac))*(1.0/fabs(cost));//*(indicerefrac*indicerefrac);///fabs(cost)//*(1.0/costt)
			}

			}


		virtual double PDF(const Ray &wi,const registro &rec) override {			
			if (!refraccion) {
				return Fresnel;//1.0;//Fresnel;//
			}else
			{ 
				return 1.0-Fresnel;//1.0 - Fresnel;
			}
		}

		double F(const double &cosi,const double &cost,const double &eT,const double &eI){
			double rpar = ((eT*cosi) - (eI*cost))/((eT*cosi) + (eI*cost));
			double rper = ((eI*cosi) - (eT*cost))/((eI*cosi) + (eT*cost));
			return (rpar*rpar + rper*rper) * 0.5;
		}

    public:
		double etaI;
		double etaT;
		double Fresnel;
		double indicerefrac;
		bool refraccion;
		Vector normal;
		double cost;
		
};



#endif

// Vector refraction(const Vector& uv, const Vector& n, double etai_over_etat) {//funcion de refraction de ray tracing in one weekend no funciona adecuadamente
//     auto cos_theta = fmin(-uv.dot(n), 1.0);
//     Vector r_out_perp =  (uv + n*cos_theta)* etai_over_etat;
//     Vector r_out_parallel = n* -sqrt(fabs(1.0 - r_out_perp.magnitud2()))  ;
//     return r_out_perp + r_out_parallel;
// }
//Codigo Basura
// class MicroFasetC: public material {
//     public:
//         MicroFasetC(const Color& a,const Color& b,const double& c) : eta(a),kappa(b),alpha(c) {}

//         virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{
// 			wo.o = rec.x;
// 			Vector v =wi.d*(-1.0);	
// 			Vector direccion=random_media(alpha);
// 			wh=LocalGlobal(rec.n,direccion);
// 			wh.normalize();
// 			wo.d=refleccion(v,wh);
// 			wo.d.normalize();
//             return (wo.d.dot(wh) > 0);
//             //return true;// (wo.d.dot(rec.n) > 0);//true;//
//         }
// 		virtual Color BDRF(const Ray &wi, Ray &wo,const registro &rec) override{

// 			Color etap = eta*eta;
// 			Color kapap = kappa*kappa;

// 			double costh = wh.dot(rec.n);


// 			double smith;

// 			double cosv1=(wi.d*-1.0).dot(rec.n);
// 			double cosv2=wo.d.dot(rec.n);

// 			if(((wi.d*-1.0).dot(wh)/cosv1)>0.f && (wo.d.dot(wh)/cosv2)>0.f){
// 				smith=GSmith(cosv1)*GSmith(cosv2);	
// 			}
// 			double cost =wh.dot(wo.d);
			
// 			double R=Fresnel(etap.x,kapap.x,cost);
// 			double G=Fresnel(etap.y,kapap.y,cost);
// 			double B=Fresnel(etap.z,kapap.z,cost);

// 			double divisor=1.0/(fabs(cosv1)*fabs(cosv2));
// 			double pdf2=(fabs(cost))/(costh);		
			
// 			return Color(R,G,B)*pdf2*divisor*smith;

// 		}

// 		virtual double PDF(const Ray &wi,const registro &rec)override {
// 			return 1.0;

// 		}

// 		double  D(const double &costh){
// 				if (costh>0){
// 				double alpha2=alpha*alpha;
// 				double costh2 = costh*costh;
// 				double tang=sqrt(1.0-costh2)/costh;
// 				return exp(-((tang*tang)/alpha2))/(pi*alpha2*costh2*costh2);
// 				}else return 0.0;
// 		}
// 		double GSmith(const double &cosv){
// 				double a=((cosv/(sqrt(max(0.0,1.0-cosv*cosv)))))*alpha;
// 				//printf("%f\n",cosv);
// 				if (a<1.6){
// 					return (3.535*a+2.181*a*a)/(1+2.276*a+2.577*a*a) ;
// 					}else return 1.0;
		
// }

//     private:
//         Color eta;
// 		Color kappa;
// 		double alpha;
// 		Vector wh;
// };

/*class Dielectrico: public material {
    public:
        Dielectrico(const double &a,const double &b) : eta(a),kappa(b) {}

        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{
			wo.o = rec.x;	
			
			Vector v = wi.d;
			double cosi = v.dot(rec.n);		

			double indicerefrac = cosi > 0.0  ? eta : 1.0/eta;
			
			double cost=sqrt(1.0-indicerefrac*indicerefrac*(1.0-cosi*cosi));
			
			Fresnel=F(cosi,cost);
			
			if (random_double()<Fresnel){
				wo.d=refleccion(v*(-1.0),rec.n);
			}else{ 
				wo.d= cosi > 0.0  ? Vector(-indicerefrac*v.x,-indicerefrac*v.y,-cost) : Vector(-indicerefrac*v.x,-indicerefrac*v.y,cost);
				
			}
			wo.d.normalize();
		return true;

		}
		virtual Color BDRF(const Ray &wi, Ray &wo,const registro &rec) override{	
			printf("%f,%f,%f\n",rec.x.x,rec.x.y,rec.x.z);
			return Color(1.0,1.0,1.0);


		}
		virtual double PDF(const Ray &wi)const override {
			return 1.0;

		}
		double F(const double &cosi,const double &cost){
			
			double rpar=(eta*cosi-cost)/(eta*cosi+cost);
			rpar=rpar*rpar;
			double rper=(cosi-eta*cost)/(cosi+eta*cost);
			rper=rper*rper;
			return 0.5*(rpar+rper);
			
		}
    private:
        double eta;
		double kappa;
		double Fresnel;
	
};*/

/*Vector refract(const Vector& uv, const Vector& n, double etai_over_etat) {
    auto cos_theta = fmin(-uv.dot(n), 1.0);
    Vector r_out_perp =  (uv + n*cos_theta)* etai_over_etat;
    Vector r_out_parallel = n* -sqrt(fabs(1.0 - r_out_perp.magnitud2()))  ;
    return r_out_perp + r_out_parallel;
}

class Dielectrico: public material {
    public:
        Dielectrico(const double &a,const double &b) : etaT(a),etaI(b) {}

        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{

            return true;
        }
		virtual Color BDRF(const Ray& wi, Ray& wo,const registro& rec) override{
			
			wo.o=rec.x;
			Vector v= wi.d*(-1.0);
			Vector normal= rec.n.dot(v) > 0 ?  rec.n* -1.0: rec.n ;
			double cosi = (v).dot(rec.n);
			//cosi = Clamp(cosi, -1, 1);
			bool entra = cosi > 0.f;
			if(!entra){
				swap(etaI, etaT);
				cosi = fabs(cosi);
			}
			indicerefrac = etaI/etaT;

			double sint =indicerefrac * sqrt(max(0.0,1.0 - cosi * cosi));
			

			if (sint >= 1){
				wo.d = refleccion(wi.d,normal);
				return Color();
			}
			double cost = sqrt( 1.0 - sint * sint);
			Fresnel=F(cosi,cost);
			
			reflect = random_double() < Fresnel;

		
			if (reflect){
				Vector wr = refleccion(v,normal);
				wr.normalize();
				double pr = Fresnel;
				double fr = Fresnel / fabs(cosi);
				wo.d=wr;
				return Color(1.0,1.0,1.0)*(fr*(fabs(normal.dot(wr))/pr));
			}else {
				Vector wt = ((v*-1.0) * indicerefrac) + normal*(indicerefrac * cosi - cost);
				wt.normalize();
				double pt = 1.0 - Fresnel;
				double ft = (indicerefrac *indicerefrac*(1.0 - Fresnel)) / fabs(cost);
				wo.d=wt;
				return Color(1.0,1.0,1.0)*(ft*(fabs(normal.dot(wt))/pt));
			}
		}

		virtual double PDF(const Ray &wi,const registro &rec) override {
			
			return 1.0;

		}

		double F(const double &cosi,const double &cost){
			double rpar = ((etaT*cosi) - (etaI*cost))/((etaT*cosi) + (etaI*cost));
			double rper = ((etaI*cosi) - (etaT*cost))/((etaI*cosi) + (etaT*cost));
			return (rpar*rpar + rper*rper) * 0.5;
		}

    public:
		double etaI;
		double etaT;
		double Fresnel;
		double indicerefrac;
		bool reflect;
};*/

