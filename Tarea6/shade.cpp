// Calcula el valor de color para el rayo dado
/*Color shade(const Ray &r,int prof) { //Agregamos la profundidad para hacer una funcion recursiva, esto nos permite lanzar un segundo rayo desde
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
	
	double pdf=1.0;

	attenuation=obj.m->BDRF(r,rebota,rec);

	
	
	double Coseno=rec.n.dot(rebota.d);
	
	return   attenuation*shade(rebota, prof-1)*(Coseno/pdf);
}*/
/*Color shade(const Ray &r,int prof) { //Agregamos la profundidad para hacer una funcion recursiva, esto nos permite lanzar un segundo rayo desde
	
	double t; 						 					 
	int id = 0;	
	Color attenuation;
	Ray rebota=r;
	double pdf;
	registro rec;
	Color emite;

	Color troughpout(1.0,1.0,1.0);
	if (!intersect(r, t, id)){
		return Color();
		}

	do{			 
	prof=prof-1;
	const Sphere &obj = spheres[id];
	rec.x=rebota.d*t+rebota.o; 
	rec.n=(rec.x-obj.p).normalize();
	rec.t=t;
	
    emite = obj.m->Emite(rec.x);	
    if (!obj.m->Rebota(r, rebota,rec)) 
        break;  
	attenuation=obj.m->BDRF(r,rebota,rec);	
	double Coseno=fabs(rec.n.dot(rebota.d));
	troughpout=troughpout*attenuation*Coseno;
	
	if (!intersect(rebota, t, id)){
		break;
		}

	}while(prof>=0) ;
	
	return  emite*troughpout;
}*////////////Explicito//////////////
/*Color shade(const Ray &r,int uz) {
	double t; 						 
    int id = 0;      			    


	//if (prof <= 0) // Si ya se ha llegado 
    //    return Color();

	if (!intersect(r, t, id)){
		return Color();}	// el rayo no intersecto objeto, return Vector() == negro

	const Sphere &obj = spheres[id];
	registro rec;
	rec.x=r.d*t+r.o; //Linea de codigo para el calculo de las coordenadas
	
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
    
	double Coseno=fabs(rec.n.dot(dir));
    Ray rebota(rec.x,dir.normalize());
    Color attenuation ;
    Color emite = obj.m->Emite(rec.x);

	if (Coseno  < 0)
        return emite;

    if (!obj.m->Rebota(r, rebota,rec) && uz==1 ) //Si el objeto es un emisor regresa el emisor, pero solo la primera vez
        return emite;  

	double pw=1.0 / (2.0 * pi * (1.0 - costmax));

	double t2; 						 						
	int id2 = 0;	
	if (!intersect(rebota, t2, id2)){
		return Color();
		}
	const Sphere &obj2 = spheres[id2];	
	Point x2=rebota.d*t2+rebota.o; 
	Color emite2 = obj2.m->Emite(x2);
	Color LD; 
    if (!obj2.m->Rebota(r, rebota,rec)){ //
			LD = attenuation.mult(emite2)*(fabs(Coseno)/pw);
		} 

	double theta;
	Point re2=random_coseno(theta);
	Vector s2; 
	Vector ti2;
	coordinateSystem(rec.n,s2,ti2);

	Point dir2(re2.dot(Point(s2.x,ti2.x,rec.n.x)),re2.dot(Point(s2.y,ti2.y,rec.n.y)),re2.dot(Point(s2.z,ti2.z,rec.n.z)));
    Ray rebota2(rec.x,dir2.normalize());
	double pdf2=(1.0/pi)*cos(theta);
	double Coseno2 = rec.n.dot(dir2);
	double q = 0.3;
	double continueprob = 1.0 - q;
	if (random_double() < q) 
		return Color();
	
	// calculo de iluminacion indirecta
	Color Lind = attenuation.mult(shade(rebota2, 0))* (Coseno2/(pdf2*continueprob)) ;

	return emite+LD+Lind;


}*/ //Explicito

// Color shade(const Ray &r) { /// path traicing interativo ruleta rusa
	
// 	double t; 						 						 
// 	int id = 0;	
// 	Color attenuation;
// 	Ray rebota=r;
// 	registro rec;
// 	Color emite;
// 	double q=0.2;
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

//     if (!obj.m->Rebota(r, rebota,rec)) {
// 		break; 
// 	}
// 		attenuation=obj.m->BDRF(r,rebota,rec);	
// 		Coseno=fabs(rec.n.dot(rebota.d));	
// 		pdf=obj.m->PDF(rebota);
// 		troughpout=troughpout*attenuation*(Coseno/(continueprob*pdf));//
// 	if (random_double()<q) 
// 		break;
	
// 	if (!intersect(rebota, t, id)){
// 		break;
// 		}

// 	}while(true) ;
	
// 	return  emite*troughpout;
// }


/*double pi=3.14159265358979323846; //Creamos el valor de pi para facilitarnos varais cosas.

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

inline Vector random_coseno(double &theta) {//Esta funcion crea direcciones aleatorias con distribucion coseno hemisferico.
    double r1 = random_double();
    double r2 = random_double();

    double phi = 2.0*pi*r2;

    double z = sqrt(1.0-r1);
	theta = acos(z);

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

		virtual double PDF(const Ray &wi) const = 0;

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

			return emit;
		}

		virtual double PDF(const Ray &wi) const override {
			return 1.0;
			}




    public:
        Color emit;
};

class Abedo : public material {
    public:
        Abedo(const Color &a,const double &b) : albedo(a),sigma(b*b) {}

		virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override {//, Color &atenuacion,double &pdf,registro &rec
			//wo.o = rec.x;
			//wo.d = random_coseno(theta); 
			
            return true;
        }
		virtual Color BDRF(const Ray& wi,Ray& wo,const registro& rec) override{
			
			wo.o=rec.x;
			wo.d=random_coseno(theta);
			double pdf=(1.0/pi)*cos(theta);

			double ro,phio,thetao;
			CarteEsfericas(wo.d,thetao,phio,ro);

			
			

			double ri,thetai,phii;
			CarteEsfericas(wi.d,thetai,phii,ri);

			

			double A=1.0-sigma/(2.0*(sigma+0.33));
			double cosio = cos((phii-phio));
			double alpha = ( thetai > thetao) ? thetai : thetao;
			double beta = ( thetai < thetao) ? thetai : thetao;
			
			double B=0;
			if (cosio=!0){
				B=(0.45*sigma)/(sigma+0.09)*cosio*sin(alpha)*tan(beta);	
			}
			
			wo.d=LocalGlobal(rec.n,wo.d);
			
			return albedo*(1.0/pi)*(A+B)*(1.0/pdf);//*(1.0/pdf)*(1.0/pdf)
			//return albedo*(1.0/pi)*(1.0/pdf);//

		}
		virtual double PDF(const Ray &wo) const override{
			//return (1.0/pi)*cos(theta);
			return 1.0;
		}




    public:
        Color albedo;
		double theta;	
		double sigma;	


};

inline double Fresnel(const double& etap,const double& kapap,const double& coste){

	double sente = sin(acos(coste));
	double nksin = etap - kapap - sente * sente;		
	
	double ab = sqrt(nksin*nksin + etap*kapap*4.0);
	double a = sqrt((ab+nksin)/2.0);

	double rpernum=ab+coste*coste-2.0*a*coste;
	double rperdem=ab+coste*coste+2.0*a*coste;
	double rper=rpernum/rperdem;

	double rparnum=ab*coste*coste + sente * sente * sente * sente - 2.0*a*coste *sente * sente;
	double rpardem=ab*coste*coste + sente * sente * sente * sente + 2.0*a*coste *sente * sente;
	double rpar=rper*rparnum/rpardem;

	return (1.0/2.0)*(rper+rpar);
};

Vector refleccion(const Vector& v, const Vector& n) {
    return  ((n*(v.dot(n)*2.0))-v).normalize();
}

class MicroFasetC: public material {
    public:
        MicroFasetC(const Color& a,const Color& b,const double &c) : eta(a),kappa(b),alpha(c) {}

        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{
			wo.o = rec.x;
			Vector v =wi.d*(-1.0);
			wo.d = refleccion(v,rec.n);
            return (wo.d.dot(rec.n) > 0);////}
        }
		virtual Color BDRF(const Ray &wi, Ray &wo,const registro &rec) override{

			Color etap = eta*eta*(1.0/(1.00029*1.00029));//*(1.0/(1.00029*1.00029))
			Color kapap = kappa*kappa*(1.0/(1.00029*1.00029));//
			
			Vector v =wi.d*(-1.0);	
			Vector direccion=random_media(alpha);
			Vector wh=LocalGlobal(rec.n,direccion);
			wh.normalize();
			wo.d=refleccion(v,wh);


			double costh = wh.dot(rec.n);
			double beckam;
			beckam=D(costh);

			double smith=1.0;

			double cosv1=wi.d.dot(rec.n)*(-1.0);
			double cosv2=wo.d.dot(rec.n);

			if((wi.d.dot(wh)/cosv1)>0.0 && (wo.d.dot(wh)/cosv2)>0.0){
				smith=GSmith(cosv1)*GSmith(cosv2);
				
			}
			double cost =wh.dot(wo.d);
			double R=Fresnel(etap.x,kapap.x,cost);
			double G=Fresnel(etap.y,kapap.y,cost);
			double B=Fresnel(etap.z,kapap.z,cost);

			double divisor=1.0/(4.0*fabs(cosv1)*fabs(cosv2));
			double pdf2=(4.0*fabs(cost))/(costh*beckam);//*beckam

			
			return Color(R,G,B)*pdf2*divisor*beckam*smith;//*divisor*pdf2*smith;//*beckam;

		}

		virtual double PDF(const Ray &wi)const override {
			return 1.0;

		}

		double  D(const double &costh){
				if (costh>0){
				double alpha2=alpha*alpha;
				double costh2 = costh*costh;
				double tang=sqrt(1.0-costh2)/costh;
				return exp(-((tang*tang)/alpha2))/(pi*alpha2*costh2*costh2);
				}else return 0.0;
		}
		double GSmith(const double &cosv){
				
				double a=(cosv/(sqrt(1.0-cosv*cosv)))*alpha;
				if (a<1.6){
					return (3.535*a+2.181*a*a)/(1+2.276*a+2.577*a*a) ;
					}else return 1.0;
		
}

    private:
        Color eta;
		Color kappa;
		double alpha;
};



class Conductor: public material {
    public:
        Conductor(const Color& a,const Color& b) : eta(a),kappa(b) {}

        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{
			wo.o = rec.x;
			Vector v =(wi.d*(-1.0)).normalize();
			wo.d = refleccion(v,rec.n);

            return (wo.d.dot(rec.n) > 0);////}
        }
		virtual Color BDRF(const Ray& wi, Ray& wo,const registro& rec) override{
			double cost = wo.d.dot(rec.n);


			Color etap = eta*eta;//*(1.0/(1.00029*1.00029));//*(1.0/(1.00029*1.00029))
			Color kapap = kappa*kappa;//*(1.0/(1.00029*1.00029));//


			double R=Fresnel(etap.x,kapap.x,cost);
			double G=Fresnel(etap.y,kapap.y,cost);
			double B=Fresnel(etap.z,kapap.z,cost);
			
			return Color(R,G,B)*(1.0/cost);

		}

		virtual double PDF(const Ray &wi)const override {
			return 1.0;

		}
    public:
        Color eta;
		Color kappa;
};



class Dielectrico: public material {
    public:
        Dielectrico(const double &a,const double &b) : eta(a),kappa(b) {}

        virtual bool Rebota(const Ray& wi,Ray& wo,const registro& rec) override{
			
			//printf("%f,%f,%f\n",wo.o.x,wo.o.y,wo.o.z);
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
	
};

*/







////funciones utilizadas/////

/*inline double Beckmann(double const &costh){
	//printf("%f\n",costh);
	if (costh>0){
	double alpha=0.5*0.5;
	double costh4=costh*costh*costh*costh;
	double tanh = tan(acos(costh));
	return exp(-(tanh*tanh)/alpha)/(pi*alpha*costh4);
	}else return 0.0;
};


inline double Gsmith(const Vector &wi,const Vector &wh,const Vector &wo,const Vector &n){
	double g1=0;
	double g2=0;
	double cosi=wi.dot(wh);
	double coso=wo.dot(wh);
	double alpha=0.5;

	double thetai=acos(cosi);
	double thetao=acos(coso);

	if ( ( cosi / wi.dot(n) ) > 0 ){
		double a1=1.0/(alpha*tan(thetai));
		if (a1>1.6)
			g1=(3.535*a1+2.181*a1*a1)/(1+2.276*a1+2.577*a1*a1);	
		else g1=1.0;
	} 

	if ( (coso / wo.dot(n)) >  0){
		double a2=1.0/(alpha*tan(thetao));
		if (a2>1.6)
			g2=(3.535*a2+2.181*a2*a2)/(1+2.276*a2+2.577*a2*a2);	
		else g2=1.0;
	} 

   return g1*g2;
};*/

