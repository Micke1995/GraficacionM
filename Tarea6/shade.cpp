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
	double h;						 
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
	double Coseno=rec.n.dot(rebota.d);
	troughpout=troughpout*attenuation*Coseno;
	
	if (!intersect(rebota, t, id)){
		break;
		}

	}while(prof>=0) ;
	
	return  emite*troughpout;
}*/









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

