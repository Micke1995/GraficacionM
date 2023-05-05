

Color shade(const Ray &r,int prof) { 
	double t; 						 						 
	int id = 0;						 
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

	Point dir(0, 24.3, 0);// Creamos la fuente puntual, en este caso es la direccion a la que vamos a muestrear, como este punto ya esta en coordenadas
							//Cartesianas y globales no es necesario crear el sistema de coordenadas ni realizar una transofrmacion.
	
	double dist=(dir-x).magnitud();//Para poder dar un color a la esfera el escalar dist y la nueva t de intersect tienen que ser muy cercanas.
    Ray rebota(dir.normalize(),x.normalize());// Invertimos la direccion del rayo siendo el origen la fuente puntual luminosa y evaluamos con intersect
    Color attenuation;
	
	double tn;
	int idn=0;
	double Coseno=n.x*(dir.x)+n.y*(dir.y)+n.z*(dir.z);//

	Color emite(10,10,10);

		if (intersect(rebota, tn,idn)){
		//printf("%f,%f,%d,%d \n",tn,dist,id,idn);
		obj.m->Rebota(r, attenuation);
			if (abs(tn-dist)<=40){
			//if (id==idn){
			return attenuation*emite*(1.0/sqrt(dist));}
			else return Color();}
		

    //return attenuation.mult(shade(rebota ,prof-1))*Coseno*emite*(1.0/dist); 

}



//////////////////////



Color shade(const Ray &r,int prof) {
	double t;
	int id = 0;

	double t1;
	int id1 = 0;

	// determinar que esfera (id) y a que distancia (t) el rayo intersecta
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() == negro
  
	const Sphere &obj = spheres[id];

	Point x = r.o + r.d*t;
	Vector n = (x - obj.p).normalize();

	// rayo que va desde la fuente puntual hacia el punto de interseccion
	Point luz(0, 24.3, 0);
	Ray lightRay = Ray (luz, x - luz);

	// direccion del rayo

	Vector lightDir = lightRay.d.normalize();

	// termino de radiancia emitida L_e de la ecuacion de iluminacion directa

	Color Le(4000,4000,4000);

	if (intersect(lightRay, t1, id1)) {
		Point x1 = lightRay.o + lightRay.d*t1;
		if (id == id1) {
			Color den = (x1 - luz);
			double denNorm = 1.0 / den.magnitud2();
			Le = Le*denNorm;
		}
		else Le = Color();
	}
	else Le = Color();

	// termino de BRDF f_r de la ecuacion de iluminacion directa para material difuso
	// simple
	Color degradado;
	obj.m->Rebota(r,degradado);

	// termino n_x.dot(omega_i) de la ecuacion de iluminacion directa

	double dotCos = n.dot(lightDir);

	// determinar el color que se regresara
	Color colorValue = Le.mult(degradado * (-dotCos));

	return colorValue;
}