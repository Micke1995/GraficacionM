// Calcula el valor de color para el rayo dado
Color shade(const Ray &r, int bounce, int cond) {
	double t;
	int id = 0;

	// determinar que esfera (id) y a que distancia (t) el rayo intersecta
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() == negro
  
	const Sphere &obj = spheres[id];  //esfera sobre la que se intersecto

	if (obj.e.x > 0 && obj.e.y > 0 && obj.e.z > 0 && cond == 1)		// si la esfera intersectada es una fuente de luz y es el primer camino del rayo, regresar luz
		return obj.e;

	if (bounce > bounceDepth) return Color();

	// determinar coordenadas del punto de interseccion
	Point x = r.o + r.d*t;

	// determinar la dirección normal en el punto de interseccion
	Vector n = (x - obj.p).normalize();

	Vector n1;
	if (n.dot(r.d) < 0) {n1 = n;}			// determinar si un rayo entra o sale de una material dielectrico
	else {n1 = n * -1;}

	// color del objeto intersectado
	Color baseColor = obj.c;
	Color bsdf = BRDF(baseColor);
	
	// ruleta rusa
	double q = 0.1;
	double continueprob = 1.0 - q;
	if (dis(gen) < q) return Color();

	// determinar el color que se regresara

	// material especular 

	if (obj.mat == 1){
		Vector wr = r.d - n*2*(n1.dot(r.d));  // direccion de refleccion especular ideal
		return obj.e + baseColor.mult(shade(Ray(x, wr), bounce++, 1));
	}
	
	// material difuso

	if (obj.mat == 0) {

		// obtener una direccion aleatoria con muestreo de coseno hemisferico, wi
		double theta, phi, probMat;
		paramCosineHemisphere(theta, phi, probMat);
		Vector wi = sampleBSDF(n1, theta, phi, obj);
		double dotCos = n1.dot(wi);
		Ray newRay = Ray(x, wi);

		// calculo de iluminacion indirecta

		Color indirectLight = bsdf.mult(shade(newRay, bounce++, 0)) * (fabs(dotCos)/(probMat*continueprob));

		// para cada esfera en la escena, checar cuales son fuentes de luz

		Color directLight = Color();
		for (int i = 0; i < totalShperes; i++){
			
			const Sphere &temp = spheres[i];
			if (temp.e.x <= 0 && temp.e.y <= 0 && temp.e.z <= 0)
				continue;

			// si la esfera i es una fuente de luz, realizar muestreo de angulo solido, wl

			double theta1, phi1, cosTmax, probLight;
			paramSolidAngle(x, theta1, phi1, cosTmax, temp);
			Vector wc = (temp.p - x).normalize();
			Vector wl = sampleLight(wc, theta1, phi1, temp);
			if (intersect(Ray(x, wl), t, id) && id == i){	// si no hay oclusion, calcular iluminacion directa
				double dotCos1 = n1.dot(wl);
				probLight = probSolidAngle(cosTmax);
				directLight = directLight + bsdf.mult(temp.e * fabs(dotCos1) * (1.0/probLight));
			}
		}
		return obj.e * cond + directLight + indirectLight;
	}

	// material dielectrico
}