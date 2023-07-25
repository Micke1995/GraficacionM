
#ifndef Volume_h
#define Volume_h

#include <stdio.h>
#include "material.h"



class Volume {
    double sigmaT;
    double sigmaA;
    double sigmaS;
    double pf;// function de fase
   
    
public:
    Volume(double sigmaa, double sigmas){
        sigmaA = sigmaa; // Coeficientes de dispersión y absorción
        sigmaS = sigmas;
        sigmaT = sigmaA + sigmaS;
        pf = 1.0 / (4*pi); // voulmen isotropico
    }
    
    double getSigmaS(){
        return sigmaS;
    }
    double getSigmaT(){
        return sigmaT;
    }
    double TrasmitanciaHomogenea(Vector x,Vector y){
        
        Vector dif = x - y;
        double dist = sqrt(dif.dot(dif));
        double transmitancia = exp(- dist * sigmaT);
        return transmitancia;
    }



    double TrasmitanciaHomogenea(double t){

        double transmitancia = exp( - t * sigmaT);  

        return transmitancia;
    }
    double funcionfase(){
        return pf;
    }
};

#endif 
