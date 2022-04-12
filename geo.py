from math import sin, cos, sqrt, atan, atan2, degrees, radians,pi
import math
import numpy as np
from datetime import date
import math as m

def naglowek(t):
    autor="Gabriela Strzalkowska"
    index="312014"
    naglowek= autor+"#"+index+"#"+ str(date.today()) + '\n'
    return naglowek
 
     

class Transformacje:
    def __init__(self, model: str = "wgs84"):
        """
        Parametry elipsoid:
            a - duża półoś elipsoidy - promień równikowy
            b - mała półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
        """
        if model == "wgs84":
            self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.a = 6378137.0
            self.b = 6356752.31414036
        elif model == "mars":
            self.a = 3396900.0
            self.b = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.a - self.b) / self.a
        self.ecc = sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 
        self.ecc2 = (2 * self.flat - self.flat ** 2) # eccentricity**2

    def N(self,b):
        """Funkcja oblicza największy promien krzywizny"""
        Np=self.a / np.sqrt((1 - self.ecc2 * (np.sin(b)) ** 2))
        return Np

    def hirvonen(self,x, y, z):
        """ Funkcja przelicza XYZ na fi, lam, h """
        r = np.sqrt(x ** 2 + y ** 2)
        b = np.arctan(z / (r * (1 - self.ecc2)))
    
        b_po = b
        while True:
            b_przed = b_po
            n = geo.N(b_przed)
            h = (r / np.cos(b_przed)) - n
            b_po = np.arctan(z / (r * (1 - (self.ecc2 * (n / (n + h))))))
            if abs(b_po - b_przed) < (0.0000001 / 206265):
                break
    
        l = np.degrees(np.arctan(y / x))
        n = geo.N(b_po)
        h = (r / np.cos(b_po)) - n
        return np.degrees(b_po), l, h
     
    def xyz2plh(self, X, Y, Z, output = 'dec_degree'):
        """
        Algorytm Hirvonena - algorytm transformacji współrzędnych ortokartezjańskich (x, y, z)
        na współrzędne geodezyjne długość szerokość i wysokośc elipsoidalna (phi, lam, h). Jest to proces iteracyjny. 
        W wyniku 3-4-krotneej iteracji wyznaczenia wsp. phi można przeliczyć współrzędne z dokładnoscią ok 1 cm.     
       """
        r   = sqrt(X**2 + Y**2)           # promień
        lat_prev = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przybliilizenie
        lat = 0
        while abs(lat_prev - lat) > 0.000001/206265:    
            lat_prev = lat
            N = self.a / sqrt(1 - self.ecc2 * sin(lat_prev)**2)
            h = r / cos(lat_prev) - N
            lat = atan((Z/r) * (((1 - self.ecc2 * N/(N + h))**(-1))))
        lon = atan(Y/X)
        N = self.a / sqrt(1 - self.ecc2 * (sin(lat))**2);
        h = r / cos(lat) - N       
        if output == "dec_degree":
            return degrees(lat), degrees(lon), h 
        elif output == "dms":
            lat = self.deg2dms(degrees(lat))
            lon = self.deg2dms(degrees(lon))
            return f"{lat[0]:02d}:{lat[1]:02d}:{lat[2]:.2f}", f"{lon[0]:02d}:{lon[1]:02d}:{lon[2]:.2f}", f"{h:.3f}"
        else:
            raise NotImplementedError(f"{output} - output format not defined")
            
    def flh2xyz(self,fi,lam,h):
        """ Funkcja przelicza wspolrzedne geodezyjne na XYZ"""
        fi=radians(fi)
        lam=radians(lam)
        N = self.a/(1-self.ecc2*(math.sin(fi))**2)**(0.5)
        X= (N+ h) * math.cos(fi) * math.cos(lam)
        Y= (N+ h) * math.cos(fi) * math.sin(lam)
        Z= (N*(1-self.ecc2)+h) * math.sin(fi) 
        return X,Y,Z
      
    def neu(self,x, y, z, x0, y0, z0):
        """ Funkcja przelicza wspolrzedne na NEU """
        b0, l0, h0 = geo.hirvonen(x0, y0, z0)
        b0, l0 = np.radians(b0), np.radians(l0)
        R = np.array([
            [-np.sin(l0), -np.cos(l0) * np.sin(b0), np.cos(b0) * np.cos(l0)],
            [np.cos(l0), -np.sin(l0) * np.sin(b0), np.sin(l0) * np.cos(b0)],
            [0, np.cos(b0), np.sin(b0)]
        ])
        n, e, u = R.dot(np.array([x0 - x, y0 - y, z0 - z]).T)
        return n, e, u
    
    def u2000(self,fi, lam):
        """ Funkcja przelicza wspolrzedne geodezyjne do ukladu 2000 """
        m=0.999923
        fi=radians(fi)
        lam=radians(lam)
        N = self.a/math.sqrt(1-self.ecc2*math.sin(fi)**2)
        t = np.tan(fi)
        e_2 = self.ecc2/(1-self.ecc2)
        n2 = e_2 * (np.cos(fi))**2    
        lam = degrees(lam)
        if lam>13.5 and lam <16.5:
            s = 5
            lam0 = 15
        elif lam>16.5 and lam <19.5:
            s = 6
            lam0 = 18
        elif lam>19.5 and lam <22.5:
            s = 7
            lam0 = 21
        elif lam>22.5 and lam <25.5:
            s = 8
            lam0 = 24  

        lam = math.radians(lam)
        lam0 = math.radians(lam0)
        l = lam - lam0
        A0 = 1 - (self.ecc2/4) - ((3*(self.ecc2**2))/64) - ((5*(self.ecc2**3))/256)   
        A2 = (3/8) * (self.ecc2 + ((self.ecc2**2)/4) + ((15 * (self.ecc2**3))/128))
        A4 = (15/256) * (self.ecc2**2 + ((3*(self.ecc2**3))/4))
        A6 = (35 * (self.ecc2**3))/3072 
        sig = self.a * ((A0*fi) - (A2*np.sin(2*fi)) + (A4*np.sin(4*fi)) - (A6*np.sin(6*fi))) 
        x = sig + ((l**2)/2) * N *np.sin(fi) * np.cos(fi) * (1 + ((l**2)/12) * ((math.cos(fi))**2) * (5 - t**2 + 9*n2 + 4*(n2**2)) + ((l**4)/360) * ((math.cos(fi))**4) * (61 - (58*(t**2)) + (t**4) + (270*n2) - (330 * n2 *(t**2))))
        y = l * (N*math.cos(fi)) * (1 + ((((l**2)/6) * (math.cos(fi))**2) * (1-t**2+n2)) +  (((l**4)/(120)) * (math.cos(fi)**4)) * (5 - (18 * (t**2)) + (t**4) + (14*n2) - (58*n2*(t**2))))
        x00 = round(m * x, 3) 
        y00 = round(m * y + (s*1000000) + 500000, 3)
        return x00,y00
    
    def u1992(self, phi,lam):
        """ Funkcja przelicza wspolrzedne geodezyjne do ukladu 1992"""
        phi = radians(phi)
        lam = radians(lam)
        L0 = radians(19)
        e2_ = (self.a**2 - self.b**2)/(self.b**2)
        eta2 = e2_ * cos(phi)**2
        t = math.tan(phi)
        l = lam - L0
        N = self.a/(1-self.ecc2*(sin(phi))**2)**(0.5)
        A0 = 1 - (self.ecc2/4) - ((3*(self.ecc2**2))/64) - ((5*(self.ecc2**3))/256)   
        A2 = (3/8) * (self.ecc2 + ((self.ecc2**2)/4) + ((15 * (self.ecc2**3))/128))
        A4 = (15/256) * (self.ecc2**2 + ((3*(self.ecc2**3))/4))
        A6 = (35 * (self.ecc2**3))/3072 
        sig = self.a * ((A0*phi) - (A2*sin(2*phi)) + (A4*sin(4*phi)) - (A6*sin(6*phi)))
        x = sig + (l**2)/2 * N * sin(phi) * cos(phi) * ( 1+(((l**2)/12) * (cos(phi)**2) * (5 - t**2 + 9*eta2 + 4*(eta2**2)) ) + (((l**4)/360) * (cos(phi)**4) * (61 - 58*(t**2) + (t**4) + 270*eta2 - 330*(eta2)*(t**2)) ) )
        y = l * N * cos(phi) * ( 1+(((l**2)/6) * (cos(phi)**2) * (1 - t**2 + eta2) ) + (((l**4)/120) * (cos(phi)**4) * (5 - 18*(t**2) + (t**4) + 14*eta2 - 58*(eta2)*(t**2)) ) )
        m0 = 0.9993
        x92 = m0*x - 5300000
        y92 = m0*y + 500000
        return f"{x92:.3f}", f"{y92:.3f}"

    def odl2D(self,X1,X2,Y1,Y2):
        """ Funkcja liczy odleglosc 2D"""
        odl2=np.sqrt((X1-X2)**2+(Y1-Y2)**2)
        return odl2
        
        
    def odl3D(self,X1,X2,Y1,Y2,Z1,Z2) :
        """Funkja liczy odleglosc 3D """
        
        od = sqrt( (X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2 )
        return(od)
    
    def azimuth(self,xA, yA, xB, yB):
        """
        Funkcja liczy azymut miedzy punktami i, i+1
        """
        dX = xB - xA
        dY = yB - yA 
        # wyznaczenie azymutu:
        if dX > 0 and dY > 0:                 
            Az = atan(dY/dX)               

        if dX < 0 and dY > 0:             
            Az= atan(dY/dX)+  pi          

        if dX < 0 and dY < 0:             
            Az= atan(dY/dX) +  pi        

        if dX > 0 and dY < 0:               
            Az= atan(dY/dX)  + 2 *  pi   

        if dX == 0 and dY > 0:               
            Az= pi /2                    
            
        if dX < 0 and dY == 0:               
            Az= pi                       

        if dX == 0 and dY < 0:               
            Az=  pi +  pi /2

        if dX > 0 and dY == 0:               
            Az= 0                        

    
        return (Az*180/pi)
    
    def wsp_satelity(self,t_oc, t):
        ni = 3.986005 * 10**14
        c = 2.99792458 * 10**8
        omega_e = 7.2921151467 * 10**(-5)
        a = 26561438.367127000000000 #to jest a^(1/2) do kwadratu
        a0 = 7.494073361158e-05
        a1 = 6.821210263297e-13  
        t_oc =316800 
        a2 =  0.000000000000e+00
        t_oe = 3.168000000000e+05
        delta_n = 3.953021909808e-09
        M0 = 2.153840309923e+00
        e = 6.699276505969e-03
        omega_w = 3.529864178612e-01
        omega = -7.829254577985e-09
        Cus = 9.115785360336e-06
        Cuc = 2.292916178703e-06
        Crs = 4.456250000000e+01
        Crc = 2.142812500000e+02
        Cis = 1.676380634308e-08
        Cic = -8.009374141693e-08
        IDOT = -1.667926630144e-10
        i0 = 9.756773856565e-01
        omega_0 = -1.777723722106e+00
        IODE=2.070000000000e+02
        apot_pol=5.153779037476e+03
    
        delta_t = a0 + a1*(t - t_oc) + a2*(t - t_oc)**2
        tk = t - delta_t - t_oe
        a = (m.sqrt(a))**2
        n0 = m.sqrt(ni / a**3)
        n = n0 + delta_n
        Mk = M0 + n * tk
        Ek = Mk
        while True:
           Ek2 = Mk + e * m.sin(Ek)
           if abs(Ek2 - Ek) < 10**(-12):
               break
           Ek = Ek2
        vk = 2 * np.arctan(m.sqrt((1 + e) / (1 - e)) * m.tan(Ek / 2))
        u = omega_w + vk
        delta_uk = Cus * m.sin(2*u) + Cuc * m.cos(2 * u)
        delta_rk = Crs * m.sin(2*u) + Crc * m.cos(2 * u)
        delta_ik = Cis * m.sin(2*u) + Cic * m.cos(2 * u) + IDOT * tk
        uk = u + delta_uk
        rk = a * (1 - e * m.cos(Ek)) + delta_rk
        ik = i0 + delta_ik
        omega_k = omega_0 + (omega - omega_e) * tk - omega_e * t_oe
        x_poch = rk * m.cos(uk)
        y_poch = rk * m.sin(uk)
        x = x_poch * m.cos(omega_k) - y_poch * m.cos(ik) * m.sin(omega_k)
        y = x_poch * m.sin(omega_k) + y_poch * m.cos(ik) * m.cos(omega_k)
        z = y_poch * m.sin(ik)
        return(x,y,z)
    

    def ro(self,xr,yr,zr):     
        t_oc =316800 
        t = 317700#moje
        xs,ys,zs=geo.wsp_satelity(t_oc,t)
        c=299792458 #m/s
        omega_e = 7.2921151467 * 10**(-5) #rad/s
        ro_n = m.sqrt((xr-xs)**2 + (yr-ys)**2 + (zr-zs)**2)
        ro_n2=2*ro_n
        while np.abs(ro_n-ro_n2)>0.001:  
            ro_n=ro_n2
            tau_n=ro_n/c 
            tn_n = t  - tau_n #nasz czas: 16:15 - ro/c
            X_n,Y_n,Z_n=geo.wsp_satelity(t_oc,tn_n)
            alfa_n = omega_e * (tau_n)
            macierz_n = np.array([[m.cos(alfa_n), m.sin(alfa_n), 0],
                           [-m.sin(alfa_n), m.cos(alfa_n), 0],
                           [0, 0, 1]]) 
            wsp_n=np.array([[X_n],
                      [Y_n],
                      [Z_n]])
            wsp_po_transf_n=macierz_n @ wsp_n
            ro_n2= m.sqrt((xr-wsp_po_transf_n[0])**2 + (yr-wsp_po_transf_n[1])**2 + (zr-wsp_po_transf_n[2])**2)
        return ro_n2

        

    


def dzialania(tablica, uklady):
    X = tablica[:, 0]
    Y = tablica[:, 1]
    Z = tablica[:, 2]
    
    x0 =np.mean(X)
    y0 =np.mean(Y)
    z0 =np.mean(Z)
    
    N=[]
    E=[]
    U=[]
    for i in range(len(X)):
        x, y, z = X[i], Y[i], Z[i]
        n, e, u = geo.neu(x, y, z, x0, y0, z0)
        N.append(n)
        E.append(e)
        U.append(u)

    raport = " "

    naglowek = "X, Y, Z"
    if '1' in uklady:
        naglowek += ", fi, lam, h"
    if '2' in uklady:
        naglowek += ", (policzone) X, Y, Z"
    if '3' in uklady:
        naglowek += ", n, e, u"
    if '4' in uklady:
        naglowek += ", X2000, Y2000"
    if '5' in uklady:
        naglowek += ", X92, Y92"
    if '8' in uklady:
        naglowek += ", elewacja"
   # if '7' in uklady:
    #    naglowek += ", odl2D(i,i+1), odl3D(i,i+1)"

    raport += naglowek
    raport = raport + "\n"
         
    for i in range(len(X)):
        x, y, z = X[i], Y[i], Z[i]
        linia = "{} {} {}".format(x, y, z)
        if '1' in uklady:
            b, l, h = geo.hirvonen(x, y, z)
            linia += ' {} {} {}'.format(b, l, h)            
        if '2' in uklady:
            x1,y1,z1 = geo.flh2xyz(b,l, h)
            linia += ' {} {} {}'.format(x1, y1, z1)        
        if '3' in uklady:
            n, e, u = N[i], E[i], U[i]
            linia += ' {} {} {}'.format(n, e, u)      
        if '4' in uklady:
            b1, l1, h1 = geo.hirvonen(x, y, z)
            x00, y00 = geo.u2000(b1, l1)
            linia += ' {} {}'.format(x00, y00)     
        if '5' in uklady:
            b, l, h = geo.hirvonen(x, y, z)
            x92, y92 = geo.u1992(b, l)
            linia += ' {} {}'.format(x92,y92)
        if '8' in uklady:
            ro_n=geo.ro(x,y,z)
            linia += ' {}'.format(ro_n)
        
            
        raport += linia + '\n'  
#def az i elewacji, def odl2 i 3d NA TEJ SAMEJ ZASADZIE             
            
    
    azymuty='\n'
    odleglosc='\n'

    for i in range(0,len(X)-1):
        x1, y1, z1 = X[i], Y[i], Z[i]
        x2, y2, z2 = X[i+1], Y[i+1], Z[i+1]
        
        if '6' in uklady:
            Az1=geo.azimuth(x1,y1,x2,y2)
            azymuty+='azymut {}-{} {}\n'.format(i+1,i+2,Az1)
        
        
        if '7' in uklady:
            odl2 = geo.odl2D(x1,x2,y1,y2)
            odl3 = geo.odl3D(x1,x2,y1,y2,z1,z2)
            
            odleglosc+='\n odleglosc 2D {}-{} {}'.format(i+1, (i+2), odl2)+'     odleglosc 3D {}-{} {}'.format(i+1, (i+2), odl3)
            
          
    
    raport += '\n'  
    raport+=azymuty+'\n'
    raport +=odleglosc+'\n'
    
    return raport


geo = Transformacje(model = "wgs84")

plik = "wsp_inp.txt"
# odczyt z pliku: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.genfromtxt.html
tablica = np.genfromtxt(plik, delimiter=',', skip_header = 4)


if __name__ == "__main__":
    
    print('''Wybierz uklad: 1 - XYZ > fi, lam, h    2 - fi, lam, h > XYZ    3 - NEU    4 - u2000    5 - u1992    6 - Az i elewacja    7 - odl. 2D i 3D    8 - elewacja \n''')
    print('''UWAGA: jesli chcesz wybrac wszystkie, wpisz od razu 1234567, nie wpisuj oddzielnie!!\n''')
    wybrane_uklady = input("Podaj uklady: ")
    
    print("Wykonuje polecenie, prosze czekac")
    
    #dzialania i zapis do pliku
    nag = naglowek(tablica)
    raport_pun = dzialania(tablica, wybrane_uklady)
    raport = open('wsp_obl.txt', 'w')
    raport.write(nag + raport_pun)
    raport.close()
    
    print("Polecenie wykonało się")

















