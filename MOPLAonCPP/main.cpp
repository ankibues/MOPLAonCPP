#include <iostream>
#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>

#include <vector>
#include <random>
#include <exception>
#include"gauss_legendre.h"

using namespace std;
using namespace Eigen;

struct getC_RDP{
Tensor<double,4> C;

};
struct xyz2thetaphi_result {
    Tensor<double,1> t;
    Tensor<double,1> p;
};
struct Q2ang_result {
    Tensor<double,2> ang1;
    Tensor<double,2> ang2;
    Tensor<double,2> ang3;
};
struct Incl_result {
    Tensor<double,2> PHI;
    Tensor<double,2> R;
};
struct RandAANG_result {
    Tensor<double,2> a;
    Tensor<double,2> ang;
};

struct FourIdentityTensors {
    Tensor<double,4> JD;
    Tensor<double,4> JS;
    Tensor<double,4> JA;
    Tensor<double,4> JM;
};
// list of functions created !
Tensor<double,4> TGreen(Tensor<double,1> a, Tensor<double,4> Cm, int n);         // green tensor calculation for each ellipsoid
xyz2thetaphi_result xyz2thetaphi(Tensor<double,1> x, Tensor<double,1> y, Tensor<double,1> z, int n);  // convert x,y,z to theta, pi
Q2ang_result ConvertQ2Angs(Tensor<double,3> q, int n);    //  Convert tranformation matrix Q to Theta[0,2pi] and Phi[0,pi]
Tensor<double,2> orienMat_whole(Tensor<double,2> u, int n)  ; // function used in ConvertQ2Angs
Incl_result Incl(Tensor<double,2> a, int n);            //  get PHI, R of an ellipsoid
RandAANG_result RandAANG(double a1, int n);            //  Generate uniformly distributed random shapes of clasts as the input of the Mulit_rigid and Multi_deformable functions
Tensor<double,4> TransforM(Tensor<double,4> X, Tensor<double,2> Q);      // 4th-order tensor transformation between coordinate systems.
double sumofelements1(Tensor<double,1> a);          // sum of elements of 1D tensor
Tensor<double,2> Q(Tensor<double,1> ang);            //Transformation matrix Q from three spherical angles
Tensor<double,3> Qvec(Tensor<double,2> ang, int n);   // Vectorized version of Q function
Tensor<double,4> FourTensorInv(Tensor<double,4> X);  //Inverse of a fourth order tensor
Tensor<double,2> C2OneDarray(Tensor<double,4> C); // convert 4th-order C(i,j,k,l) to 1d array(1x81) row-major
Tensor<double,4> ConvertBK(Tensor<double,3> x,int h,Tensor<double,3> b); // convert function to use in inverse of 4th order tensor
FourIdentityTensors FourIdentity(void ); // generating four 4th order identity tensors
Tensor<double,2> RodrgRot(Tensor<double,2> a); // Rodrigues Rotation Approximation
inline double Inva(Tensor<double,2> x);   // The invariant of a 2nd order tensor
Tensor<double,4> contract(Tensor<double,4> X, Tensor<double,4> Y); // double-index contraction between two 4th-order tensors
Tensor<double,2> Multiply(Tensor<double,4> X, Tensor<double,2> y); // double-index contraction of a 4th-order tensor and a 2nd-order tensor.
inline double contract1(Tensor<double,2> x, Tensor<double,2> y); // double-index contraction between two 2nd-order tensors.
double sumofelements( Tensor<double,2> a); // sum of elements of 3*3 Tensor
Tensor<double,2> eye(int d);  // identity matrix generator
double normfunc(Tensor<double,4> eps ); // norm of 4th order tensor
Tensor<double,2> Wd(Tensor<double,2> a, Tensor<double,2> m, Tensor<double,2> d); // ellipsoid vorticity referred to the frame tracking its semi-axes
Tensor<double,1> C2OneDvec(Tensor<double,4> C);     // 4th order tensor to 1 D vector --ccp equivalent of reshape(a,[],1).....used in transform function
// constant values taken
const double PI = 2*acos(0.0);

inline Tensor<double,2> transp(Tensor<double,2> a)
{
    Tensor<double,2> b(3,3);
    b.setZero();
    for(int i=2;i>=0;i--)
        for(int j=2;j>=0;j--)
    {
        b(i,j)=a(j,i);
    }

    return b;
}


// sum of elements of 3*3 Tensor
inline double sumofelements( Tensor<double,2> a)
{
    double sum=0;
    for(int i=0;i<3;i++)
    {
        for (int j=0;j<3;j++)
        {
            sum=sum + a(i,j);
        }
    }

    return sum;
}

// sum of elements of 5*5 Tensor
inline double sumofelements5( Tensor<double,2> a)
{
    double sum=0;
    for(int i=0;i<5;i++)
    {
        for (int j=0;j<5;j++)
        {
            sum=sum + a(i,j);
        }
    }

    return sum;
}
/* Multiply double-index contraction of a 4th-order tensor and a 2nd-order tensor.
Input: X, a 4th-order tensor,   3*3*3*3 matrix;    y, a 2nd-order tensor,         3*3 matrix Output: m, a 2nd-order tensor, 3*3 matrix.
*/
Tensor<double,2> Multiply(Tensor<double,4> X, Tensor<double,2> y)
 {
    Tensor<double,2> m(3,3);
    m.setZero();
    Tensor<double,2> x(3,3);
    x.setZero();
    for (int ia=0;ia<3;ia++)
     {
        for(int ja=0;ja<3;ja++)
         {

      for(int k=0;k<3;k++)
    {
      for(int l=0;l<3;l++)
      {

          x(k,l) = X(ia,ja,k,l);
      }

         }

    m(ia,ja) = sumofelements((x*y));

        }
    }
    return m;
}


  // identity tensor generator
Tensor<double, 2> eye(int d)
{
    Tensor<double, 2> a(d,d);
    a.setZero();
        for(int i=0;i<d;i++)
    {
      for(int j=0;j<d;j++)
      {
          if(i==j)
            a(i,j)= 1;
          else
            a(i,j)=0;
      }
    }
     return a;
}

 //Normfunc  Norm of a 4th order tensor Input: X: a 4th-order tensor, 3*3*3*3 matrix; Output:a scalar,indicating the magnitude of the 4th order tensor.
double normfunc(Tensor<double,4> eps )
{

MatrixXd ep(3,3);
ep.setZero();
double m=0;
for(int ii=0;ii<3;ii++)
{
    for(int jj=0;jj<3;jj++)
    {
for(int k=0;k<3;k++)
    {
      for(int l=0;l<3;l++)
      {

          ep(k,l) = eps(ii,jj,k,l);
      }

         }
 m= m+ ep.norm();
    }
}
return m;
}


// contract1 double-index contraction between two 2nd-order tensors.

inline double contract1(Tensor<double,2> x, Tensor<double,2> y)
{
 double m= sumofelements(x*y);
    return m;

}

/*  contract function    double-index contraction between two 4th-order tensors.
Input:  X, Y, two 4th-order tensors      Output: m, a 4th-order tensor;  */

Tensor<double,4> contract(Tensor<double,4> X, Tensor<double,4> Y)
{

    Tensor<double,4> M(3,3,3,3);
    M.setZero();
    Tensor<double,2> x(3,3);
    x.setZero();
    Tensor<double,2> y(3,3);
    y.setZero();
    for(int i=0;i<3;i++)
       {
        for(int j=0;j<3;j++)
          {
            for(int k=0;k<3;k++)
               {
                  for (int l=0;l<3;l++)
                   {

                          for(int kk=0;kk<3;kk++)                     // CPP equivalent of matlab command  x = reshape(X(i,j,:,:),3,3);
                              {
                          for(int ll=0;ll<3;ll++)
                                  {

                                  x(kk,ll) = X(i,j,kk,ll);
                                   }
                              }

                                           for(int ia=0;ia<3;ia++)      // CPP equivalent of Matlab commmand y=Y(:,:,k,l)
                                           {
                                          for(int ja=0;ja<3;ja++)
                                             {
                                               y(ia,ja) = Y(ia,ja,k,l);
                                               }
                                             }


                    M(i,j,k,l) = sumofelements((x*y));
                   }
               }
          }
       }
return M;

}

// Rodrigues' Rotation Approximation  Input:  A, 3*3 matrix; Output: z, 3*3 matrix
   Tensor<double,2> RodrgRot(Tensor<double,2> a)
   {
    Tensor<double,2> z(3,3);
    z.setZero();
    MatrixXd b(3,3);
    b.setZero();
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
            b(i,j)= a(i,j);
    }

    double omega = b.norm()/pow(2,0.5);
    if (omega == 0)
        b = b.setIdentity();
    else
    {
        MatrixXd omega1(3,3);    omega1.setZero();
       omega1 = b/omega;
        b = b.setIdentity() + (omega1 * sin(omega)) + ((1 - cos(omega)) * omega1*omega1);
    }

    for(int ii=0;ii<3;ii++)
    {
        for(int jj=0;jj<3;jj++)
            z(ii,jj)= b(ii,jj);
    }
    return z;

   }

// Inva The invariant of a 2nd order tensor   Input: x: a 2nd-order tensor, 3*3 matrix;    Output: m: the invariant, a scalar
inline double Inva(Tensor<double,2> x)
{
   double k = pow((0.5* contract1(x,x)),0.5);
   return k;
}

// FourIdentity function 4th-order identity tensors Output: Jd, Js,Ja,Jm

FourIdentityTensors FourIdentity(void )
{
   MatrixXd delta(3,3); delta.setZero();
    delta.setIdentity();

// allocate J, Jm
   Tensor<double,4> J(3,3,3,3);
   Tensor<double,4> J1(3,3,3,3);
   Tensor<double,4> Jm(3,3,3,3);
    J.setZero();
    J1.setZero();
    Jm.setZero();          // Check if these 3 lines are really needed. Do we need to set these values 0?
// Eqn(3) in Jiang(2014)
   for (int i=0;i<3;i++)
      {
       for (int j=0;j<3;j++)
         {
            for (int k=0;k<3;k++)
            {
               for (int l=0;l<3;l++)
               {
                   J(i,j,k,l)  = delta(i,k)*delta(j,l);
                   J1(i,j,k,l) = delta(j,k)*delta(i,l);
                   Jm(i,j,k,l) = (delta(i,j)*delta(k,l))/3;
               }
            }
         }
     }
     Tensor<double,4> Js(3,3,3,3);
     Tensor<double,4> Ja(3,3,3,3);
     Tensor<double,4> Jd(3,3,3,3);
     Js.setZero();
     Ja.setZero();
     Jd.setZero();
   Js = 0.5*(J + J1);
   Ja = 0.5*(J - J1);
   Jd = Js - Jm;

   FourIdentityTensors r; // returning via structure
   r.JA= Ja;
   r.JD= Jd;
   r.JM= Jm;
   r.JS= Js;

   return r;
}


// convert function to use in getting inverse of 4th order tensor
Tensor<double,4> ConvertBK(Tensor<double,2> x,int h,Tensor<double,3> b)
{
    Tensor<double,4> M(3,3,3,3);
     M.setZero();
    Tensor<double,2> bb(h,h);
    bb.setZero();

    for (int i=0;i<3;i++)
        {
        for (int j=0;j<3;j++)
            {
            for(int k=0;k<3;k++)
            {
                for(int l=0;l<3;l++)
                {
                     for(int m=0;m<h;m++)
                    {
                         for(int n=0;n<h;n++)
                         {
                            bb(m,n)= b(i,j,m)*b(k,l,n);
                         }

                    }

                   M(i,j,k,l)=sumofelements5(x*bb);
                }
            }
          }
        }
        return M;

}


// Inverse of a 4th-order symmetric Tensor
Tensor<double,4> FourTensorInv(Tensor<double,4> X)
{
  double sqrt2 = pow(2,0.5);
  double sqrt3 = pow(3,0.5);
Tensor<double,3> b(3,3,6);
b.setZero();                                          // defining all the values of b
b(0,0,0) = -1/(sqrt2*sqrt3) ;
b(1,1,0) = - 1/(sqrt2*sqrt3) ;
b(2,2,0) =  2/(sqrt2*sqrt3) ;
b(0,0,1) = -1/sqrt2 ;
b(1,1,1) = 1/sqrt2 ;
b(1,2,2) = 1/sqrt2 ;
b(2,1,2) = 1/sqrt2 ;
b(0,2,3) = 1/sqrt2 ;
b(2,0,3) = 1/sqrt2 ;
b(0,1,4) = 1/sqrt2 ;
b(1,0,4) = 1/sqrt2 ;
b(0,0,5) = 1/sqrt3 ;
b(1,1,5) = 1/sqrt3 ;
b(2,2,5) = 1/sqrt3 ;

    int h = 5;
    MatrixXd M(5,5); M.setZero();
    MatrixXd P(5,5); P.setZero();
    Tensor<double,2> Mt(h,h);
      Mt.setZero();
    Tensor<double,2> B(3,3);
    Tensor<double,2> D(3,3);
    B.setZero(); D.setZero();
    Tensor<double,4> m(3,3,3,3);
    m.setZero();
    for(int lamda = 0;lamda<h;lamda++)
        {
        for(int xi = 0;xi<h; xi++)
        {

            for(int kk=0;kk<3;kk++)                     // CPP equivalent of matlab command b(;,;,xi)
                              {
                          for(int ll=0;ll<3;ll++)
                                  {

                              B(kk,ll)= b(kk,ll,xi);
                                   }
                              }
             for(int k=0;k<3;k++)                     // CPP equivalent of matlab command b(;,;,lambda)
                              {
                          for(int l=0;l<3;l++)
                                  {
                             D(k,l)= b(k,l,lamda);
                                   }
                              }

        M(lamda,xi)= contract1((Multiply(X,B)),D);
        }
        }
    // get the inverse of M
     P = M.inverse();
 for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            Mt(i,j)= P(i,j);

    m = ConvertBK(Mt,h,b);
    return m;
}



// Transformation matrix Q from three spherical angles  Input:  ang, three spherical angles defined in Jiang(2007a), in radian, 3*n matrix;  Output: q,   3*3*n matrix

Tensor<double,3> Qvec(Tensor<double,2> ang, int n)
    {
    Tensor<double,3> q(3,3,n);      q.setZero();
    for(int k=0;k<n;k++)
    {
     Vector3d a(3);           a.setZero();
     Vector3d b(3);           b.setZero();
     Vector3d c(3);           c.setZero();
     MatrixXd d(3,3);          d.setZero();

    a << sin(ang(1,k))*cos(ang(0,k)),
         sin(ang(1,k))*sin(ang(0,k)),
           cos(ang(1,k));

    if(ang(1,k) == PI/2)
        b << -sin(ang(0,k))*sin(ang(2,k)),
              cos(ang(0,k))*sin(ang(2,k)),
                  cos(ang(2,k));
    else
        b << cos(atan(tan(ang(1,k))*cos(ang(0,k)-ang(2,k))))*cos(ang(2,k)),
             cos(atan(tan(ang(1,k))*cos(ang(0,k)-ang(2,k))))*sin(ang(2,k)),
              -sin(atan(tan(ang(1,k))*cos(ang(0,k)-ang(2,k))));

    c = a.cross(b);
    d<< a,b,c;
    for(int i=0;i<3;i++)
    {
        for (int j=0;j<3;j++)
        {
            q(i,j,k)=d(j,i);
        }
    }

    }
    return q;
    }

// function wp = Wd(a,m,d) % the Ellipsoid vorticity referred to the frame tracking its semi-axes
 Tensor<double,2> Wd(Tensor<double,1> a, Tensor<double,2> m, Tensor<double,2> d)
 {
   Tensor<double,2> wp(3,3);
   wp.setZero();

  for(int i = 0;i<3;i++)
  {
      for(int j=0;j<3;j++)
      {
          if (i == j)
              wp(i,j) = 0;
          else
            if (a(i) == a(j))
              wp(i,j) = m(i,j);
          else
          {
              double r    = (pow(a(i),2) + pow(a(j),2))/pow(a(i),2)-pow(a(j),2);
              wp(i,j) = r * d(i,j);
              }

      }

  }
  return wp;
 }

double sumofelements1(Tensor<double,1> a) // sum of 1D array
{
    double sum=0;
    for(int i=0;i<81;i++)
    {
        sum=sum + a(i);
    }
    return sum;

}

// convert a 4th order tensor to 1D vector
Tensor<double,1> C2OneDvec(Tensor<double,4> C)
{
Tensor<double,1> Cld(81);
Cld.setZero();
int jj = 0;

for (int k=0;k<3;k++)
{
   for (int l =0; l<3; l++)
   {
    for (int i=0;i<3;i++)
    {
         for (int j=0;j<3;j++)
        {
             Cld(jj)=C(j,i,l,k);
             jj++;
        }
    }
   }
}
return Cld;
}

//4th-order tensor transformation between coordinate systems Input:  X, 3*3*3*3 matrix; Q, 3*3 matrix Output: ss, 3*3*3*3 matrix
Tensor<double,4> TransforM(Tensor<double,4> X, Tensor<double,2> Q)
{
    Tensor<double,4> ss(3,3,3,3);
    Tensor<double,4> qq(3,3,3,3);

    ss.setZero();
    qq.setZero();
    Tensor<double,1> q(81);      q.setZero();
    Tensor<double,1> x(81);       x.setZero();
    for(int i=0;i<3;i++)
        {
        for (int j=0;j<3;j++)
            {
            for (int k=0;k<3;k++)
                {
                for (int l=0;l<3;l++)
                  {
                    for (int m=0;m<3;m++)
                        {
                        for (int n=0;n<3;n++)
                           {
                            for (int s=0;s<3;s++)
                                {
                                for (int t=0;t<3;t++)
                                    {
                                    qq(m,n,s,t)= Q(i,m)*Q(j,n)*Q(k,s)*Q(l,t);
                                    }
                                }
                           }
                        }

          ss(i,j,k,l)=sumofelements1((C2OneDvec(qq))*(C2OneDvec(X)));

                }
             }
           }
        }
    return ss;
}

 /*RandAANG function
Generate uniformly distributed random shapes of clasts as the input of the Mulit_rigid and Multi_deformable functions
 Input:  a1,  the maximun semi-axis of clasts to generate random shapes;
         n,   number of clasts considered;
Output: a,   random shapes: a(1,:)from 1 to a1, a(2,:)from 1 to a(1,:),
        a(3,:)=1, 3*n matrix;
        ang, uniform distribution of clasts, 3*n matrix;*/

RandAANG_result RandAANG(double a1, int n)
{
Tensor<double,2> A(3,n);            A.setZero();
Tensor<double,2> ANG(3,n);          ANG.setZero();
  // random shapes
default_random_engine generator;
uniform_real_distribution<double> distribution (0.0,1.0);
for(int i=0;i<n;i++)
A(0,i)= 1+ (a1-1)* distribution(generator) ;

for(int j=0;j<n;j++)
A(1,j)= 1+ (A(0,j)-1)* distribution(generator) ;

for(int k=0;k<n;k++)
A(2,k)= 1;

// generate a uniform distribution
for(int l=0;l<n;l++)
ANG(0,l)= 2*PI* distribution(generator) ;

for(int m=0;m<n;m++)
ANG(1,m)= acos(-1  + 2 * distribution(generator) ) ;

for(int nn=0;nn<n;nn++)
ANG(2,nn)= (atan(cos(ANG(1,n))) * tan( PI* distribution(generator) )) + ANG(0,n) + 0.5*PI ;

RandAANG_result r;
r.a=A;
r.ang=ANG;

return r;
}
/* Incl(a)  get PHI, R of an ellipsoid           PHI measures of the triaxiality of an ellipsoid   R measures of ellipsoid deviation from sphere
 Input:  a are three semi-axes of n ellipsoids, 3*n matrix     Output: PHI, 1*n matrix, in rad         R,   1*n matrix */
Incl_result Incl(Tensor<double,2> a, int n)
{
  Tensor<double,2> A(3,n);
  A.setZero();
  A=a;

 double k;
 for(int i=0;i<3;i++)
 {
         if ((A(0,i)) < (A(1,i)))
        {
           k = A(0,i);
           A(0,i)= A(1,i);
           A(1,i)=k;

         }
         if ((A(0,i))<(A(2,i)))
         {
            k = A(0,i);
           A(0,i)=A(2,i);
           A(2,i)=k;
         }
         if ((A(1,i))<(A(2,i)))
         {
             k=A(1,i);
             A(1,i)=A(2,i);
             A(2,i)=k;
         }

     }
Tensor<double,2> x(1,n);                    x.setZero();
Tensor<double,2> y(1,n);                    y.setZero();
Tensor<double,2> r(1,n);                    r.setZero();
Tensor<double,2> phi(1,n);                  phi.setZero();

for(int p=0;p<n;p++)
    x(0,n)= log(A(1,p)/A(2,p));
for(int q=0;q<n;q++)
     y(0,n)= log(A(0,q)/A(1,q));

  for(int l=0;l<n;l++)
  {
     if(x(0,l)==0)
        phi(0,l)= PI/2;
     else
     {
    double k = y(0,l)/x(0,l);
  phi(0,l)  = atan(k);
  }
  }
 for(int k=0;k<n;k++)
  r(0,k) = sqrt(pow(y(0,k),2) + pow(x(0,k),2));

  Incl_result res;
  res.PHI = phi;
  res.R= r;

  return res;

}
// signum function defined
inline double sign(double n)
{
if (n < 0) return -1;
if (n > 0) return 1;
return 0;
}
Tensor<double,2> orienMat_whole(Tensor<double,2> u, int n)     // function used in Convert2Angs
{
    Tensor<double,2> angs(2,n);
    angs.setZero();
    for(int k=0;k<n;k++)
    {
        if (u(0,k)!= 0)
        angs(0,k)= atan(u(1,k)/u(0,k));

        if (u(0,k)== 0)
        angs(0,k) = 0.5*PI *sign(u(1,k));

        if (u(0,k)>=0 && u(1,k)<0)
        angs(0,k) = angs(0,k) + (2*PI);

        if (u(0,k)<0)
        angs(0,k) = angs(0,k) + PI;

        double normU = sqrt(pow(u(0,k),2) + pow(u(1,k),2) + pow(u(2,k),2));
        angs(1,k) = acos(u(2,k)/normU);
    }
    return angs;
}

/* ConvertQ2Angs  Convert tranformation matrix Q to Theta[0,2pi] and Phi[0,pi] (i.e., cover the whole range of 0<=Phi<=pi) of a1, a2, a3.
  Input:  q,    3*3*n matrix;                                       Output: ang1,   theta, phi of a1, in radian, 2*n matrix;
                                                                            ang2,   theta, phi of a2, in radian, 2*n matrix;
                                                                            ang3,   theta, phi of a3, in radian, 2*n matrix;*/
Q2ang_result ConvertQ2Angs(Tensor<double,3> q, int n)
{

    Tensor<double,2> a1(3,n);             a1.setZero();

    Tensor<double,2> a2(3,n);             a2.setZero();
    Tensor<double,2> a3(3,n);             a3.setZero();
    Tensor<double,2> Ang1(2,n);           Ang1.setZero();
    Tensor<double,2> Ang2(2,n);           Ang2.setZero();
    Tensor<double,2> Ang3(2,n);           Ang3.setZero();
    for(int k=0;k<n;k++)
    {
            for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
        {
            a1(j,k)= q(i,j,k);
            a2(j,k)= q(i,j,k);
            a3(j,k)= q(i,j,k);
        }

  Ang1 = orienMat_whole(a1,n);
  Ang2 = orienMat_whole(a2,n);
  Ang3 = orienMat_whole(a3,n);
    }
Q2ang_result r;
        r.ang1=Ang1;
        r.ang2=Ang2;
        r.ang3=Ang3;

        return r;
}
/* converts (X,Y,Z) to (Theta,Phi) coordinates on the unit sphere. Input, real X, Y, Z, the Cartesian coordinates of a point on the unit sphere.
Output, real T, P, the Theta and Phi coordinates of the point*/

xyz2thetaphi_result xyz2thetaphi(Tensor<double,1> x, Tensor<double,1> y, Tensor<double,1> z,int n)
{
    Tensor<double,1> P(n);     P.setZero();
    Tensor<double,1> T(n);     T.setZero();
    for(int i=0;i<n;i++)
    {
      P(i)= acos(z(i));
      double s= pow(x(i),2) + pow(y(i),2);
      if(s != 0 )
        T(i)= acos(x(i)/s);
      else
        T(i)= acos(x(i));

      if(y(i)<0)
        T(i)= T(i) + PI;
    }

  xyz2thetaphi_result r;
  r.t= T;
  r.p= P;

  return r;
}
// function to be used in gauss legendre quadrature
inline double XI(double x, double y, int i)
{
    if(i==0)
        return cos(x)*sin(y);
    if(i==1)
        return sin(x)*sin(y);
    else
        return cos(y);

}
// function to be used in gauss legendre quadrature
double Ainv(double x, double y, int i, int j,Tensor<double,4> Cm)
{

  double xi[3];
        xi[0]= cos(x)*sin(y);
        xi[1]= sin(x)*sin(y);
        xi[2]= cos(y);


    Tensor<double,2> A(3,3);          A.setZero();
for(int ii =0;ii<3;ii++)
 {
       for(int jj =0;jj<3;jj++)
       {
           for(int m =0;m<3;m++)
             {
               for(int nn =0;nn<3;nn++)

			        {

                    A(ii,jj)=Cm(ii,m,jj,nn)*xi[m]*xi[nn];

			        }
             }
       }
 }
// for incompressible...define Av

Tensor<double,2>Av(4,4);        Av.setZero();
for(int iii =0;iii<3;iii++)
 {
       for(int jij =0;jij<3;jij++)
       {
        Av(iii,jij)= A(iii,jij);
       }
}
Av(0,3)= xi[0];
Av(1,3)= xi[1];
Av(2,3)= xi[2];
Av(3,3)= 0;
Av(3,0)=xi[0];
Av(3,1)=xi[1];
Av(3,2)=xi[2];


	double det;

	det =  Av(0,0)*Av(1,1)*Av(2,2)*Av(3,3) + Av(0,0)*Av(1,2)*Av(2,3)*Av(3,1) + Av(0,0)*Av(1,3)*Av(2,1)*Av(3,2)
	     + Av(0,1)*Av(1,0)*Av(2,3)*Av(3,2) + Av(0,1)*Av(1,2)*Av(2,0)*Av(3,3) + Av(0,1)*Av(1,3)*Av(2,2)*Av(3,0)
		 + Av(0,2)*Av(1,0)*Av(2,1)*Av(3,3) + Av(0,2)*Av(1,1)*Av(2,3)*Av(3,0) + Av(0,2)*Av(1,3)*Av(2,0)*Av(3,1)
		 + Av(0,3)*Av(1,0)*Av(2,2)*Av(3,1) + Av(0,3)*Av(1,1)*Av(2,0)*Av(3,2) + Av(0,3)*Av(1,2)*Av(2,1)*Av(3,0)
		 - Av(0,0)*Av(1,1)*Av(2,3)*Av(3,2) - Av(0,0)*Av(1,2)*Av(2,1)*Av(3,3) - Av(0,0)*Av(1,3)*Av(2,2)*Av(3,1)
		 - Av(0,1)*Av(1,0)*Av(2,2)*Av(3,3) - Av(0,1)*Av(1,2)*Av(2,3)*Av(3,0) - Av(0,1)*Av(1,3)*Av(2,0)*Av(3,2)
		 - Av(0,2)*Av(1,0)*Av(2,3)*Av(3,1) - Av(0,2)*Av(1,1)*Av(2,0)*Av(3,3) - Av(0,2)*Av(1,3)*Av(2,1)*Av(3,0)
		 - Av(0,3)*Av(1,0)*Av(2,1)*Av(3,2) - Av(0,3)*Av(1,1)*Av(2,2)*Av(3,0) - Av(0,3)*Av(1,2)*Av(2,0)*Av(3,1);

    if(det==0.0){
		cout<< "det Av is zero,hence returning Av itself !\n ";
     exit(1);
	}

    Tensor<double,2> B(4,4);
    B.setZero();

    B(0,0)=(Av(1,1)*Av(2,2)*Av(3,3) + Av(1,2)*Av(2,3)*Av(3,1) + Av(1,3)*Av(2,1)*Av(3,2) - Av(1,1)*Av(2,3)*Av(3,2)- Av(1,2)*Av(2,1)*Av(3,3) - Av(1,3)*Av(2,2)*Av(3,1))/det;
    B(0,1)=(Av(0,1)*Av(2,3)*Av(3,2) + Av(0,2)*Av(2,1)*Av(3,3) + Av(0,3)*Av(2,2)*Av(3,1) - Av(0,1)*Av(2,2)*Av(3,3)- Av(0,2)*Av(2,3)*Av(3,1) - Av(0,3)*Av(2,1)*Av(3,2))/det;
	B(0,2)=(Av(0,1)*Av(1,2)*Av(3,3) + Av(0,2)*Av(1,3)*Av(3,1) + Av(0,3)*Av(1,1)*Av(3,2) - Av(0,1)*Av(1,3)*Av(3,2)- Av(0,2)*Av(1,1)*Av(3,3) - Av(0,3)*Av(1,2)*Av(3,1))/det;
    B(0,3)=(Av(0,1)*Av(1,3)*Av(2,2) + Av(0,2)*Av(1,1)*Av(2,3) + Av(0,3)*Av(1,2)*Av(2,1) - Av(0,1)*Av(1,2)*Av(2,3)- Av(0,2)*Av(1,3)*Av(2,1) - Av(0,3)*Av(1,1)*Av(2,2))/det;

    B(1,0)=(Av(1,0)*Av(2,3)*Av(3,2) + Av(1,2)*Av(2,0)*Av(3,3) + Av(1,3)*Av(2,2)*Av(3,0) - Av(1,0)*Av(2,2)*Av(3,3)- Av(1,2)*Av(2,3)*Av(3,0) - Av(1,3)*Av(2,0)*Av(3,2))/det;
	B(1,1)=(Av(0,0)*Av(2,2)*Av(3,3) + Av(0,2)*Av(2,3)*Av(3,0) + Av(0,3)*Av(2,0)*Av(3,2) - Av(0,0)*Av(2,3)*Av(3,2)- Av(0,2)*Av(2,0)*Av(3,3) - Av(0,3)*Av(2,2)*Av(3,0))/det;
	B(1,2)=(Av(0,0)*Av(1,3)*Av(3,2) + Av(0,2)*Av(1,0)*Av(3,3) + Av(0,3)*Av(1,2)*Av(3,0) - Av(0,0)*Av(1,2)*Av(3,3)- Av(0,2)*Av(1,3)*Av(3,0) - Av(0,3)*Av(1,0)*Av(3,2))/det;
	B(1,3)=(Av(0,0)*Av(1,2)*Av(2,3) + Av(0,2)*Av(1,3)*Av(2,0) + Av(0,3)*Av(1,0)*Av(2,2) - Av(0,0)*Av(1,3)*Av(2,2)- Av(0,2)*Av(1,0)*Av(2,3) - Av(0,3)*Av(1,2)*Av(2,0))/det;

	B(2,0)=(Av(1,0)*Av(2,1)*Av(3,3) + Av(1,1)*Av(2,3)*Av(3,0) + Av(1,3)*Av(2,0)*Av(3,1) - Av(1,0)*Av(2,3)*Av(3,1)- Av(1,1)*Av(2,0)*Av(3,3) - Av(1,3)*Av(2,1)*Av(3,0))/det;
    B(2,1)=(Av(0,0)*Av(2,3)*Av(3,1) + Av(0,1)*Av(2,0)*Av(3,3) + Av(0,3)*Av(2,1)*Av(3,0) - Av(0,0)*Av(2,1)*Av(3,3)- Av(0,1)*Av(2,3)*Av(3,0) - Av(0,3)*Av(2,0)*Av(3,1))/det;
    B(2,2)=(Av(0,0)*Av(1,1)*Av(3,3) + Av(0,1)*Av(1,3)*Av(3,0) + Av(0,3)*Av(1,0)*Av(3,1) - Av(0,0)*Av(1,3)*Av(3,1)- Av(0,1)*Av(1,0)*Av(3,3) - Av(0,3)*Av(1,1)*Av(3,0))/det;
    B(2,3)=(Av(0,0)*Av(1,3)*Av(2,1) + Av(0,1)*Av(1,0)*Av(2,3) + Av(0,3)*Av(1,1)*Av(2,0) - Av(0,0)*Av(1,1)*Av(2,3)- Av(0,1)*Av(1,3)*Av(2,0) - Av(0,3)*Av(1,0)*Av(2,1))/det;

   	B(3,0)=(Av(1,0)*Av(2,2)*Av(3,1) + Av(1,1)*Av(2,0)*Av(3,2) + Av(1,2)*Av(2,1)*Av(3,0) - Av(1,0)*Av(2,1)*Av(3,2)- Av(1,1)*Av(2,2)*Av(3,0) - Av(1,2)*Av(2,0)*Av(3,1))/det;
   	B(3,1)=(Av(0,0)*Av(2,1)*Av(3,2) + Av(0,1)*Av(2,2)*Av(3,0) + Av(0,2)*Av(2,0)*Av(3,1) - Av(0,0)*Av(2,2)*Av(3,1)- Av(0,1)*Av(2,0)*Av(3,2) - Av(0,2)*Av(2,1)*Av(3,0))/det;
   	B(3,2)=(Av(0,0)*Av(1,2)*Av(3,1) + Av(0,1)*Av(1,0)*Av(3,2) + Av(0,2)*Av(1,1)*Av(3,0) - Av(0,0)*Av(1,1)*Av(3,2)- Av(0,1)*Av(1,2)*Av(3,0) - Av(0,2)*Av(1,0)*Av(3,1))/det;
   	B(3,3)=(Av(0,0)*Av(1,1)*Av(2,2) + Av(0,1)*Av(1,2)*Av(2,0) + Av(0,2)*Av(1,0)*Av(2,1) - Av(0,0)*Av(1,2)*Av(2,1)- Av(0,1)*Av(1,0)*Av(2,2) - Av(0,2)*Av(1,1)*Av(2,0))/det;




 return B(i,j);
}

// function f for gauss legendre quadrature
double f(double x, double y, void* data,int ii, int jj, int kk, int ll,Tensor<double,4> Cm, Tensor<double,1> a)
                        {

                      double rho= pow((pow((a(0)*XI(x,y,0)),2) + pow((a(1)*XI(x,y,1)),2) + pow((a(2)*XI(x,y,2)),2)),1.5);

                         return ( XI(x,y,jj)*XI(x,y,ll)*Ainv(x,y,ii,kk,Cm) *sin(y)) /rho;
			            }

// green tensor calculation
Tensor<double,4> TGreen(Tensor<double,1> a, Tensor<double,4> Cm, int n)
{

 Tensor<double,4> T(3,3,3,3);
 T.setZero();
double kk =(a(0)*a(1)*a(2))/(4*PI);
 for(int i =0;i<3;i++)
 {
       for(int  j =0;j<3;j++)
       {
           for(int k =0;k<3;k++)
             {
               for(int l =0;l<3;l++)

			        {

			          T(i,j,k,l)= kk * Gauss_legendre_2D_cube(n,f,NULL,0,PI,0,2*PI,i,j,k,l,Cm,a);

                        T(k,j,i,l) = T(i,j,k,l);
                       T(i,l,k,j) = T(i,j,k,l);
                       T(k,l,i,j) = T(i,j,k,l);
			        }
             }
       }
 }
return T;

      }



Tensor<double,4> sumalongdim5(Tensor<double,5> a , int n)
{
    Tensor<double,4> b(3,3,3,3);
    b.setZero();
 double bo;
    for(int i=0;i<3;i++)
        {for(int j=0;j<3;j++)
       {
        for(int k=0;k<3;k++)
     {
             for(int l=0;l<3;l++)
    {
             bo=0;
                for(int m=0;m<n;m++)
               {

               bo= a(i,j,k,l,m)+ bo;

                }
                b(i,j,k,l)=bo;


    }
     }}
        }
      return b;
}


Tensor<double,5> M(Tensor<double,2> eta, int n)
{
    Tensor<double,5> m(3,3,3,3,n);
    m.setConstant(0);
  FourIdentityTensors r;
r = FourIdentity();
    for(int i=0;i<n;i++)
  {
      for(int ii=0;ii<3;ii++)
        {

        for(int jj=0;jj<3;jj++)
        {
           for(int kk=0;kk<3;kk++)
               {
               for(int ll=0;ll<3;ll++)
                     m(ii,jj,kk,ll,i)= (1/(2*eta(0,i)))*r.JD(ii,jj,kk,ll);
               }
        }
        }
  }

  return m;
}

Tensor<double,3> E(Tensor<double,2> eta, Tensor<double,2> D,Tensor<double,2> Ne, int n)
{
    Tensor<double,3> e0(3,3,n);
    e0.setConstant(0);

    for(int i=0;i<n;i++)
  {

      for(int ii=0;ii<3;ii++)
        {

        for(int jj=0;jj<3;jj++)
        e0(ii,jj,i)= (1-Ne(0,n))*D(ii,jj);


        }
  }

  return e0;
}


Tensor<double, 5> mtan(Tensor<double, 5> m,Tensor<double,2> Ne,int n)   //generate the tangent compliances(mtan) of RDEs
{
      Tensor<double,5> Mtan(3,3,3,3,n);
      Mtan.setConstant(0);

   for(int j=0;j<n;j++)
      {
          for(int iii=0;iii<3;iii++)
        {

        for(int jjj=0;jjj<3;jjj++)
        {
          for(int kkk=0;kkk<3;kkk++)
               {
               for(int lll=0;lll<3;lll++)
{

                 Mtan(iii,jjj,kkk,lll,j)= Ne(0,j)* m(iii,jjj,kkk,lll,j) ;

}

           }
        }
        }
       }
return Mtan;

}

Tensor <double,4> mbar(Tensor<double,5> mTan, int n)
{

Tensor<double, 4> tan(3,3,3,3);
tan.setZero();
    double bo;
    for(int p=0;p<3;p++)
        {for(int qq=0;qq<3;qq++)
       {
        for(int k=0;k<3;k++)
     {
             for(int l=0;l<3;l++)
    {
             bo=0;
                for(int m=0;m<n;m++)
               {

               bo= (mTan(p,qq,k,l,m)+ bo);

                }
                tan(p,qq,k,l)=bo/n;


    }
     }}
        }

        return tan;
        }

inline Tensor<double,3> onetothree(Tensor<double,1> a, int i,int j,int n, int steps)
        {
            Tensor<double,3> Q(3,n,steps);
            Q.setZero();
            for(int l=0;l<3;l++)
            Q(l,i,j)=a(l);

           return Q;
        }

        inline Tensor<double,2> onetotwo(Tensor<double,1> a, int i)
        {
            Tensor<double,2> Q(3,i);
            Q.setZero();
            for(int l=0;l<3;l++)
            Q(l,i)=a(l);

           return Q;
        }

        inline Tensor<double,1> twotoone(Tensor<double,2> a, int i)
        {
            Tensor<double,1> Q(3);
            Q.setZero();
            for(int l=0;l<3;l++)
            Q(l)=a(l,i);

           return Q;
        }

inline Tensor<double,4> twotofour(Tensor<double,2> a, int i,int step,int n,int steps)
        {
            Tensor<double,4> Q(3,3,n,steps);
            Q.setZero();
          for(int k=0;k<3;k++)
          {
            for(int l=0;l<3;l++)
            Q(k,l,i,step)=a(k,l);
          }
           return Q;
        }

inline Tensor<double,3> twotothree(Tensor<double,2> a, int i,int n)
        {
            Tensor<double,3> Q(3,3,n);
            Q.setZero();
          for(int k=0;k<3;k++)
          {
            for(int l=0;l<3;l++)
               {Q(k,l,i)=a(k,l);}
          }
           return Q;
        }

        inline Tensor<double,2> threetotwo(Tensor<double,3> a, int i)
        {
            Tensor<double,2> Q(3,3);
            Q.setZero();
          for(int k=0;k<3;k++)
          {
            for(int l=0;l<3;l++)
            Q(k,l)=a(k,l,i);
          }
           return Q;
        }
 Tensor<double,5> fourtofive(Tensor<double,4> a, int i,int n)
{
   Tensor<double,5> Q(3,3,3,3,n);
            Q.setZero();
        for(int j=0;j<3;j++)
          {
          for(int k=0;k<3;k++)
           {
             for(int l=0;l<3;l++)
             {
               for(int m=0;m<3;m++)
             Q(j,k,l,m,i)= a(j,k,l,m);
             }
           }
          }
           return Q;
}
        inline Tensor<double,4> fivetofour(Tensor<double,5> a, int i)
        {
            Tensor<double,4> Q(3,3,3,3);
            Q.setZero();
        for(int j=3;j>0;j--)
          {
          for(int k=3;k>0;k--)
           {
             for(int l=3;l>0;l--)
             {
               for(int m=3;m>0;m--)
            {Q(j-1,k-1,l-1,m-1)=a(j-1,k-1,l-1,m-1,i);}
             }
           }
          }
           return Q;
        }

        inline Tensor<double,2> Tensor_product(Tensor<double,2>a, Tensor<double,2> b)  // try operator overloading later
{
    Tensor<double,2> c(3,3);
       c.setZero();
    for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
            double s=0;
            for(int k=0;k<3;k++)
            {
              c(i,j)= s + a(i,k)*b(k,j);
              s=c(i,j);
            }
        }

}
    return c;
}
inline Tensor<double,1> diag(Tensor<double,2> a, double t)
{   Tensor<double,1> b(3);
    b.setZero();
    for(int i=0;i<3;i++)
    {
        b(i)= exp(a(i,i)*t);
    }
    return b;

}

Tensor<double,2> sortingfunc(Tensor<double,1> a,Tensor<double,2> q)  // sorting function
{
    Tensor<double,2> QA(3,4);
    QA.setZero();
     double s=0;
    if (a(0)>a(1) && a(0)>a(2))
    {
        s= a(1);
        if (a(1)<a(2))
       {
           a(1)= a(2);
           a(2)= s;

           double p=0;
         for(int i=3;i>0;i--)
         {
             p= q(1,i-1);
             q(1,i-1)=q(2,i-1);
             q(2,i-1)=p;
         }

         }
    }

    if(a(1)>a(0) && a(1)>a(2))
 {
     s=a(0);
     a(0)=a(1);
     a(1)=s;
     double kk=0;
         for(int jj=3;jj>0;jj--)
         {
             kk= q(0,jj-1);
             q(0,jj-1)=q(1,jj-1);
             q(1,jj-1)=kk;
         }
     if(a(1)<a(2))
     {
         double k=0;
         for(int j=3;j>0;j--)
         {
             k= q(1,j-1);
             q(1,j-1)=q(2,j-1);
             q(2,j-1)=k;
         }

         s=a(1);
         a(1)=a(2);
         a(2)=s;
     }

 }
  if(a(2)>a(0) && a(2)>a(1))
 {
     s=a(0);
     a(0)=a(2);
     a(2)=s;
     double mk=0;
         for(int jj=3;jj>0;jj--)
         {
             mk= q(0,jj-1);
             q(0,jj-1)=q(2,jj-1);
             q(2,jj-1)=mk;
         }
     if(a(1)<a(2))
     {
         double l=0;
         for(int m=3;m>0;m--)
         {
             l= q(1,m-1);
             q(1,m-1)=q(2,m-1);
             q(2,m-1)=l;
         }
         s=a(1);
         a(1)=a(2);
         a(2)=s;
     }

 }
for(int n=3;n>0;n--)
{
    for(int nn=3;nn>0;nn--)
    {
        QA(n-1,nn-1)=q(n-1,nn-1);
    }
}


QA(0,3)= a(0);
QA(1,3)= a(1);
QA(2,3)= a(2);
return QA;

}

Tensor<double,2> sortingfunc2(Tensor<double,2> qa) // sorting function for (3*4) matrix
{

     double s=0;
    if (qa(0,3)>qa(1,3) && qa(0,3)>qa(2,3))
    {
        s= qa(1,3);
        if (qa(1,3)<qa(2,3))
       {
           qa(1,3)= qa(2,3);
           qa(2,3)= s;

           double p=0;
         for(int i=3;i>0;i--)
         {
             p= qa(1,i-1);
             qa(1,i-1)=qa(2,i-1);
             qa(2,i-1)=p;
         }

         }
    }

    if(qa(1,3)>qa(0,3) && qa(1,3)>qa(2,3))
 {
     s= qa(0,3);
     qa(0,3)= qa(1,3);
     qa(1,3)= s;
     double kk=0;
         for(int jj=3;jj>0;jj--)
         {
             kk= qa(0,jj-1);
             qa(0,jj-1)=qa(1,jj-1);
             qa(1,jj-1)=kk;
         }
     if(qa(1,3)<qa(2,3))
     {
         double k=0;
         for(int j=3;j>0;j--)
         {
             k= qa(1,j-1);
             qa(1,j-1)=qa(2,j-1);
             qa(2,j-1)=k;
         }

         s=qa(1,3);
         qa(1,3)=qa(2,3);
         qa(2,3)=s;
     }

 }
  if(qa(2,3)>qa(0,3) && qa(2,3)>qa(1,3))
 {
     s=qa(0,3);
     qa(0,3)=qa(2,3);
     qa(2,3)=s;
     double mk=0;
         for(int jj=3;jj>0;jj--)
         {
             mk= qa(0,jj-1);
             qa(0,jj-1)=qa(2,jj-1);
             qa(2,jj-1)=mk;
         }
     if(qa(1,3)<qa(2,3))
     {
         double l=0;
         for(int m=3;m>0;m--)
         {
             l= qa(1,m-1);
             qa(1,m-1)=qa(2,m-1);
             qa(2,m-1)=l;
         }
         s=qa(1,3);
         qa(1,3)=qa(2,3);
         qa(2,3)=s;
     }

 }
return qa;

}
int main()
{

// MOPLA begins
/*
Modeling the mechanicall behavior of the heterogeneous rock mass which is composed of rheologically distinctive elements(RDEs).
Here, the heterogeneous rock mass is replaced by a homogeneous effective medium(HEM) and the rehological properties of HEM at a point
are represented by the overall effective propertives of all surrounding RDEs.
Scripts are based on the self-consistent solution of the partitioning and homogenization equations.(Jiang, 2013; Jiang, 2014)*/

//**************** Input parameters***********************

// Imposed Macro-scale flow field

Tensor<double,2> L(3,3);

L.setZero();
L(0,0)= 0.5;
L(0,1)= 1;
L(1,1)= -0.5;

// the number of RDEs(Rheologically Distinctive Elements)
int n     = 100;

//the maximum semi-axils of RDEs (a1>a2>a3; a3=1)
  double a1    = 10;
// the maximum viscosity of RDEs
   double vmax  = 10;
// the minimum viscosity of RDEs
   double vmin  = 1;
// the maximum stress exponent of RDEs
   double Nmax  = 4;
//  the minimum stress exponent of RDEs
   double Nmin  = 1;
// time increment of each step during the computation; see Jiang (2007) for the choice of tincr.
   double tincr = 0.01;
// total steps of the computation
   int steps = 100;

// decomposition of the bulk flow L into a strain rate tensor D and a vorticity tensor W

Tensor<double,2> D(3,3);    D.setZero();
Tensor<double,2> W(3,3);    W.setZero();
   D = 0.5 * (L + transp(L) );
   W = 0.5 * (L - transp(L));

//  generate 4th-order identity tensors
FourIdentityTensors r;
r = FourIdentity();
Tensor<double,4> Jd;              Jd.setZero();
    Tensor<double,4> Js;          Js.setZero();
    Tensor<double,4> Ja;          Ja.setZero();
    Tensor<double,4> Jm;          Jm.setZero();
Jd=r.JD;
Js=r.JS;
Ja=r.JA;
Jm=r.JM;

//**************** Initial shapes, orientations and constitutive properties of RDEs*******************

//generate a population of uniformly distributed RDEs with a1:a2:1,a2 from a1 to 1.
   RandAANG_result rang;
   rang=RandAANG(a1,n);

Tensor<double,2> randm(1,n);        randm.setZero();
Tensor<double,2> eta(1,n);          eta.setZero();
Tensor<double,2> Ne(1,n);           Ne.setZero();
randm.setRandom();


//generate the viscosities(eta) of RDEs
   eta = vmin + (vmax-vmin)*randm;

//  generate the stress exponents(Ne) of RDEs
   Ne  = Nmin + (Nmax-Nmin)*randm;

Tensor<double,2> A(3,n);
A.setZero();
A= rang.a;

//generate the transformation tensors(q),compliance tensors(m)and pre-strain rate term(e0) of RDEs
 Tensor<double,3> q(3,3,n);              q.setZero();
 Tensor<double,5> m(3,3,3,3,n);          m.setZero();
 Tensor<double,3> e0(3,3,n);             e0.setZero();

q= Qvec(rang.ang,n);
m = M(eta,n);
e0= E(eta,D,Ne,n);




  //strain rate invariant at which ellipsoid viscosity is defiend
  Tensor<double,2> REF(1,n);
  REF.setConstant(1);
  REF=REF*Inva(D);


// *************initial guess of the properties of the HEM******************

// generate the tangent compliances(mtan) of RDEs

  Tensor<double,5> MTAN(3,3,3,3,n);
  MTAN.setZero();
 MTAN= mtan(m,Ne,n);

// initial homogenized compliance(M_bar) of HEM
Tensor<double,4> M_bar(3,3,3,3);
   M_bar.setZero();
   M_bar = mbar(MTAN,n);

// initial homogenized stiffness(C_bar) of HEM
   Tensor<double,4> C_bar(3,3,3,3);
   C_bar.setZero();
   C_bar = FourTensorInv(M_bar);           // This C_bar is used in Green Tensor calculation( obvi, after transformation to the RDE axis)

//initial pre-strain rate term(E0) of HEM
 Tensor<double,4> mm(3,3,3,3);
 mm.setZero();
   mm = FourTensorInv(mbar(m,5));

   Tensor<double,4> t1(3,3,3,3);
   t1.setZero();
   t1=contract(M_bar,mm);

   Tensor<double,2> E0(3,3);
   E0.setZero();
   E0= Multiply((Jd-t1),D);

//  the macroscopic stress(Stress_bar) of HEM
   Tensor<double,2> Stress_bar(3,3);
   Stress_bar.setZero();
   Stress_bar = Multiply(C_bar,(D-E0));

  // far-field stress
   Tensor<double,2> SIGMA(3,3);
   SIGMA.setZero();
   SIGMA = Stress_bar;

//  invariant of the HEM deviatoric stress(macroscale)
double BStress= Inva(Multiply(Jd,Stress_bar));

 //partitioned stress fields inside RDEs
 //  sigma            = repmat(Stress_bar,1,1,n);// got it !!


    // lets first solve for an RDE independently .....

// preallocate variables:--------------------------------------------------

Tensor<double,5> c_bar_all(3,3,3,3,n);                            c_bar_all.setZero();
Tensor<double,5> T_all(3,3,3,3,n);                                T_all.setZero();
Tensor<double,5> H_arc_all(3,3,3,3,n);                            H_arc_all.setZero();
Tensor<double,3> e_all(3,3,n);                                    e_all.setZero();
Tensor<double,4> Q_evl(3,3,n,steps);                              Q_evl.setZero();
Tensor<double,3> A_evl(3,n,steps);                                A_evl.setZero();
Tensor<double,5> C_bar_evl(3,3,3,3,steps);                        C_bar_evl.setZero();
Tensor<double,5> M_bar_evl(3,3,3,3,steps);                        M_bar_evl.setZero();
Tensor<double,2>vis(1,n);                                         vis.setZero();
Tensor<double,2>eI(1,n);                                          eI.setZero();
Tensor<double,4> New_M_bar(3,3,3,3);                              New_M_bar.setZero();
         Tensor<double,2> Q_RDE(3,3);                             Q_RDE.setZero();
         Tensor<double,4> C_RDE(3,3,3,3);                         C_RDE.setZero();
         Tensor<double,1> A_RDE(3);                               A_RDE.setZero();
         Tensor<double,4> T(3,3,3,3);                             T.setZero();
         Tensor<double,2> e_RDE(3,3);                                 e_RDE.setZero();               //partitioned strain rate inside RDE
         Tensor<double,2> sigma(3,3);                             sigma.setZero();                    //partitioned stress field inside RDE
         Tensor<double,4> s1(3,3,3,3);                            s1.setZero();    // defining terms for calculating the 4th-order Interaction tensor(H_arc)
         Tensor<double,4> s(3,3,3,3);                             s.setZero();
         Tensor<double,4> H_arc(3,3,3,3);                         H_arc.setZero();
         Tensor<double,4> t0(3,3,3,3);                            t0.setZero();   // defining terms for calculating the 4th-order stress-partitioning tensor(B)

         Tensor<double,4> B(3,3,3,3);                             B.setZero();
         Tensor<double,2> t2(3,3);                                t2.setZero();
         Tensor<double,2> e0_RDE(3,3);                            e0_RDE.setZero();                 // pre-strain rate in RDE
         Tensor<double,4> m_RDE(3,3,3,3);                          m_RDE.setZero();
         Tensor<double,1> Ne_RDE(1);                                Ne_RDE.setZero();
         Tensor<double,1> REF_RDE(1);                               REF_RDE.setZero();
         Tensor<double,1> eta_RDE(1);                               eta_RDE.setZero();
         Tensor<double,2> beta(3,3);                                beta.setZero();

        double alpha=0;
        double t3=0;
        double eta_RDEval=0;
        double REF_RDEval=0;
        double v1=0;

        Tensor<double,4> u1(3,3,3,3);               u1.setZero();
        Tensor<double,2> u2(3,3);               u2.setZero();
        Tensor<double,2> we(3,3);                   we.setZero();
        Tensor<double,2> wEp(3,3);                 wEp.setZero();
        Tensor<double,4> S_eshelby(3,3,3,3);         S_eshelby.setZero();
        Tensor<double,4> PI_eshelby(3,3,3,3);        PI_eshelby.setZero();
        Tensor<double,4> z(3,3,3,3);                  z.setZero();
        Tensor<double,2> de(3,3);                   de.setZero();
        Tensor<double,2> D_bar(3,3);                D_bar.setZero();
        Tensor<double,2> W_bar(3,3);               W_bar.setZero();
        Tensor<double,2> theta(3,3);                 theta.setZero();
        Tensor<double,2> qa(3,4);                   qa.setZero();
        Tensor<double,1> aa(3);                    aa.setZero();
        Tensor<double,2> qq(3,3);                         qq.setZero();


for(int step=0; step<steps; step ++)    // this loop is the ultimate loop.......defines the number of steps our deformation will go......so no bakchodi with this !!!

 {

for(int g=100;g>0;g--)//Outer loop:When all elements are calculated, using the new set of M(k),B(k),e0(k),beta(k) update new M_bar and E0 until it approaches balance.
{



        Tensor<double,4> MB(3,3,3,3);             MB.setZero();   //  for <M:B>          ask Mengmeng or lucy, why these are  required ??....or understand yourself !
        Tensor<double,4> BB(3,3,3,3);             BB.setZero();   //  for <B>
        Tensor<double,2> Mb(3,3);                 Mb.setZero();   //  for <M:B+e0>
        Tensor<double,2> bb(3,3);                 bb.setZero();   //  for <beta>
        Tensor<double,4> invB(3,3,3,3);           invB.setZero();



      for(int i=0; i<n; i++)   // so this loop is for individual calculation of  stress and strain for an RDE, where n is the number of RDEs.// use multiple threads
      {

        sigma = Stress_bar;    // partitioned stress field inside the RDE equalled to the Macroscopic stress of HEM !!

         Q_RDE=threetotwo(q,i);
         C_RDE= TransforM(C_bar,Q_RDE);
         A_RDE= twotoone(A,i);

         T=TGreen(A_RDE,C_RDE,5); // T -> green tensor calculated for the RDE : 50 here is the number OF steps considered for green tensor calculation....see for this later
         T_all= fourtofive(T,i,n);


                    // initial value of sigma
                   Tensor<double,2> stress(3,3);            stress.setZero();
                   stress= sigma;

         //4th-order Interaction tensor(H_arc)

            s1 = contract(Jd,contract(T,C_RDE));
            s= TransforM(s1,transp(Q_RDE));
            H_arc =contract(FourTensorInv(FourTensorInv(s)-Jd),M_bar);


     // some terms required in the next loop:
         Ne_RDE= twotoone(Ne,i);
         m_RDE= fivetofour(m,i);
         double Ne_RDEval;
          Ne_RDEval = Ne_RDE(0);
         double e1,v1;
         e1=0; v1=0;
         e0_RDE= threetotwo(e0,i);             // this value of e0_RDE will be required only in the first time loop//


         for(int nn=100;nn >0;nn--)
         {


             // the 4th-order stress-partitioning tensor(B)
             t0 = Ne_RDEval * m_RDE ;
             t1= FourTensorInv(t0 + H_arc);
             B= contract(t1,(M_bar + H_arc));                    // M_bar here is initial homogenised compliance of HEM

            t2= E0 - e0_RDE;
           //  the second order stress-partitioning tensor(beta)
             beta = Multiply(t1,t2);

            // calculate new partitioned stress field inside RDE
             sigma = Multiply(B,SIGMA) + beta;                     // final value of sigma

               alpha = fabs(Inva(sigma-stress)/Inva(stress));
              stress= sigma;
            // calculate new partitioned strain rate of RDE

            e_RDE= Multiply(m_RDE,sigma);
           // calculate new pre-strain rate term of RDE
            e0_RDE = (1-Ne_RDEval)*e_RDE;

         // calculate new strain rate invariant
             e1= Inva(e_RDE);

         //calculate new viscosity of RDE at the new strain rate state
              t3 = (1/Ne_RDEval)-1;
             REF_RDE = twotoone(REF,i);
             eta_RDE = twotoone(eta,i);
              eta_RDEval =  eta_RDE(0);
              REF_RDEval =  REF_RDE(0);

              v1 = pow((e1/REF_RDEval),t3) * eta_RDEval;
          //  calculate new compliance tensor of RDE
                m_RDE = 1/(2*v1)*Jd;

          if(alpha<0.01)
            break;
         }

         e_all= twotothree(e_RDE,i,n);

          //  e0 = twotothree(e0_RDE,i);     // updating initial pre strain rate.....dont know if this is required.!!


      // update the strain-rate invariant for an RDE
            eI(0,i)  = e1;
      //  update the effective viscosity for an RDE
            vis(0,i) = v1;

            MB = MB+contract(t0,B);                  // <M(tan):B>*n%%
            Mb = Mb+Multiply(t0,beta)+e0_RDE;      // %%<M(tan):beta+e0>*n%%
            BB = BB+B;                                // %%<B>*n$$
            bb = bb+beta;                             // %%<beta>*n%%

      }


      //calculate new homogenized compliance for HEM
         BB= (1/n) * BB;
         MB= (1/n) * MB;
        invB         = FourTensorInv(BB);
        New_M_bar    = contract(MB,invB);
 // compare the current homogenized compliance and the previous one
        double delta        = normfunc(New_M_bar-M_bar)/normfunc(M_bar);
//  replace the previous homogenized compliance with the current one
        M_bar        = New_M_bar;
//  calculate new homogenized stiffness for HEM
        C_bar        = FourTensorInv(M_bar);
// calculate new pre-strain rate term for HEM
        E0           = ((1/n) * Mb)-Multiply(M_bar,((1/n) * bb));
// calculate new homogeneous macroscopic stress for HEM
        Stress_bar   = Multiply(C_bar,D-E0);
//  calculate new far-field stress
        SIGMA        = Multiply(invB,(Stress_bar - ((1/n)* bb)));
//  calculate new secound invariant of the macroscopic deviatoric stress

        double NewStress    = Inva(Multiply(Jd,Stress_bar));
// compare the current macroscopic deviatoric stress and the previous one
        double gamma        = fabs((NewStress/BStress)-1);
//  replace the previous invariant of the macroscopic deviatoric stress with the current one
        BStress      = NewStress;
//The outer loop continues until the current homogenized compliance and macroscopic deviatoric stress coincide with the previous ones respectively within specific tolerances

      if (delta < 0.02 && gamma<0.02)
            break;


}

 // update the strain-rate invariants for all RDEs
     REF=eI;
 // update the effective viscosities for all RDEs
     eta=vis;

// write updated C_bar to C_bar_evl

C_bar_evl=fourtofive(C_bar, step,n);

// write updated M_bar to M_bar_evl

M_bar_evl=fourtofive(M_bar, step,n);

// Evolution of RDEs------

for(int o=0;o<n;o++)
{
    //describe D,W in the RDE's coordinate system

      Q_RDE=threetotwo(q,o);
D_bar= Tensor_product(Q_RDE,Tensor_product(D,transp(Q_RDE)));
W_bar= Tensor_product(Q_RDE,Tensor_product(W,transp(Q_RDE)));

//calculate Eshelby Tensors(S, PI) based on T


        T= fivetofour(T_all,o);
        C_RDE = fivetofour(c_bar_all,o);
        z    = contract(T,C_RDE);
        S_eshelby    = contract(Jd,z);
        PI_eshelby   = contract(Ja,z);


// strain rate of RDE
        e_RDE=threetotwo(e_all,o);
de = Tensor_product(Q_RDE,Tensor_product(e_RDE,transp(Q_RDE)));        // Ellipsoid strain-rate


// vorticity of RDE


        u1    = contract(PI_eshelby,FourTensorInv(S_eshelby));
        u2    = de - D_bar;
        A_RDE= twotoone(A,o);
        we    = Multiply(u1, u2)+ W_bar;
        wEp   = Wd(A_RDE, W_bar, de);

// angular velocity of RDE

        theta = we - wEp;
// update Q

         qq   = Tensor_product((RodrgRot(theta * -tincr)),Q_RDE);
// update a

         aa   = A_RDE*diag(de,tincr);
// make sure that Q and a are in the descending oreder of a(a1>=a2>=a3)

         qa = sortingfunc(aa,qq);
// Boudinage if the RDE is too elongated or flattened(a1:a3>100 or a2:a3>100)
         if ((qa(0,3)/qa(2,3))>100)
          {qa(1,4)=0.5*qa(1,4);}

        if ((qa(1,3)/qa(2,3))>100)
            {qa(1,3)=0.5*qa(1,3);}

        qa = sortingfunc2(qa);


        A_RDE(0)= qa(0,3);                                          // this has been done to avoid loops
        A_RDE(1)= qa(1,3);
        A_RDE(2)= qa(2,3);

        Q_RDE(0,0)= qa(0,0);   Q_RDE(0,1)=qa(0,1); Q_RDE(0,2)=qa(0,2);
        Q_RDE(1,0)= qa(1,0);   Q_RDE(1,1)=qa(1,1); Q_RDE(1,2)=qa(1,2);
        Q_RDE(2,0)= qa(2,0);   Q_RDE(2,1)=qa(2,1); Q_RDE(2,2)=qa(2,2);



// write updated Q to Q_evl
        Q_evl= twotofour(Q_RDE,o,step,n,steps);
//  write updated a to A_evl
        A_evl= onetothree(A_RDE,o,step,n,steps);

       }
cout<< ".";
  }    // **********1 step ends here ***********



//}


    return 0;

}
