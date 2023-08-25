#include <iostream>
#include <cstdio>
#include<cmath>
using namespace std;
void findcordinatesofp(float x1,float y1,float z1,float x2,float y2,float z2,float x3,float y3,float z3,float d12,float d23,
float d31,float r1,float r2,float r3)
{
    float r12 = (r1*r1-(x1*x1+y1*y1+z1*z1))-(r2*r2-(x2*x2+y2*y2+z2*z2));
    float r31 = (r3*r3-(x3*x3+y3*y3+z3*z3))-(r1*r1-(x1*x1+y1*y1+z1*z1));
     cout<<endl;
    cout<<endl;
    cout<<endl;
    cout<<"Equation of Plane intersection sphere with s1 as center and r1 as radius and sphere s2 as center and r2 as radius is:"<<endl;
    float l1=-2*(x1-x2),l2=-2*(x3-x1),m1=-2*(y1-y2),m2=-2*(y3-y1),n1=-2*(z1-z2),n2=-2*(z3-z1);
    printf("%fx+%fy+%fz=%f",l1,m1,n1,r12);
    cout<<endl;
    cout<<endl;
    cout<<endl;
    cout<<"Equation of Plane intersection sphere with s1 as center and r1 as radius and sphere s3 as center and r3 as radius is:"<<endl;
    printf("%fx+%fy+%fz=%f",l2,m2,n2,r31);
    cout<<endl;
    cout<<endl;
    cout<<endl;
    cout<<"We will be getting two equations of the form ax+bz=k and implies the planes are parallel to y-axis."<<endl;
    cout<<endl;
    cout<<endl;
   float lx1=(r12*n2-r31*n1)/(n2*l1-n1*l2);
   float lz1=(r12*l2-r31*l1)/(l2*n1-l1*n2);
   
   
    cout<<"Equation of the line intersecting both the planes is:"<<endl;
    printf("x/%f=z/%f",lx1,lz1);
    cout<<endl;
    cout<<endl;
    cout<<endl;
    float ly1=y1+abs(sqrt(pow(r1,2)-pow(x1-lx1,2)-pow(z1-lz1,2)));
    cout<<"Cordinates of Point P are:"<<endl;
    cout<<endl;
    printf("(%f,%f,%f)",lx1,ly1,lz1);
}
int main()
{
    float d12,d23,d31,x1,x2,x3,y1,y2,y3,z1,z2,z3,r1,r2,r3;
    cout<<"Please enter the cordinates of sensor 1, sensor2 and sensor3 (x1,y1,z1,x2,y2,z2,x3,y3,z3):"<<endl;
    cin>>x1>>y1>>z1>>x2>>y2>>z2>>x3>>y3>>z3;
    cout<<endl;
    cout<<"The distance between s1-s2, s2-s3, s3-s1 respectively are :"<<endl;
    cout<<endl;
    d12 = sqrt(pow(x1-x2,2)+pow(y1-y2,2)+pow(z1-z2,2));
    d23 = sqrt(pow(x3-x2,2)+pow(y3-y2,2)+pow(z3-z2,2));
    d31 = sqrt(pow(x1-x3,2)+pow(y1-y3,2)+pow(z1-z3,2));
    cout<<d12<<" "<<d23<<" "<<d31;
    cout<<endl;
    cout<<endl;
    cout<<endl;
    cout<<"Please enter the distance between s1-P, s2-P, s3-P respectively:"<<endl;
    cin>>r1>>r2>>r3;
    findcordinatesofp(x1,y1,z1,x2,y2,z2,x3,y3,z3,d12,d23,d31,r1,r2,r3);

    return 0;
}
