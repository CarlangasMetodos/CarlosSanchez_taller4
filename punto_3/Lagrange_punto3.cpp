#include <iostream>
using namespace std;

struct Data{
    int x, y;
};

struct Container{
    Data element;
};

double interpolate(Data f[], int xi, int n)
{
    double result = 0;

    for (int i=0; i<n; i++)
    {
        double term = f[i].y;
        for (int j=0;j<n;j++)
        {
            if (j!=i)
                term = term*(xi - f[j].x)/double(f[i].x - f[j].x);
        }

        result += term;
    }

    return result;
}

int main()
{
    iostream inFile;
    inFile.open("datos.txt");
    if (!inFile) {
    exit(1);
    }
    Container 
    while(!inFile.eof())
    {


    Data f[] = {{0,2}, {1,3}, {2,12}, {5,147}};
    return 0;
}

