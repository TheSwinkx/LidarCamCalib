#ifndef BOXSIZE_H
#define BOXSIZE_H

struct BoxSize {
    BoxSize(double x, double y, double z) : _x(x), _y(y), _z(z) {}

    double _x, _y, _z;
};

#endif