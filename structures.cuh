#pragma once

#include <cmath>

typedef unsigned char uchar;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct uchar_4 {
    __host__ __device__ uchar_4() = default;
    __host__ __device__ uchar_4(uchar x, uchar y, uchar z, uchar w)
        : x(x), y(y), z(z), w(w) {}

    uchar x;
    uchar y;
    uchar z;
    uchar w;
};

struct float_3 {
    __host__ __device__ float_3() = default;
    __host__ __device__ float_3(float x, float y, float z)
        : x(x), y(y), z(z) {}

    float x;
    float y;
    float z;
};

__host__ __device__
float dot(const float_3& a, const float_3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
float_3 norm(const float_3& v) {
    float l = sqrt(dot(v, v));
    return float_3(v.x / l, v.y / l, v.z / l);
}

__host__ __device__
float_3 prod(const float_3& a, const float_3& b) {
    return float_3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__
float_3 mult(const float_3& a, const float_3& b, const float_3& c, const float_3& v) {
    return float_3(a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z);
}

__host__ __device__
float_3 operator+ (const float_3& lhs, const float_3& rhs) {
    return float_3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__host__ __device__
float_3& operator+= (float_3& lhs, const float_3& rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

__host__ __device__
float_3 operator- (const float_3& lhs) {
    return float_3(-lhs.x, -lhs.y, -lhs.z);
}

__host__ __device__
float_3 operator- (const float_3& lhs, const float_3& rhs) {
    return float_3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__
float_3& operator-= (float_3& lhs, const float_3& rhs) {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

__host__ __device__
float_3 operator* (const float_3& lhs, float rhs) {
    return float_3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

__host__ __device__
float_3 operator* (const float_3& lhs, const float_3& rhs) {
    return float_3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

__host__ __device__
float_3& operator*= (float_3& lhs, float rhs) {
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    return lhs;
}

__host__ __device__
float_3& operator*= (float_3& lhs, const float_3& rhs) {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}

__host__ __device__
float_3 operator/ (const float_3& lhs, float rhs) {
    return float_3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

__host__ __device__
float_3 operator/ (const float_3& lhs, float_3& rhs) {
    return float_3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

__host__ __device__
float_3& operator/= (float_3& lhs, float rhs) {
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    return lhs;
}

__host__ __device__
float_3& operator/= (float_3& lhs, float_3& rhs) {
    lhs.x /= rhs.x;
    lhs.y /= rhs.y;
    lhs.z /= rhs.z;
    return lhs;
}

struct Material {
    __host__ __device__ Material() = default;
    __host__ __device__ Material(const float_3& color, 
                                 float reflection, 
                                 float refraction, 
                                 float diffussion = 0.5f, 
                                 int specular_pow = 64)
        : color(color), 
          reflection(reflection), 
          refraction(refraction), 
          diffussion(diffussion), 
          specular_pow(specular_pow) {}

    float_3 color;
    float reflection;
    float refraction;
    float diffussion;
    int specular_pow;
};

struct Triangle {
    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const float_3& a, 
                                 const float_3& b, 
                                 const float_3& c, 
                                 const Material& material, 
                                 bool is_floor = false, 
                                 bool is_internal_edge = false)
        : a(a), 
          b(b), 
          c(c), 
          material(material), 
          is_floor(is_floor), 
          is_internal_edge(is_internal_edge) {}

    float_3 a;
    float_3 b;
    float_3 c;
    Material material;
    bool is_floor;
    bool is_internal_edge;
};

struct Ray {
    __host__ __device__ Ray() = default;
    __host__ __device__ Ray(const float_3& from, 
                            const float_3& dir, 
                            int pixel_id_x = 0, 
                            int pixel_id_y = 0, 
                            float power = 1.0f)
        : from(from), 
          dir(norm(dir)), 
          pixel_id_x(pixel_id_x), 
          pixel_id_y(pixel_id_y), power(power) {}

    float_3 from;
    float_3 dir;
    int pixel_id_x;
    int pixel_id_y;
    float power;
};

__host__ __device__
bool operator< (const Ray& lhs, const Ray& rhs) {
    return lhs.power < rhs.power;
}

struct Light {
    Light(const float_3& coord, const float_3& color, float power = 1.0f)
        : coord(coord), color(color), power(power) {}

    float_3 coord;
    float_3 color;
    float power;
};

struct Line {
    Line() = default;
    Line(int point_id_1, int point_id_2)
        : point_id_1(point_id_1), point_id_2(point_id_2) {}

    int point_id_1;
    int point_id_2;
};

struct Edge {
    Edge() = default;
    Edge(const Line& line, const std::pair<int, int>& f1, const std::pair<int, int>& f2)
        : line(line), f1(f1), f2(f2) {}

    Line line;
    std::pair<int, int> f1;
    std::pair<int, int> f2;
};

struct Diod {
    Diod() = default;
    Diod(const float_3& point, const Material& material, float radius)
        : point(point), material(material), radius(radius) {}

    float_3 point;
    Material material;
    float radius;
};
