#pragma once

#define EPS 1e-5

#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

#define _i(_index) ((_index) + ((_index) >> 5))

#include "structures.cuh"
#include "scene.cuh"

#define _USE_MATH_DEFINES
#include <math.h>

struct Intersection {
    float distance;
    int trig_id;
};

__host__ __device__
float_3 to_color_range(const float_3& pixel);

__host__ __device__
uchar_4 float3_to_uchar4(const float_3& val);

__host__ __device__
float_3 uchar4_to_float3(const uchar_4& val);

__global__
void gpu_ssaa(float_3* render_image, uchar_4* result_image, int width, int height, int sqrt_ray_per_pixel);

void cpu_ssaa(float_3* render_image, uchar_4* result_image, int width, int height, int sqrt_ray_per_pixel);

__host__ __device__
float_3 get_color(const Ray& ray, 
                  const float_3& dir_to_light,
                  const Light& light,
                  const Triangle& triangle,
                  const float_3& trig_normal,
                  float_3* floor,
                  int floor_width, 
                  int floor_height);

__host__ __device__
float_3 floor_color(const Triangle& triangle, const Ray& ray, float_3* floor, int floor_width, int floor_height);

__host__ __device__
float dist(const float_3& lhs, const float_3& rhs);

__host__ __device__
float_3 get_symmetric_dir(const float_3& ray_dir, const float_3& trig_normal);

__host__ __device__
float_3 normal(const Triangle& triangle);

__host__ __device__
void find_triangle_intersection(Triangle* trigs, int trigs_num, const Ray& ray, Intersection& result, int cur_trig_id = -1);

__host__ __device__
float triangle_intersection(const Ray& ray, const Triangle& triangle);

__global__
void kernel_scan(int* data, int* shift, int n);

__global__
void kernel_shift(int* data, int* shift, int n);

void scan(int* dev_data, int n);

__global__
void calc_bin(Ray* rays, int rays_num, int* bin, float min_power);

__global__
void sort_rays(Ray* rays, Ray* rays_copy, int rays_num, int num_of_zeros, int* bin, int* scan_data);

__host__ __device__
void ray_trace(Ray* render_rays, int rays_num,
               Triangle* trigs, int trigs_num,
               Light* lights, int lights_num,
               Diod* diods, int diods_num,
               float_3* floor,
               float_3* render_image,
               int scaled_width, 
               int scaled_height,
               int floor_width, 
               int floor_height,
               int start, 
               int step);

__global__
void start_gpu_ray_trace(Ray* render_rays, 
                         int rays_num,
                         Triangle* trigs, 
                         int trigs_num,
                         Light* lights, 
                         int lights_num,
                         Diod* diods, 
                         int diods_num,
                         float_3* floor,
                         float_3* render_image,
                         int scaled_width, 
                         int scaled_height,
                         int floor_width, 
                         int floor_height);

                         
__host__ __device__
float_3 to_color_range(const float_3& pixel) {
    return float_3(min(pixel.x, 1.0f), min(pixel.y, 1.0f), min(pixel.z, 1.0f));
}

__host__ __device__
uchar_4 float3_to_uchar4(const float_3& val) {
    uchar_4 result;

    result.x = round(255.0f * val.x);
    result.y = round(255.0f * val.y);
    result.z = round(255.0f * val.z);
    result.w = 0;

    return result;
}

__host__ __device__
float_3 uchar4_to_float3(const uchar_4& val) {
    float_3 result;

    result.x = val.x / 255.0f;
    result.y = val.z / 255.0f;
    result.z = val.z / 255.0f;

    return result;
}

__global__
void gpu_ssaa(float_3* render_image, uchar_4* result_image, int width, int height, int sqrt_ray_per_pixel) {
    int sc_width = width * sqrt_ray_per_pixel;

    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int id_y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int x = id_x; x < width; x += offset_x) {
        for (int y = id_y; y < height; y += offset_y) {
            float_3 sum(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < sqrt_ray_per_pixel; ++i) {
                for (int j = 0; j < sqrt_ray_per_pixel; ++j) {
                    sum += render_image[y * sc_width * sqrt_ray_per_pixel + sc_width * j + x * sqrt_ray_per_pixel + i];
                }
            }
            sum /= float(sqrt_ray_per_pixel * sqrt_ray_per_pixel);
            result_image[y * width + x] = float3_to_uchar4(to_color_range(sum));
        }
    }
}

void cpu_ssaa(float_3* render_image, uchar_4* result_image, int width, int height, int sqrt_ray_per_pixel) {
    int sc_width = width * sqrt_ray_per_pixel;

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            float_3 sum(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < sqrt_ray_per_pixel; ++i) {
                for (int j = 0; j < sqrt_ray_per_pixel; ++j) {
                    sum += render_image[y * sc_width * sqrt_ray_per_pixel + sc_width * j + x * sqrt_ray_per_pixel + i];
                }
            }
            sum /= float(sqrt_ray_per_pixel * sqrt_ray_per_pixel);
            result_image[y * width + x] = float3_to_uchar4(to_color_range(sum));
        }
    }
}

__host__ __device__
float_3 get_color(const Ray& ray, 
                 const float_3& dir_to_light,
                 const Light& light,
                 const Triangle& triangle,
                 const float_3& trig_normal,
                 float_3* floor,
                 int floor_width, 
                 int floor_height) {

    float diffuse_component = triangle.material.diffussion * (dot(trig_normal, dir_to_light));

    float_3 refl_ray_dir = get_symmetric_dir(dir_to_light, trig_normal);
    float specular_component = triangle.material.reflection * powf(dot(refl_ray_dir, -ray.dir), triangle.material.specular_pow);
    if (specular_component < 0.0f)
        specular_component = 0.0f;

    float_3 color = triangle.material.color * light.color;
    if (triangle.is_floor) {
        color *= floor_color(triangle, ray, floor, floor_width, floor_height);
    }

    return color * light.power * (diffuse_component + specular_component);
}

__host__ __device__
float_3 floor_color(const Triangle& triangle, const Ray& ray, float_3* floor, int floor_width, int floor_height) {
    float_3 e1 = triangle.b - triangle.a;
    float_3 e2 = triangle.c - triangle.a;

    float_3 p = prod(ray.dir, e2);
    float div = dot(p, e1);

    float_3 t = ray.from - triangle.a;
    float_3 q = prod(t, e1);

    float u = dot(p, t) / div;
    float v = dot(q, ray.dir) / div;

    int scaled_u = floor_width * u;
    int scaled_v = floor_height * v;

    return floor[scaled_u * floor_width + scaled_v];
}

__host__ __device__
float dist(const float_3& lhs, const float_3& rhs) {
    return sqrt((rhs.x - lhs.x) * (rhs.x - lhs.x) +
                (rhs.y - lhs.y) * (rhs.y - lhs.y) +
                (rhs.z - lhs.z) * (rhs.z - lhs.z));
}

__host__ __device__
float_3 get_symmetric_dir(const float_3& ray_dir, const float_3& trig_normal) {
    return norm(trig_normal * (2.0f * dot(trig_normal, ray_dir)) - ray_dir);
}

__host__ __device__
float_3 normal(const Triangle& triangle) {
    return norm(prod(triangle.c - triangle.a, triangle.b - triangle.a));
}

__host__ __device__
void find_triangle_intersection(Triangle* trigs, int trigs_num, const Ray& ray, Intersection& result, int cur_trig_id) {
    float ts;
    result.distance = -1.0f;
    result.trig_id = -1;

    for (int i = 0; i < trigs_num && i != cur_trig_id; ++i) {
        ts = triangle_intersection(ray, trigs[i]);

        if (ts >= EPS && (ts < result.distance || result.trig_id == -1)) {
            result.distance = ts;
            result.trig_id = i;
        }
    }
}

__host__ __device__
float triangle_intersection(const Ray& ray, const Triangle& triangle) {
    float_3 e1 = triangle.b - triangle.a;
    float_3 e2 = triangle.c - triangle.a;
    float_3 p = prod(ray.dir, e2);
    float div = dot(p, e1);

    if (fabs(div) < EPS)
        return -1.0f;
    float_3 t = ray.from - triangle.a;
    float u = dot(p, t) / div;
    if (u < 0.0f || u > 1.0f)
        return -1.0f;
    float_3 q = prod(t, e1);
    float v = dot(q, ray.dir) / div;
    if (v < 0.0f || v + u > 1.0f)
        return -1.0f;
    return dot(q, e2) / div;
}

__host__ __device__
void ray_trace(Ray* render_rays, 
               int rays_num,
               Triangle* trigs, 
               int trigs_num,
               Light* lights, 
               int lights_num,
               Diod* diods, 
               int diods_num,
               float_3* floor,
               float_3* render_image,
               int scaled_width, 
               int scaled_height,
               int floor_width, 
               int floor_height,
               int start, 
               int step) {

    for (int ray_id = start; ray_id < rays_num; ray_id += step) {
        Intersection intersection_info, ray_to_light_intersection;
        find_triangle_intersection(trigs, trigs_num, render_rays[ray_id], intersection_info);

        if (intersection_info.trig_id < 0) {
            render_rays[ray_id].power = 0.0f;
            render_rays[rays_num + ray_id].power = 0.0f;
            continue;
        }

        Ray cur_ray = render_rays[ray_id];
        Triangle cur_trig = trigs[intersection_info.trig_id];
        bool is_diod = false;

        float_3 intersection_pos = cur_ray.from + cur_ray.dir * intersection_info.distance;
        float_3 result_color(0.0f, 0.0f, 0.0f);

        if (cur_trig.is_internal_edge) {
            for (int i = 0; i < diods_num; ++i) {
                if (dist(intersection_pos, diods[i].point) <= diods[i].radius) {
                    is_diod = true;
                    break;
                }
            }
        }

        float_3 trig_normal = normal(cur_trig);
        if (dot(cur_ray.dir, trig_normal) > 0.0f)
            trig_normal *= -1;

        if (is_diod) {
            result_color = float_3(1.0f, 1.0f, 1.0f);
        }
        else {
            for (int light_id = 0; light_id < lights_num; ++light_id) {
                float_3 dir_to_light = norm(lights[light_id].coord - intersection_pos);
                Ray ray_to_light(intersection_pos, dir_to_light);

                if (dot(dir_to_light, trig_normal) < 0.0f)
                    continue;

                find_triangle_intersection(trigs, trigs_num, ray_to_light, ray_to_light_intersection, intersection_info.trig_id);
                if (ray_to_light_intersection.trig_id >= 0)
                    continue;

                result_color += get_color(cur_ray,
                    dir_to_light, lights[light_id],
                    cur_trig, trig_normal,
                    floor,
                    floor_width, floor_height);
            }
        }

        result_color *= cur_ray.power;

        render_image[(scaled_height - 1 - cur_ray.pixel_id_y) * scaled_width + cur_ray.pixel_id_x] += result_color;

        Ray reflected_ray(intersection_pos, get_symmetric_dir(-cur_ray.dir, trig_normal), cur_ray.pixel_id_x, cur_ray.pixel_id_y);
        reflected_ray.power = cur_ray.power * cur_trig.material.reflection;

        Ray refracted_ray(intersection_pos, cur_ray.dir, cur_ray.pixel_id_x, cur_ray.pixel_id_y);
        refracted_ray.power = cur_ray.power * cur_trig.material.refraction;

        render_rays[ray_id] = reflected_ray;
        render_rays[rays_num + ray_id] = refracted_ray;
    }
}

__global__
void start_gpu_ray_trace(Ray* render_rays, 
                         int rays_num,
                         Triangle* trigs, 
                         int trigs_num,
                         Light* lights, 
                         int lights_num,
                         Diod* diods, 
                         int diods_num,
                         float_3* floor,
                         float_3* render_image,
                         int scaled_width, 
                         int scaled_height,
                         int floor_width, 
                         int floor_height) {

    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = blockDim.x * gridDim.x;

    ray_trace(render_rays, 
              rays_num,
              trigs, trigs_num,
              lights, lights_num,
              diods, diods_num,
              floor, render_image,
              scaled_width, scaled_height,
              floor_width, floor_height,
              id_x, offset_x);
}

__global__
void kernel_scan(int* data, int* shift, int n) {
    __shared__ int s_data[BLOCK_SIZE + 8];
    int idx = threadIdx.x;
    int global_offset = (gridDim.x * blockIdx.y + blockIdx.x) * BLOCK_SIZE;
    int offset, s, index, tmp;

    s_data[_i(idx)] = ((global_offset + idx) < n) ? data[global_offset + idx] : 0;
    s_data[_i(idx + BLOCK_SIZE / 2)] = ((global_offset + idx + BLOCK_SIZE / 2) < n) ? data[global_offset + idx + BLOCK_SIZE / 2] : 0;

    for (s = 1; s <= BLOCK_SIZE / 2; s <<= 1) {
        __syncthreads();
        offset = s - 1;
        index = 2 * s * idx;
        if (index < BLOCK_SIZE)
            s_data[_i(offset + index + s)] += s_data[_i(offset + index)];
    }
    if (idx == 0) {
        shift[gridDim.x * blockIdx.y + blockIdx.x] = s_data[_i(BLOCK_SIZE - 1)];
        s_data[_i(BLOCK_SIZE - 1)] = 0;
    }
    for (s = BLOCK_SIZE / 2; s >= 1; s >>= 1) {
        __syncthreads();
        offset = s - 1;
        index = 2 * s * idx;
        if (index < BLOCK_SIZE) {
            tmp = s_data[_i(offset + index + s)];
            s_data[_i(offset + index + s)] += s_data[_i(offset + index)];
            s_data[_i(offset + index)] = tmp;
        }
    }
    __syncthreads();

    if ((global_offset + idx) < n)
        data[global_offset + idx] = s_data[_i(idx)];
    if ((global_offset + idx + BLOCK_SIZE / 2) < n)
        data[global_offset + idx + BLOCK_SIZE / 2] = s_data[_i(idx + BLOCK_SIZE / 2)];
}

__global__
void kernel_shift(int* data, int* shift, int n) {
    int idx = threadIdx.x;
    int offset = gridDim.x * blockIdx.y + blockIdx.x;
    int diff = shift[offset];
    offset *= BLOCK_SIZE;
    if (offset + idx < n)
        data[offset + idx] += diff;
    if (offset + idx + BLOCK_SIZE / 2 < n)
        data[offset + idx + BLOCK_SIZE / 2] += diff;
}

void scan(int* dev_data, int n) {
    int num_of_blocks = (n - 1) / BLOCK_SIZE + 1;
    dim3 blocks(std::min(num_of_blocks, MAX_BLOCKS), (num_of_blocks - 1) / MAX_BLOCKS + 1);
    dim3 threads(BLOCK_SIZE / 2);
    int* dev_shift;
    CSC(cudaMalloc(&dev_shift, num_of_blocks * sizeof(int)));

    kernel_scan << <blocks, threads >> > (dev_data, dev_shift, n);
    CSC(cudaGetLastError());
    if (num_of_blocks == 1) {
        CSC(cudaFree(dev_shift));
        return;
    }
    scan(dev_shift, num_of_blocks);
    kernel_shift << <blocks, threads >> > (dev_data, dev_shift, n);
    CSC(cudaGetLastError());
    CSC(cudaFree(dev_shift));
}

__global__
void calc_bin(Ray* rays, int rays_num, int* bin, float min_power) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = blockDim.x * gridDim.x;

    for (int i = id_x; i < rays_num; i += offset_x) {
        if (rays[i].power < min_power)
            bin[i] = 1;
        else
            bin[i] = 0;
    }
}

__global__
void sort_rays(Ray* rays, Ray* rays_copy, int rays_num, int num_of_zeros, int* bin, int* scan_data) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = blockDim.x * gridDim.x;

    for (int i = id_x; i < rays_num; i += offset_x) {
        if (bin[i] == 0)
            rays[i - scan_data[i]] = rays_copy[i];
        else
            rays[num_of_zeros + scan_data[i]] = rays_copy[i];
    }
}
