#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include "structures.cuh"
#include "figures.cuh"
#include "functions.cuh"

#include <omp.h>

#define _USE_MATH_DEFINES
#include <math.h>

class Scene {
public:
    Scene(int width, int height)
        : width(width), height(height) {}

    void add_figure(const Figure& figure) {
        const std::vector<Triangle>& figure_trigs = figure.get_triangles();
        trigs.insert(trigs.end(), figure_trigs.begin(), figure_trigs.end());

        const std::vector<Light>& figure_lights = figure.get_lights();
        lights.insert(lights.end(), figure_lights.begin(), figure_lights.end());

        const std::vector<Diod>& figure_diods = figure.get_diods();
        diods.insert(diods.end(), figure_diods.begin(), figure_diods.end());
    }

    void add_light(const Light& source) {
        lights.push_back(source);
    }

    void add_floor(const float_3& A, const float_3& B, const float_3& C, const float_3& D,
        const Material& floor_material, const std::string& floor_path) {
        trigs.push_back(Triangle(A, D, B, floor_material, true)); 
        trigs.push_back(Triangle(C, D, B, floor_material, true)); 

        FILE* fp = fopen(floor_path.c_str(), "rb");
        fread(&floor_width, sizeof(int), 1, fp);
        fread(&floor_height, sizeof(int), 1, fp);
        uchar_4* floor_data = new uchar_4[floor_width * floor_height];
        floor.resize(floor_width * floor_height);
        fread(floor_data, sizeof(uchar_4), floor_width * floor_height, fp);
        fclose(fp);

        for (size_t i = 0; i < floor.size(); ++i) {
            floor[i] = uchar4_to_float3(floor_data[i]);
        }
        delete[] floor_data;
    }

    std::vector<uchar_4> render(const float_3& camera_pos, const float_3& camera_dir, float view_angle,
        int max_depth = 1, int sqrt_ray_per_pixel = 1,
        bool is_gpu = true, bool write = true) {

        int scaled_width = width * sqrt_ray_per_pixel;
        int scaled_height = height * sqrt_ray_per_pixel;

        int rays_num = scaled_width * scaled_height;
        int rays_capacity = 2 * rays_num;

        std::vector<Ray> render_rays(rays_capacity);
        std::vector<float_3> render_image(rays_num, float_3(0.0f, 0.0f, 0.0f));

        float_3 bz = norm(camera_dir - camera_pos);
        float_3 bx = norm(prod(bz, float_3(0.0f, 0.0f, 1.0f)));
        float_3 by = norm(prod(bx, bz));

#pragma omp parallel 
        {
            init_rays(render_rays,
                camera_pos,
                scaled_width,
                scaled_height,
                view_angle,
                bx, by, bz,
                omp_get_thread_num(),
                omp_get_num_threads());
        }

        if (is_gpu) {
            if (write)
                std::cout << "GPU\n";
            return gpu_render(render_rays, render_image,
                max_depth, scaled_width, scaled_height,
                rays_num, rays_capacity,
                sqrt_ray_per_pixel,
                write);

        }
        else {
            if (write)
                std::cout << "CPU\n";
            return cpu_render(render_rays, render_image,
                max_depth, scaled_width, scaled_height,
                rays_num, rays_capacity,
                sqrt_ray_per_pixel,
                write);
        }
    }

    void initialize_gpu_data() {

        CSC(cudaMalloc(&dev_trigs, sizeof(Triangle) * trigs.size()));
        CSC(cudaMalloc(&dev_lights, sizeof(Light) * lights.size()));
        CSC(cudaMalloc(&dev_diods, sizeof(Diod) * diods.size()));
        CSC(cudaMalloc(&dev_floor, sizeof(float_3) * floor.size()));

        CSC(cudaMemcpy(dev_trigs, trigs.data(), sizeof(Triangle) * trigs.size(), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_lights, lights.data(), sizeof(Light) * lights.size(), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_diods, diods.data(), sizeof(Diod) * diods.size(), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_floor, floor.data(), sizeof(float_3) * floor.size(), cudaMemcpyHostToDevice));
    }

    void clear_gpu_data() {

        CSC(cudaFree(dev_trigs));
        CSC(cudaFree(dev_lights));
        CSC(cudaFree(dev_diods));
        CSC(cudaFree(dev_floor));
    }

private:

    int cpu_clean_rays(std::vector<Ray>& rays, int rays_num) {
        std::sort(rays.rbegin(), rays.rend());

        for (int i = 0; i < rays_num; ++i) {
            if (rays[i].power < min_power) {
                return i;
            }
        }
        return rays_num;
    }

    int gpu_clean_rays(Ray* rays, int rays_num) {

        int* bin, * scan_data;
        Ray* rays_copy;

        CSC(cudaMalloc(&bin, sizeof(int) * rays_num));
        CSC(cudaMalloc(&scan_data, sizeof(int) * rays_num));
        CSC(cudaMalloc(&rays_copy, sizeof(Ray) * rays_num));

        calc_bin << <256, 256 >> > (rays, rays_num, bin, min_power);
        CSC(cudaGetLastError());

        CSC(cudaMemcpy(scan_data, bin, sizeof(int) * rays_num, cudaMemcpyDeviceToDevice));
        scan(scan_data, rays_num);

        int scan_last, bin_last;
        CSC(cudaMemcpy(&scan_last, &scan_data[rays_num - 1], sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaMemcpy(&bin_last, &bin[rays_num - 1], sizeof(int), cudaMemcpyDeviceToHost));
        int num_of_zeros = rays_num - scan_last;
        if (bin_last == 1)
            --num_of_zeros;

        CSC(cudaMemcpy(rays_copy, rays, sizeof(Ray) * rays_num, cudaMemcpyDeviceToDevice));

        sort_rays << <256, 256 >> > (rays, rays_copy, rays_num, num_of_zeros, bin, scan_data);
        CSC(cudaGetLastError());

        CSC(cudaFree(bin));
        CSC(cudaFree(scan_data));
        CSC(cudaFree(rays_copy));

        return num_of_zeros;
    }

    std::vector<uchar_4> gpu_render(std::vector<Ray>& render_rays, std::vector<float_3>& render_image,
        int max_depth, int scaled_width, int scaled_height,
        int rays_num, int rays_capacity,
        int sqrt_ray_per_pixel,
        bool write) {

        Ray* dev_render_rays;
        float_3* dev_render_image;

        CSC(cudaMalloc(&dev_render_rays, sizeof(Ray) * render_rays.size()));
        CSC(cudaMalloc(&dev_render_image, sizeof(float_3) * render_image.size()));

        CSC(cudaMemcpy(dev_render_rays, render_rays.data(), sizeof(Ray) * render_rays.size(), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_render_image, render_image.data(), sizeof(float_3) * render_image.size(), cudaMemcpyHostToDevice));

        int max_capacity = rays_capacity;

        for (int cur_depth = 1; cur_depth <= max_depth; ++cur_depth) {
            if (write)
                std::cout << "\tRecursion depth: " << cur_depth
                << "\tNumber of rays: " << rays_num << std::endl;

            start_gpu_ray_trace <<<256, 256>>> (dev_render_rays, rays_num,
                dev_trigs, trigs.size(),
                dev_lights, lights.size(),
                dev_diods, diods.size(),
                dev_floor,
                dev_render_image,
                scaled_width, scaled_height,
                floor_width, floor_height);
            CSC(cudaGetLastError());

            rays_num = gpu_clean_rays(dev_render_rays, rays_capacity);

            if (!rays_num)
                break;
            rays_capacity = 2 * rays_num;

            if (max_capacity < rays_capacity) {
                Ray* new_dev_rays;
                CSC(cudaMalloc(&new_dev_rays, sizeof(Ray) * rays_capacity));
                CSC(cudaMemcpy(new_dev_rays, dev_render_rays, sizeof(Ray) * rays_num, cudaMemcpyDeviceToDevice));
                CSC(cudaFree(dev_render_rays));
                dev_render_rays = new_dev_rays;
                max_capacity = rays_capacity;
            }
        }

        uchar_4* dev_result_image;
        CSC(cudaMalloc(&dev_result_image, sizeof(uchar_4) * width * height));
        gpu_ssaa << <dim3(32, 32), dim3(32, 32) >> > (dev_render_image, dev_result_image, width, height, sqrt_ray_per_pixel);
        CSC(cudaGetLastError());

        std::vector<uchar_4> result_image(width * height);
        CSC(cudaMemcpy(result_image.data(), dev_result_image, sizeof(uchar_4) * width * height, cudaMemcpyDeviceToHost));

        CSC(cudaFree(dev_render_rays));
        CSC(cudaFree(dev_render_image));
        CSC(cudaFree(dev_result_image));

        return result_image;
    }

    std::vector<uchar_4> cpu_render(std::vector<Ray>& render_rays, std::vector<float_3>& render_image,
        int max_depth, int scaled_width, int scaled_height,
        int rays_num, int rays_capacity,
        int sqrt_ray_per_pixel,
        bool write = true) {

        for (int cur_depth = 1; cur_depth <= max_depth; ++cur_depth) {
            if (write)
                std::cout << "\tRecursion depth: " << cur_depth
                << "\tNumber of rays: " << rays_num << std::endl;

            #pragma omp parallel 
            {
                ray_trace(render_rays.data(), rays_num,
                    trigs.data(), trigs.size(),
                    lights.data(), lights.size(),
                    diods.data(), diods.size(),
                    floor.data(),
                    render_image.data(),
                    scaled_width, scaled_height,
                    floor_width, floor_height,
                    omp_get_thread_num(),
                    omp_get_num_threads());
            }

            rays_num = cpu_clean_rays(render_rays, rays_capacity);
            if (!rays_num)
                break;
            rays_capacity = 2 * rays_num;

            if (int(render_rays.size()) < rays_capacity) {
                render_rays.resize(rays_capacity);
            }
        }

        std::vector<uchar_4> result_image(width * height);
        cpu_ssaa(render_image.data(), result_image.data(), width, height, sqrt_ray_per_pixel);

        return result_image;
    }

    void init_rays(std::vector<Ray>& rays,
        const float_3& camera_pos, int scaled_width, int scaled_height, float view_angle,
        const float_3& bx, const float_3& by, const float_3& bz,
        int start = 0, int step = 1) {

        float dw = 2.0f / (scaled_width - 1.0f);
        float dh = 2.0f / (scaled_height - 1.0f);
        float z = 1.0f / tan(view_angle * M_PI / 360.0f);

        int rays_num = scaled_width * scaled_height;
        for (int idx = start; idx < rays_num; idx += step) {
            int i = idx % scaled_width;
            int j = idx / scaled_width;

            float_3 v(-1.0f + dw * i, (-1.0f + dh * j) * float(scaled_height) / scaled_width, z);
            float_3 dir = norm(mult(bx, by, bz, v));

            rays[idx] = Ray(camera_pos, dir, i, j);
        }
    }

    int width;
    int height;

    int floor_width;
    int floor_height;

    std::vector<Triangle> trigs;
    std::vector<Light> lights;
    std::vector<Diod> diods;
    std::vector<float_3> floor;

    Triangle* dev_trigs;
    Light* dev_lights;
    Diod* dev_diods;
    float_3* dev_floor;

    const float min_power = 0.00001f;
};
