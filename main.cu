#include <iostream>
#include <string>
#include <chrono>

#include "structures.cuh"
#include "scene.cuh"
#include "figures.cuh"

#include "mpi.h"

#define _USE_MATH_DEFINES
#include <math.h>

float_3 cilindric_to_decart(float r, float f, float z) {
    return float_3(r * cos(f), r * sin(f), z);
}

void print_default_settings() {
    std::cout << "512" << std::endl;
    std::cout << "images/%d.data" << std::endl;
    std::cout << "800 600 120" << std::endl;
    std::cout << "4.5 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0" << std::endl;
    std::cout << "1.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0" << std::endl;
    std::cout << "0.0 3.5 -1.1 0.6 0.0 0.0 1.2 0.4 0.1 2" << std::endl;
    std::cout << "2.5 0.0 0.6 0.2 0.6 0.0 1.6 0.8 0.3 7" << std::endl;
    std::cout << "-2.5 0.0 0.5 0.0 0.7 0.7 1.7 0.7 0.5 5" << std::endl;
    std::cout << "-5.0 -5.0 -2.0 -5.0 5.0 -2.0 5.0 5.0 -2.0 5.0 -5.0 -2.0" << std::endl;
    std::cout << "textures/floor.data 1.0 0.0 1.0 0.5" << std::endl;
    std::cout << "8" << std::endl;
    std::cout << "-10.0 0.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "10.0 0.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "10.0 10.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "10.0 -10.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "0.0 -10.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "0.0 10.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "10.0 0.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "-10.0 0.0 50.0 0.2 0.2 0.2" << std::endl;
    std::cout << "50 4" << std::endl;
}

int main(int argc, char** argv) {

    bool is_gpu = true;

    if (argc == 2) {
        if (!strcmp(argv[1], "--gpu"))
            is_gpu = true;
        else if (!strcmp(argv[1], "--cpu"))
            is_gpu = false;
        else if (!strcmp(argv[1], "--default")) {
            print_default_settings();
            return 0;
        }
    }

    int frames, width, height, source_num, source_per_line, recursion_depth, sqrt_ray_per_pixel;
    char image_path[256];
    float view_angle;
    float rc_0, zc_0, fc_0, Ac_r, Ac_z, wc_r, wc_z, wc_f, pc_r, pc_z;
    float rn_0, zn_0, fn_0, An_r, An_z, wn_r, wn_z, wn_f, pn_r, pn_z;
    float_3 center, color;
    float radius, reflection, refraction;

    float_3 floor_A, floor_B, floor_C, floor_D;
    char floor_path[256];
    float_3 floor_color;
    float floor_reflection;

    bool write = false;

    float_3 source_coord, source_color;

    int id_proc, num_of_procs, device_count;

    cudaGetDeviceCount(&device_count);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs);
    MPI_Barrier(MPI_COMM_WORLD);

    cudaSetDevice(id_proc % device_count);

    if (id_proc == 0) {
        write = true;

        std::cin >> frames;
        std::cin >> image_path;
        std::cin >> width >> height;
        std::cin >> view_angle;
        std::cin >> rc_0 >> zc_0 >> fc_0 >> Ac_r >> Ac_z >> wc_r >> wc_z >> wc_f >> pc_r >> pc_z
                 >> rn_0 >> zn_0 >> fn_0 >> An_r >> An_z >> wn_r >> wn_z >> wn_f >> pn_r >> pn_z;

    }

    MPI_Bcast(&frames, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image_path, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&view_angle, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&rc_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zc_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fc_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ac_r, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ac_z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wc_r, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wc_z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wc_f, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pc_r, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pc_z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&rn_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zn_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fn_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&An_r, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&An_z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wn_r, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wn_z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wn_f, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pn_r, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pn_z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    Scene scene(width, height);

    if (id_proc == 0) {
        std::cin >> center.x >> center.y >> center.z
                 >> color.x >> color.y >> color.z
                 >> radius >> reflection >> refraction
                 >> source_per_line;
    }

    MPI_Bcast(&center.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflection, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&refraction, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_per_line, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Material material_1(color, reflection, refraction);
    scene.add_figure(Tetraeder(center, radius, material_1, source_per_line));

    if (id_proc == 0) {
        std::cin >> center.x >> center.y >> center.z
                 >> color.x >> color.y >> color.z
                 >> radius >> reflection >> refraction
                 >> source_per_line;
    }

    MPI_Bcast(&center.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflection, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&refraction, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_per_line, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Material material_2(color, reflection, refraction);
    scene.add_figure(Dodecahedron(center, radius, material_2, source_per_line));

    if (id_proc == 0) {
        std::cin >> center.x >> center.y >> center.z
                 >> color.x >> color.y >> color.z
                 >> radius >> reflection >> refraction
                 >> source_per_line;
    }

    MPI_Bcast(&center.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflection, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&refraction, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_per_line, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Material material_3(color, reflection, refraction);
    scene.add_figure(Icosahedron(center, radius, material_3, source_per_line));

    if (id_proc == 0) {
        std::cin >> floor_A.x >> floor_A.y >> floor_A.z
                 >> floor_B.x >> floor_B.y >> floor_B.z
                 >> floor_C.x >> floor_C.y >> floor_C.z
                 >> floor_D.x >> floor_D.y >> floor_D.z
                 >> floor_path
                 >> floor_color.x >> floor_color.y >> floor_color.z
                 >> floor_reflection;
    }

    MPI_Bcast(&floor_A.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_A.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_A.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_B.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_B.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_B.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_C.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_C.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_C.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_D.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_D.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_D.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&floor_path, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Bcast(&floor_color.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_color.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_color.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&floor_reflection, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    Material floor_material(floor_color, floor_reflection, 0.0f);
    scene.add_floor(floor_A, floor_B, floor_C, floor_D, floor_material, floor_path);

    if (id_proc == 0) {
        std::cin >> source_num;
    }

    MPI_Bcast(&source_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < source_num; ++i) {
        if (id_proc == 0) {
            std::cin >> source_coord.x >> source_coord.y >> source_coord.z
                     >> source_color.x >> source_color.y >> source_color.z;
        }

        MPI_Bcast(&source_coord.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source_coord.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source_coord.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source_color.x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source_color.y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source_color.z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        scene.add_light(Light(source_coord, source_color));
    }

    if (id_proc == 0) {
        std::cin >> recursion_depth >> sqrt_ray_per_pixel;
    }

    MPI_Bcast(&recursion_depth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sqrt_ray_per_pixel, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (is_gpu)
        scene.initialize_gpu_data();

    char buff[512];
    float step = 2.0f * M_PI / frames;

    MPI_Barrier(MPI_COMM_WORLD);

    clock_t t_start_ = clock();

    for (int frame_id = id_proc + 1; frame_id <= frames; frame_id += num_of_procs) {
        auto start = std::chrono::high_resolution_clock::now();

        float t = step * (frame_id - 1);

        float rc = rc_0 + Ac_r * sin(wc_r * t + pc_r);
        float zc = zc_0 + Ac_z * sin(wc_z * t + pc_z);
        float fc = fc_0 + wc_f * t;

        float rn = rn_0 + An_r * sin(wn_r * t + pn_r);
        float zn = zn_0 + An_z * sin(wn_z * t + pn_z);
        float fn = fn_0 + wn_f * t;

        float_3 pc = cilindric_to_decart(rc, fc, zc);
        float_3 pv = cilindric_to_decart(rn, fn, zn);

        if (write)
            std::cout << "Frame number: " << frame_id << std::endl;

        std::vector<uchar_4> image = scene.render(pc, pv, view_angle, recursion_depth, sqrt_ray_per_pixel, is_gpu, write);

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

        if (write)
            std::cout << "Render time: " << seconds << " seconds" << std::endl;

        sprintf(buff, image_path, frame_id);
        FILE* out = fopen(buff, "wb");
        fwrite(&width, sizeof(int), 1, out);
        fwrite(&height, sizeof(int), 1, out);
        fwrite(image.data(), sizeof(uchar_4), width * height, out);
        fclose(out);

        if (write)
            std::cout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    clock_t t_end_ = clock();

    if (write)
        std::cout << "TIME: " << (double)(t_end_ - t_start_) / CLOCKS_PER_SEC << "\n";

    if (is_gpu)
        scene.clear_gpu_data();

    MPI_Finalize();

    return 0;
}