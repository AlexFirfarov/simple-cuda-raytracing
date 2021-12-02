#pragma once

#include <vector>
#include <cmath>

#include "structures.cuh"

class Figure {
public:
    Figure(const float_3& center, float radius, const Material& material, size_t source_per_line)
        : center(center), radius(radius), material(material), source_per_line(source_per_line) {}

    const std::vector<Triangle>& get_triangles() const {
        return trigs;
    }

    const std::vector<Light>& get_lights() const {
        return lights;
    }

    const std::vector<Diod>& get_diods() const {
        return diods;
    }

protected:
    virtual void generate_vertexes() = 0;
    virtual void scale_vertexes() = 0;
    virtual void generate_edges() = 0;
    virtual void generate_triangles() = 0;

    void add_edge(const Line& line, const std::pair<int, int>& f) {
        const float_3& point_1 = vertexes[line.point_id_1];
        const float_3& point_2 = vertexes[line.point_id_2];

        float_3 point1_dir = vertexes[f.first] - point_1;
        float_3 point2_dir = vertexes[f.second] - point_2;
        float_3 point1_p = point_1 + point1_dir * edge_koef;
        float_3 point2_p = point_2 + point2_dir * edge_koef;

        trigs.emplace_back(point_1 * 1.01f, point1_p * 1.01f, point_2 * 1.01f, edge_material);
        trigs.emplace_back(point_2 * 1.01f, point2_p * 1.01f, point1_p * 1.01f, edge_material);

        trigs.emplace_back(point_1 * 0.99f, point1_p * 0.99f, point_2 * 0.99f, edge_material, false, true);
        trigs.emplace_back(point_2 * 0.99f, point2_p * 0.99f, point1_p * 0.99f, edge_material, false, true);
    }

    void transform_triangles_to_center() {
        for (auto& triangle : trigs) {
            triangle.a += center;
            triangle.b += center;
            triangle.c += center;
        }
    }

    void generate_diods() {
        for (const auto& edge : edges) {
            float_3 point_1 = vertexes[edge.line.point_id_1] * 0.98f;
            float_3 point_2 = vertexes[edge.line.point_id_2] * 0.98f;

            float edge_length = dist(point_1, point_2);
            float step = edge_length / float(source_per_line + 1);
            float_3 dir = norm(point_2 - point_1);

            for (size_t i = 1; i <= source_per_line; ++i) {
                float_3 diod_point = point_1 + dir * step * i;
                float diod_radius = edge_length * diod_koef;
                diods.push_back(Diod(diod_point + center, diod_material, diod_radius));
            }
        }
    }

    void generate_lights() {
        lights.push_back(Light(center, float_3(1.0f, 1.0f, 1.0f)));
    }

    void initialize_figure() {
        generate_vertexes();
        scale_vertexes();
        generate_edges();
        generate_triangles();
        transform_triangles_to_center();
        generate_diods();
        generate_lights(); 
    }

    float dist(const float_3& lhs, const float_3& rhs) const {
        return sqrt((rhs.x - lhs.x) * (rhs.x - lhs.x) +
                    (rhs.y - lhs.y) * (rhs.y - lhs.y) +
                    (rhs.z - lhs.z) * (rhs.z - lhs.z));
    }

    float_3 center;
    float radius;
    size_t source_per_line;
    Material material;

    const float edge_koef = 0.05f;
    const float diod_koef = 0.022f;

    std::vector<float_3> vertexes;
    std::vector<Triangle> trigs;
    std::vector<Light> lights;
    std::vector<Edge> edges;
    std::vector<Diod> diods;

    const Material edge_material = Material(float_3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, 0.0f);
    const Material diod_material = Material(float_3(1.0f, 1.0f, 1.0f), 0.0f, 0.0f, 1.0f);
};

class Tetraeder : public Figure {
public:
    Tetraeder(const float_3& center, float radius, const Material& material, size_t source_per_line)
        : Figure(center, radius, material, source_per_line) {
        initialize_figure();
    }

private:
    void generate_vertexes() override {
        vertexes.resize(4);

        vertexes[0] = float_3(-1.0f, -1.0f / sqrt(3.0f), -1.0f / sqrt(6.0f));
        vertexes[1] = float_3(1.0f, -1.0f / sqrt(3.0f), -1.0f / sqrt(6.0f));
        vertexes[2] = float_3(0.0f, 2.0f / sqrt(3.0f), -1.0f / sqrt(6.0f));
        vertexes[3] = float_3(0.0f, 0.0f, 3.0f / sqrt(6.0f));
    }

    void scale_vertexes() override {
        float edge_length = 4.0f * radius / sqrt(6.0f);
        for (auto& vertex : vertexes) {
            vertex *= 0.5f * edge_length;
        }
    }

    void generate_edges() override {
        edges.resize(6);

        edges[0] = Edge(Line(0, 2), { 3, 3 }, { 1, 1 });
        edges[1] = Edge(Line(2, 1), { 3, 3 }, { 0, 0 });
        edges[2] = Edge(Line(0, 1), { 3, 3 }, { 2, 2 });
        edges[3] = Edge(Line(0, 3), { 2, 2 }, { 1, 1 });
        edges[4] = Edge(Line(2, 3), { 0, 0 }, { 1, 1 });
        edges[5] = Edge(Line(1, 3), { 2, 2 }, { 0, 0 });
    }

    void generate_triangles() override {

        for (const auto& edge : edges) {
            add_edge(edge.line, edge.f1);
            add_edge(edge.line, edge.f2);
        }

        size_t st = trigs.size();
        trigs.resize(st + 4);

        trigs[st + 0] = Triangle(vertexes[0], vertexes[1], vertexes[2], material);
        trigs[st + 1] = Triangle(vertexes[0], vertexes[2], vertexes[3], material);
        trigs[st + 2] = Triangle(vertexes[1], vertexes[2], vertexes[3], material);
        trigs[st + 3] = Triangle(vertexes[0], vertexes[1], vertexes[3], material);
    }
};

class Dodecahedron : public Figure {
public:
    Dodecahedron(const float_3& center, float radius, const Material& material, size_t source_per_line)
        : Figure(center, radius, material, source_per_line) {
        initialize_figure();
    }

private:
    void generate_vertexes() override {
        vertexes.resize(20);

        vertexes[0] = float_3(1.0f, 1.0f, 1.0f);
        vertexes[1] = float_3(1.0f, 1.0f, -1.0f);
        vertexes[2] = float_3(1.0f, -1.0f, 1.0f);
        vertexes[3] = float_3(1.0f, -1.0f, -1.0f);
        vertexes[4] = float_3(-1.0f, 1.0f, 1.0f);
        vertexes[5] = float_3(-1.0f, 1.0f, -1.0f);
        vertexes[6] = float_3(-1.0f, -1.0f, 1.0f);
        vertexes[7] = float_3(-1.0f, -1.0f, -1.0f);

        vertexes[8] = float_3(0.0f, 1.0f / phi, phi);
        vertexes[9] = float_3(0.0f, 1.0f / phi, -phi);
        vertexes[10] = float_3(0.0f, -1.0f / phi, phi);
        vertexes[11] = float_3(0.0f, -1.0f / phi, -phi);

        vertexes[12] = float_3(phi, 0.0f, 1.0f / phi);
        vertexes[13] = float_3(phi, 0.0f, -1.0f / phi);
        vertexes[14] = float_3(-phi, 0.0f, 1.0f / phi);
        vertexes[15] = float_3(-phi, 0.0f, -1.0f / phi);

        vertexes[16] = float_3(1.0f / phi, phi, 0.0f);
        vertexes[17] = float_3(1.0f / phi, -phi, 0.0f);
        vertexes[18] = float_3(-1.0f / phi, phi, 0.0f);
        vertexes[19] = float_3(-1.0f / phi, -phi, 0.0f);
    }

    void scale_vertexes() override {
        float edge_length = 2.0f * radius / (sqrt(3.0f) * phi);
        for (auto& vertex : vertexes) {
            vertex *= 0.5f * phi * edge_length;
        }
    }

    void generate_edges() override {
        edges.resize(30);

        edges[0] = Edge(Line(10, 8), { 2, 0 }, { 6, 4 });
        edges[1] = Edge(Line(10, 2), { 8, 12 }, { 6, 17 });
        edges[2] = Edge(Line(2, 12), { 10, 0 }, { 17, 13 });
        edges[3] = Edge(Line(12, 0), { 2, 8 }, { 13, 16 });
        edges[4] = Edge(Line(0, 8), { 12, 10 }, { 16, 4 });
        edges[5] = Edge(Line(6, 10), { 14, 8 }, { 19, 2 });
        edges[6] = Edge(Line(2, 17), { 10, 19 }, { 12, 3 });
        edges[7] = Edge(Line(17, 19), { 2, 6 }, { 3, 7 });
        edges[8] = Edge(Line(19, 6), { 17, 10 }, { 7, 14 });
        edges[9] = Edge(Line(6, 14), { 10, 4 }, { 19, 15 });
        edges[10] = Edge(Line(14, 4), { 6, 8 }, { 15, 18 });
        edges[11] = Edge(Line(4, 8), { 14, 10 }, { 18, 0 });
        edges[12] = Edge(Line(12, 13), { 2, 3 }, { 0, 1 });
        edges[13] = Edge(Line(13, 3), { 12, 17 }, { 1, 11 });
        edges[14] = Edge(Line(3, 17), { 13, 2 }, { 11, 19 });
        edges[15] = Edge(Line(19, 7), { 17, 11 }, { 6, 15 });
        edges[16] = Edge(Line(7, 11), { 19, 3 }, { 15, 9 });
        edges[17] = Edge(Line(11, 3), { 7, 17 }, { 9, 13 });
        edges[18] = Edge(Line(7, 15), { 11, 5 }, { 19, 14 });
        edges[19] = Edge(Line(15, 14), { 5, 4 }, { 7, 6 });
        edges[20] = Edge(Line(15, 5), { 14, 18 }, { 7, 9 });
        edges[21] = Edge(Line(5, 18), { 15, 4 }, { 9, 16 });
        edges[22] = Edge(Line(18, 4), { 5, 14 }, { 16, 8 });
        edges[23] = Edge(Line(5, 9), { 15, 11 }, { 18, 1 });
        edges[24] = Edge(Line(9, 11), { 5, 7 }, { 1, 3 });
        edges[25] = Edge(Line(9, 1), { 11, 13 }, { 5, 16 });
        edges[26] = Edge(Line(1, 13), { 9, 3 }, { 16, 12 });
        edges[27] = Edge(Line(1, 16), { 9, 18 }, { 13, 0 });
        edges[28] = Edge(Line(16, 0), { 18, 8 }, { 1, 12 });
        edges[29] = Edge(Line(16, 18), { 1, 5 }, { 0, 4 });
    }

    void generate_triangles() override {

        for (const auto& edge : edges) {
            add_edge(edge.line, edge.f1);
            add_edge(edge.line, edge.f2);
        }

        size_t st = trigs.size();
        trigs.resize(st + 36);

        trigs[st + 0] = Triangle(vertexes[10], vertexes[6], vertexes[19], material);
        trigs[st + 1] = Triangle(vertexes[10], vertexes[19], vertexes[17], material);
        trigs[st + 2] = Triangle(vertexes[10], vertexes[17], vertexes[2], material);

        trigs[st + 3] = Triangle(vertexes[8], vertexes[10], vertexes[2], material);
        trigs[st + 4] = Triangle(vertexes[8], vertexes[2], vertexes[12], material);
        trigs[st + 5] = Triangle(vertexes[8], vertexes[12], vertexes[0], material);

        trigs[st + 6] = Triangle(vertexes[2], vertexes[17], vertexes[3], material);
        trigs[st + 7] = Triangle(vertexes[2], vertexes[3], vertexes[13], material);
        trigs[st + 8] = Triangle(vertexes[2], vertexes[13], vertexes[12], material);

        trigs[st + 9] = Triangle(vertexes[10], vertexes[6], vertexes[14], material);
        trigs[st + 10] = Triangle(vertexes[10], vertexes[14], vertexes[4], material);
        trigs[st + 11] = Triangle(vertexes[10], vertexes[4], vertexes[8], material);

        trigs[st + 12] = Triangle(vertexes[17], vertexes[19], vertexes[7], material);
        trigs[st + 13] = Triangle(vertexes[17], vertexes[7], vertexes[11], material);
        trigs[st + 14] = Triangle(vertexes[17], vertexes[11], vertexes[3], material);

        trigs[st + 15] = Triangle(vertexes[15], vertexes[14], vertexes[6], material);
        trigs[st + 16] = Triangle(vertexes[15], vertexes[6], vertexes[19], material);
        trigs[st + 17] = Triangle(vertexes[15], vertexes[19], vertexes[7], material);

        trigs[st + 18] = Triangle(vertexes[4], vertexes[14], vertexes[15], material);
        trigs[st + 19] = Triangle(vertexes[4], vertexes[15], vertexes[5], material);
        trigs[st + 20] = Triangle(vertexes[4], vertexes[5], vertexes[18], material);

        trigs[st + 21] = Triangle(vertexes[5], vertexes[15], vertexes[7], material);
        trigs[st + 22] = Triangle(vertexes[5], vertexes[7], vertexes[11], material);
        trigs[st + 23] = Triangle(vertexes[5], vertexes[11], vertexes[9], material);

        trigs[st + 24] = Triangle(vertexes[18], vertexes[5], vertexes[9], material);
        trigs[st + 25] = Triangle(vertexes[18], vertexes[9], vertexes[1], material);
        trigs[st + 26] = Triangle(vertexes[18], vertexes[1], vertexes[16], material);

        trigs[st + 27] = Triangle(vertexes[8], vertexes[4], vertexes[18], material);
        trigs[st + 28] = Triangle(vertexes[8], vertexes[18], vertexes[16], material);
        trigs[st + 29] = Triangle(vertexes[8], vertexes[16], vertexes[0], material);

        trigs[st + 30] = Triangle(vertexes[1], vertexes[9], vertexes[11], material);
        trigs[st + 31] = Triangle(vertexes[1], vertexes[11], vertexes[3], material);
        trigs[st + 32] = Triangle(vertexes[1], vertexes[3], vertexes[13], material);

        trigs[st + 33] = Triangle(vertexes[12], vertexes[0], vertexes[16], material);
        trigs[st + 34] = Triangle(vertexes[12], vertexes[16], vertexes[1], material);
        trigs[st + 35] = Triangle(vertexes[12], vertexes[1], vertexes[13], material);
    }

    const float phi = (1.0f + sqrt(5.0f)) / 2.0f;
    const float diod_koef = 0.022f * 1.3f;
};

class Icosahedron : public Figure {
public:
    Icosahedron(const float_3& center, float radius, const Material& material, size_t source_per_line)
        : Figure(center, radius, material, source_per_line) {
        initialize_figure();
    }

private:
    void generate_vertexes() override {
        vertexes.resize(12);

        vertexes[0] = float_3(0.0f, phi, 1.0f);
        vertexes[1] = float_3(0.0f, phi, -1.0f);
        vertexes[2] = float_3(0.0f, -phi, 1.0f);
        vertexes[3] = float_3(0.0f, -phi, -1.0f);

        vertexes[4] = float_3(1.0f, 0.0f, phi);
        vertexes[5] = float_3(1.0f, 0.0f, -phi);
        vertexes[6] = float_3(-1.0f, 0.0f, phi);
        vertexes[7] = float_3(-1.0f, 0.0f, -phi);

        vertexes[8] = float_3(phi, 1.0f, 0.0f);
        vertexes[9] = float_3(phi, -1.0f, 0.0f);
        vertexes[10] = float_3(-phi, 1.0f, 0.0f);
        vertexes[11] = float_3(-phi, -1.0f, 0.0f);
    }

    void scale_vertexes() override {
        float edge_length = 4.0f * radius / sqrt(10.0f + 2.0f * sqrt(5.0f));
        for (auto& vertex : vertexes) {
            vertex *= 0.5f * edge_length;
        }
    }

    void generate_edges() override {
        edges.resize(30);

        edges[0] = Edge(Line(2, 4), { 9, 9 }, { 6, 6 });
        edges[1] = Edge(Line(4, 9), { 8, 8 }, { 2, 2 });
        edges[2] = Edge(Line(9, 2), { 4, 4 }, { 3, 3 });
        edges[3] = Edge(Line(2, 6), { 4, 4 }, { 11, 11 });
        edges[4] = Edge(Line(6, 4), { 0, 0 }, { 2, 2 });
        edges[5] = Edge(Line(6, 0), { 4, 4 }, { 10, 10 });
        edges[6] = Edge(Line(0, 4), { 6, 6 }, { 8, 8 });
        edges[7] = Edge(Line(4, 8), { 9, 9 }, { 0, 0 });
        edges[8] = Edge(Line(8, 9), { 4, 4 }, { 5, 5 });
        edges[9] = Edge(Line(9, 5), { 3, 3 }, { 8, 8 });
        edges[10] = Edge(Line(5, 3), { 9, 9 }, { 7, 7 });
        edges[11] = Edge(Line(3, 9), { 2, 2 }, { 5, 5 });
        edges[12] = Edge(Line(3, 2), { 9, 9 }, { 11, 11 });
        edges[13] = Edge(Line(3, 11), { 2, 2 }, { 7, 7 });
        edges[14] = Edge(Line(11, 2), { 3, 3 }, { 6, 6 });
        edges[15] = Edge(Line(11, 6), { 2, 2 }, { 10, 10 });
        edges[16] = Edge(Line(11, 7), { 3, 3 }, { 10, 10 });
        edges[17] = Edge(Line(7, 3), { 11, 11 }, { 5, 5 });
        edges[18] = Edge(Line(7, 10), { 11, 11 }, { 1, 1 });
        edges[19] = Edge(Line(10, 11), { 7, 7 }, { 6, 6 });
        edges[20] = Edge(Line(10, 6), { 11, 11 }, { 0, 0 });
        edges[21] = Edge(Line(10, 0), { 6, 6 }, { 1, 1 });
        edges[22] = Edge(Line(0, 1), { 10, 10 }, { 8, 8 });
        edges[23] = Edge(Line(10, 1), { 0, 0 }, { 7, 7 });
        edges[24] = Edge(Line(1, 7), { 10, 10 }, { 5, 5 });
        edges[25] = Edge(Line(7, 5), { 1, 1 }, { 3, 3 });
        edges[26] = Edge(Line(0, 8), { 4, 4 }, { 1, 1 });
        edges[27] = Edge(Line(1, 8), { 0, 0 }, { 5, 5 });
        edges[28] = Edge(Line(1, 5), { 7, 7 }, { 8, 8 });
        edges[29] = Edge(Line(5, 8), { 9, 9 }, { 1, 1 });
    }

    void generate_triangles() override {

        for (const auto& edge : edges) {
            add_edge(edge.line, edge.f1);
            add_edge(edge.line, edge.f2);
        }

        size_t st = trigs.size();
        trigs.resize(st + 20);

        trigs[st + 0] = Triangle(vertexes[4], vertexes[9], vertexes[2], material);
        trigs[st + 1] = Triangle(vertexes[6], vertexes[4], vertexes[2], material);
        trigs[st + 2] = Triangle(vertexes[2], vertexes[3], vertexes[9], material);
        trigs[st + 3] = Triangle(vertexes[6], vertexes[11], vertexes[2], material);
        trigs[st + 4] = Triangle(vertexes[2], vertexes[11], vertexes[3], material);
        trigs[st + 5] = Triangle(vertexes[0], vertexes[6], vertexes[4], material);
        trigs[st + 6] = Triangle(vertexes[4], vertexes[9], vertexes[8], material);
        trigs[st + 7] = Triangle(vertexes[3], vertexes[9], vertexes[5], material);
        trigs[st + 8] = Triangle(vertexes[11], vertexes[7], vertexes[3], material);
        trigs[st + 9] = Triangle(vertexes[7], vertexes[11], vertexes[10], material);
        trigs[st + 10] = Triangle(vertexes[11], vertexes[10], vertexes[6], material);
        trigs[st + 11] = Triangle(vertexes[10], vertexes[6], vertexes[0], material);
        trigs[st + 12] = Triangle(vertexes[0], vertexes[10], vertexes[1], material);
        trigs[st + 13] = Triangle(vertexes[1], vertexes[7], vertexes[10], material);
        trigs[st + 14] = Triangle(vertexes[0], vertexes[1], vertexes[8], material);
        trigs[st + 15] = Triangle(vertexes[1], vertexes[7], vertexes[5], material);
        trigs[st + 16] = Triangle(vertexes[7], vertexes[3], vertexes[5], material);
        trigs[st + 17] = Triangle(vertexes[1], vertexes[5], vertexes[8], material);
        trigs[st + 18] = Triangle(vertexes[0], vertexes[8], vertexes[4], material);
        trigs[st + 19] = Triangle(vertexes[8], vertexes[5], vertexes[9], material);
    }

    const float phi = (1.0f + sqrt(5.0f)) / 2.0f;
};