#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // For std::vector conversion
#include <pybind11/numpy.h>    // For NumPy arrays
#include "particle_system.hpp"
#include "potential_field.hpp"
#include "density_grid.hpp"

namespace py = pybind11;

// Zero-copy path: accepts ONLY a native float32 numpy array (no forcecast).
// If the input is already float32 and C-contiguous, no allocation occurs.
// For any other input type (list, float64 array, etc.), pybind11 will skip
// this overload and fall through to the std::vector overload below, which
// performs a single O(N) copy -- the same cost as the original binding.
using NativeFloat32Array = py::array_t<float>;

PYBIND11_MODULE(stochastic_swarm, m) {
    m.doc() = "StochasticSwarm Python bindings - High-performance particle simulation";

    // -------------------------------------------------------------------------
    // PotentialField
    // -------------------------------------------------------------------------
    py::class_<PotentialField, std::shared_ptr<PotentialField>>(m, "PotentialField")
        .def(py::init<size_t, float>(),
             py::arg("num_basis"),
             py::arg("domain_size"),
             "Create parametric potential field with RBF basis functions\n\n"
             "Args:\n"
             "    num_basis: Number of Gaussian basis functions\n"
             "    domain_size: Physical size of simulation domain")

        .def("set_strengths", &PotentialField::set_strengths,
             py::arg("strengths"),
             "Set amplitude of each basis function (RL action)\n\n"
             "Args:\n"
             "    strengths: List of amplitudes (positive=repulsive, negative=attractive)")

        .def("set_parameters", &PotentialField::set_parameters,
             py::arg("centers_x"),
             py::arg("centers_y"),
             py::arg("strengths"),
             py::arg("widths"),
             "Set all potential field parameters at once")

        .def("get_centers_x", &PotentialField::get_centers_x,
             "Get x-coordinates of basis function centers")
        .def("get_centers_y", &PotentialField::get_centers_y,
             "Get y-coordinates of basis function centers")
        .def("get_strengths", &PotentialField::get_strengths,
             "Get current strength/amplitude values")
        .def("get_widths", &PotentialField::get_widths,
             "Get spatial width (sigma) of each basis function")
        .def_property_readonly("num_basis", &PotentialField::get_num_basis,
                               "Number of basis functions")
        .def("compute_force", &PotentialField::compute_force,
             py::arg("x"), py::arg("y"),
             "Compute force vector at position (x, y)\n\n"
             "Returns:\n"
             "    Tuple (Fx, Fy) of force components");

    // -------------------------------------------------------------------------
    // DensityGrid
    // -------------------------------------------------------------------------
    py::class_<DensityGrid, std::shared_ptr<DensityGrid>>(m, "DensityGrid")
        .def(py::init<size_t, size_t, float>(),
             py::arg("nx"),
             py::arg("ny"),
             py::arg("domain_size"),
             "Create density grid for spatial binning\n\n"
             "Args:\n"
             "    nx: Number of grid cells in x direction\n"
             "    ny: Number of grid cells in y direction\n"
             "    domain_size: Physical size of simulation domain")

        // OverLoad 1 -- zero-copy fast path.
        // Matched ONLY when x and y are already float32 C-contiguous 1-D arrays
        // (i.e. the output of ParticleSystem.get_x() / get_y()).
        // No allocation, no copy: raw pointer is passed straight to C++.
        .def("update",
             [](DensityGrid& dg, NativeFloat32Array x, NativeFloat32Array y) {
                 if (x.ndim() != 1 || y.ndim() != 1) {
                     throw std::runtime_error("x and y must be 1-D arrays");
                 }
                 if (x.shape(0) != y.shape(0)) {
                     throw std::runtime_error("x and y must have the same length");
                 }
                 dg.update(x.data(), y.data(),
                           static_cast<size_t>(x.shape(0)));
             },
             py::arg("x"),
             py::arg("y"),
             "Update density grid from float32 NumPy arrays (zero-copy)\n\n"
             "Args:\n"
             "    x: float32 NumPy array of x positions\n"
             "    y: float32 NumPy array of y positions")

        // Overload 2 -- legacy / fallback path.
        // Accepts any Python sequence (list, float64 array, etc.) via pybind11
        // STL auto-conversion.  One O(N) copy -- identical cost to the original
        // binding.  This overload is only reached when x/y are NOT float32 arrays.
        .def("update",
             py::overload_cast<const std::vector<float>&,
                               const std::vector<float>&>(&DensityGrid::update),
             py::arg("x"),
             py::arg("y"),
             "Update density grid from particle position sequences\n\n"
             "Args:\n"
             "    x: sequence of x positions (list, float64 array, etc.)\n"
             "    y: sequence of y positions")

        .def("clear", &DensityGrid::clear,
             "Clear grid (reset all counts to zero)")

        .def("normalize", &DensityGrid::normalize,
             "Normalize grid to density (particles per unit area)")

        // Zero-copy NumPy array pointing directly at C++ memory (READ-ONLY)
        .def("get_grid", [](DensityGrid& dg) {
             auto& grid = dg.get_grid_mutable();
             return py::array_t<float>(
                 {dg.get_ny(), dg.get_nx()},
                 {dg.get_nx() * sizeof(float), sizeof(float)},
                 grid.data(),
                 py::cast(&dg, py::return_value_policy::reference)
             );
         }, py::return_value_policy::reference_internal,
            "Get density grid as NumPy array (zero-copy, shape=(ny, nx))")

        .def_property_readonly("shape", [](const DensityGrid& dg) {
             return py::make_tuple(dg.get_ny(), dg.get_nx());
         }, "Grid dimensions (ny, nx)")

        .def_property_readonly("nx", &DensityGrid::get_nx, "Number of grid cells in x")
        .def_property_readonly("ny", &DensityGrid::get_ny, "Number of grid cells in y");

    // -------------------------------------------------------------------------
    // ParticleSystem
    // -------------------------------------------------------------------------
    py::class_<ParticleSystem>(m, "ParticleSystem")
        .def(py::init<size_t, float, size_t, size_t>(),
             py::arg("num_particles"),
             py::arg("temperature"),
             py::arg("num_basis") = 0,
             py::arg("grid_res")  = 32,
             "Create particle system with Langevin dynamics\n\n"
             "Args:\n"
             "    num_particles: Number of particles to simulate\n"
             "    temperature: Thermal energy (controls noise strength)\n"
             "    num_basis: Number of basis functions for potential field (0 = no field)\n"
             "    grid_res: Grid resolution for density grid (default 32x32)")

        .def("initialize_random", &ParticleSystem::initialize_random,
             py::arg("domain_size"),
             "Initialize particles with random positions and thermal velocities\n\n"
             "Args:\n"
             "    domain_size: Size of simulation domain (periodic boundaries)")

        .def("step", &ParticleSystem::step,
             "Advance simulation by one timestep using Euler-Maruyama integration")

        .def("record_velocities", &ParticleSystem::record_velocities,
             "Record current velocities for VACF analysis")

        .def_property_readonly("num_particles",
                               &ParticleSystem::get_num_particles,
                               "Number of particles in the system")

        // Position accessors -- return NumPy arrays (copies; positions change each step)
        .def("get_x", [](const ParticleSystem& ps) {
             const auto& v = ps.get_x();
             return py::array_t<float>(v.size(), v.data());
         }, "Get x positions as NumPy array")

        .def("get_y", [](const ParticleSystem& ps) {
             const auto& v = ps.get_y();
             return py::array_t<float>(v.size(), v.data());
         }, "Get y positions as NumPy array")

        .def("get_vx", [](const ParticleSystem& ps) {
             const auto& v = ps.get_vx();
             return py::array_t<float>(v.size(), v.data());
         }, "Get x velocities as NumPy array")

        .def("get_vy", [](const ParticleSystem& ps) {
             const auto& v = ps.get_vy();
             return py::array_t<float>(v.size(), v.data());
         }, "Get y velocities as NumPy array")

        .def("get_initial_x", [](const ParticleSystem& ps) {
             const auto& v = ps.get_initial_x();
             return py::array_t<float>(v.size(), v.data());
         }, "Get initial x positions as NumPy array")

        .def("get_initial_y", [](const ParticleSystem& ps) {
             const auto& v = ps.get_initial_y();
             return py::array_t<float>(v.size(), v.data());
         }, "Get initial y positions as NumPy array")

        .def("get_vx_history", &ParticleSystem::get_vx_history,
             "Get x velocity history for VACF computation")

        .def("get_vy_history", &ParticleSystem::get_vy_history,
             "Get y velocity history for VACF computation")

        .def("set_potential_params", &ParticleSystem::set_potential_params,
             py::arg("strengths"),
             "Set potential field strengths (RL action interface)\n\n"
             "Args:\n"
             "    strengths: List of amplitudes for each basis function")

        .def("get_potential_field", &ParticleSystem::get_potential_field,
             "Get potential field object for advanced control\n\n"
             "Returns:\n"
             "    PotentialField object or None if no field configured")

        .def("update_density_grid", &ParticleSystem::update_density_grid,
             "Update density grid from current particle positions\n"
             "Call this before getting the grid for RL observation")

        .def("get_density_grid",
             py::overload_cast<>(&ParticleSystem::get_density_grid),
             py::return_value_policy::reference_internal,
             "Get density grid object for RL observation\n\n"
             "Returns:\n"
             "    DensityGrid object with zero-copy access");

    // Module version
    m.attr("__version__") = "1.1.0";
}
