/**
 * Python wrappers for RaiSim using pybind11.
 *
 * Copyright (c) 2019, Brian Delhaisse <briandelhaisse@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // automatic conversion between std::vector, std::list, etc to Python list/tuples/dict
#include <pybind11/eigen.h>   // automatic conversion between Eigen data types to Numpy data types

#include <iostream>

#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, etc.
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/OgreVis.hpp"
//#include "visualizer/raisimKeyboardCallback.hpp"
//#include "visualizer/helper.hpp"
//#include "visualizer/guiState.hpp"
//#include "visualizer/raisimBasicImguiPanel.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = pybind11;
using namespace raisim;


void init_object(py::module &);
void init_constraints(py::module &);
void init_contact(py::module &);
// void init_visualizer(py::module &);


// The PYBIND11_MODULE() macro creates a function that will be called when an import statement is issued from within
// Python. In the following, "raisim" is the module name, "m" is a variable of type py::module which is the main
// interface for creating bindings. The method module::def() generates binding code that exposes the C++ function
// to Python.
PYBIND11_MODULE(raisimpy, m) {

	m.doc() = "Python wrappers for the RaiSim library and visualizer."; // docstring for the module


	/*************/
	/* Materials */
    /*************/
    py::class_<raisim::MaterialPairProperties>(m, "MaterialPairProperties", "Raisim Material Pair Properties (friction and restitution).")
        .def(py::init<>(), "Initialize the material pair properties.")
        .def(py::init<double, double, double>(),
        "Initialize the material pair properties.\n\n"
        "Args:\n"
        "    friction (float): coefficient of friction.\n"
        "    restitution (float): coefficient of restitution.\n"
        "    threshold (float): restitution threshold.",
        py::arg("friction"), py::arg("restitution"), py::arg("threshold"));


    py::class_<raisim::MaterialManager>(m, "MaterialManager", "Raisim Material Manager.")
        .def(py::init<>(), "Initialize the material pair manager.")
        .def(py::init<const std::string>(),
        "Initialize the material manager by uploading the material data from a file.\n\n"
        "Args:\n"
        "    xml_file (float): xml file.",
        py::arg("xml_file"))
        .def("set_material_pair_properties", &raisim::MaterialManager::setMaterialPairProp, R"mydelimiter(
        Set the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            threshold (float): restitution threshold.
        )mydelimiter",
        py::arg("material1"), py::arg("material2"), py::arg("friction"), py::arg("restitution"), py::arg("threshold"))
        .def("get_material_pair_properties", &raisim::MaterialManager::getMaterialPairProp, R"mydelimiter(
        Get the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.

        Returns:
            MaterialPairProperties: material pair properties (friction, restitution, and restitution threshold).
        )mydelimiter",
        py::arg("material1"), py::arg("material2"))
        .def("set_default_material_properties", &raisim::MaterialManager::setDefaultMaterialProperties, R"mydelimiter(
        Set the default material properties.

        Args:
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            threshold (float): restitution threshold.
        )mydelimiter",
        py::arg("friction"), py::arg("restitution"), py::arg("threshold"))
    ;


    /******************/
	/* raisim.contact */
	/******************/
	init_contact(m);

    /*****************/
	/* raisim.object */
	/*****************/
	init_object(m);  // define primitive shapes and articulated systems)

    /*********************/
	/* raisim.constraint */
	/*********************/
    init_constraints(m);

    /*********/
	/* World */
	/*********/
	py::class_<raisim::World>(m, "World", "Raisim world.", py::dynamic_attr()) // enable dynamic attributes for C++ class in Python
	    .def(py::init<>(), "Initialize the World.")
	    .def(py::init<const std::string &>(), "Initialize the World from the given config file.", py::arg("configFile"))
	    .def("set_time_step", &raisim::World::setTimeStep, R"mydelimiter(
	    Set the given time step `dt` in the simulator.

	    Args:
	        dt (float): time step to be set in the simulator.
	    )mydelimiter",
	    py::arg("dt"))
	    .def("get_time_step", &raisim::World::getTimeStep, R"mydelimiter(
	    Get the current time step that has been set in the simulator.

	    Returns:
	        float: time step.
	    )mydelimiter")

//	    .def("add_sphere", &raisim::World::addSphere, R"mydelimiter(
//	    Add dynamically a sphere into the world.
//
//	    Args:
//	        radius (float): radius of the sphere.
//	        mass (float): mass of the sphere.
//	        material (str): material to be applied to the sphere.
//	        collision_group (unsigned long): collision group.
//	        collision_mask (unsigned long): collision mask.
//	    Returns:
//	        Sphere: the sphere instance.
//	    )mydelimiter",
//	    py::arg("radius"), py::arg("mass"), py::arg("material") = "default", py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1))
//	    .def("add_box", &raisim::World::addBox, R"mydelimiter(
//	    Add dynamically a box into the world.
//
//	    Args:
//	        x (float): length along the x axis.
//	        y (float): length along the y axis.
//	        z (float): length along the z axis.
//	        mass (float): mass of the box.
//	        material (str): material to be applied to the box.
//	        collision_group (unsigned long): collision group.
//	        collision_mask (unsigned long): collision mask.
//	    Returns:
//	        Box: the box instance.
//	    )mydelimiter",
//	    py::arg("x"), py::arg("y"), py::arg("z"), py::arg("mass"), py::arg("material") = "default", py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1))
//        .def("add_cylinder", &raisim::World::addCylinder, R"mydelimiter(
//	    Add dynamically a cylinder into the world.
//
//	    Args:
//	        radius (float): radius of the cylinder.
//	        height (float): height of the cylinder.
//	        mass (float): mass of the cylinder.
//	        material (str): material to be applied to the cylinder.
//	        collision_group (unsigned long): collision group.
//	        collision_mask (unsigned long): collision mask.
//	    Returns:
//	        Cylinder: the cylinder instance.
//	    )mydelimiter",
//	    py::arg("radius"), py::arg("height"), py::arg("mass"), py::arg("material") = "default", py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1))
//	    .def("add_cone", &raisim::World::addCone, R"mydelimiter(
//	    Add dynamically a cone into the world.
//
//	    Args:
//	        radius (float): radius of the cone.
//	        height (float): height of the cone.
//	        mass (float): mass of the cone.
//	        material (str): material to be applied to the cone.
//	        collision_group (unsigned long): collision group.
//	        collision_mask (unsigned long): collision mask.
//	    Returns:
//	        Cone: the cone instance.
//	    )mydelimiter",
//	    py::arg("radius"), py::arg("height"), py::arg("mass"), py::arg("material") = "default", py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1))
//	    .def("add_capsule", &raisim::World::addCapsule, R"mydelimiter(
//	    Add dynamically a capsule into the world.
//
//	    Args:
//	        radius (float): radius of the capsule.
//	        height (float): height of the capsule.
//	        mass (float): mass of the capsule.
//	        material (str): material to be applied to the capsule.
//	        collision_group (unsigned long): collision group.
//	        collision_mask (unsigned long): collision mask.
//	    Returns:
//	        Capsule: the capsule instance.
//	    )mydelimiter",
//	    py::arg("radius"), py::arg("height"), py::arg("mass"), py::arg("material") = "default", py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1))
//	    .def("add_ground", &raisim::World::addGround, R"mydelimiter(
//	    Add dynamically a ground into the world.
//
//	    Args:
//	        height (float): height of the ground.
//	        material (str): material to be applied to the ground.
//	        collision_mask (unsigned long): collision mask.
//	    Returns:
//	        Ground: the ground instance.
//	    )mydelimiter",
//	    py::arg("height"), py::arg("material") = "default", py::arg("collision_mask") = CollisionGroup(-1))

//        .def("add_heightmap", &raisim::World::, R"mydelimiter(
//	    Add dynamically a ground into the world.
//
//	    Args:
//	        height (float): height of the ground.
//	        material (str): material to be applied to the ground.
//	        collision_mask (unsigned long): collision mask.
//	    Returns:
//	        Ground: the ground instance.
//	    )mydelimiter",
//	    py::arg("height"), py::arg("material") = "default", py::arg("collision_mask") = CollisionGroup(-1))


	    .def("integrate", &raisim::World::integrate, "this function is simply calling both `integrate1()` and `integrate2()` one-by-one.")
	    .def("integrate1", &raisim::World::integrate1, R"mydelimiter(
        It performs:
        1. deletion contacts from previous time step
        2. collision detection
        3. register contacts to each body
        4. calls `preContactSolverUpdate1()` of each object
        )mydelimiter")
        .def("integrate2", &raisim::World::integrate2, R"mydelimiter(
        It performs
        1. calls `preContactSolverUpdate2()` of each body
        2. run collision solver
        3. calls `integrate` method of each object
        )mydelimiter")
        .def("get_gravity", [](raisim::World &world) {
            Vec<3> gravity = world.getGravity();
            return convert_vec_to_np(gravity);
        }, R"mydelimiter(
        Get the gravity vector from the world.

        Returns:
            np.array[float[3]]: gravity vector.
        )mydelimiter")
        .def("set_gravity", [](raisim::World &world, py::array_t<double> array) {
            raisim::Vec<3> gravity = convert_np_to_vec<3>(array);
            world.setGravity(gravity);
        }, R"mydelimiter(
        Set the gravity vector in the world.

        Args:
            np.array[float[3]]: gravity vector.
        )mydelimiter", py::arg("gravity"))
        .def("set_erp", &raisim::World::setERP, "Set the error reduction parameter (ERP).", py::arg("erp"), py::arg("erp2")=0)
        .def("set_contact_solver_parameters", &raisim::World::setContactSolverParam, R"mydelimiter(
        Set contact solver parameters.

        Args:
            alpha_init (float): alpha init.
            alpha_min (float): alpha minimum.
            alpha_decay (float): alpha decay.
            max_iters (float): maximum number of iterations.
            threshold (float): threshold.
        )mydelimiter",
        py::arg("alpha_init"), py::arg("alpha_min"), py::arg("alpha_decay"), py::arg("max_iters"), py::arg("threshold"))
        .def("get_world_time", &raisim::World::getWorldTime, R"mydelimiter(
        Return the total integrated time (which is updated at every `integrate2()`` call).

        Returns:
            float: world time.
        )mydelimiter")
        .def("set_world_time", &raisim::World::setWorldTime, R"mydelimiter(
        Set the world time.

        Args:
            time (float): world time
        )mydelimiter", py::arg("time"))
        ;

	// visualizer class
//	py::class_<raisim::OgreVis, std::unique_ptr<raisim::Ogrevis, py::nodelete>>(m, "Visualizer", "Ogre visualizer for Raisim.")
//	    .def(py::init(&raisim::OgreVis::get), "Create Ogre visualizer instance (singleton).", py::return_value_policy::reference)
//	    .def("get", &raisim::OgreVis::get, "Get the single Ogre visualizer instance (singleton).")
//	    .def();
}