/**
 * Python wrappers for raisim.object using pybind11.
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
//#include <pybind11/numpy.h>   // numpy types

#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, etc.
#include "raisim/object/singleBodies/Box.hpp"
#include "raisim/object/singleBodies/Capsule.hpp"
#include "raisim/object/singleBodies/Compound.hpp"
#include "raisim/object/singleBodies/Cone.hpp"
#include "raisim/object/singleBodies/Cylinder.hpp"
#include "raisim/object/singleBodies/Mesh.hpp"
#include "raisim/object/singleBodies/SingleBodyObject.hpp"
#include "raisim/object/singleBodies/Sphere.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = pybind11;
using namespace raisim;


void init_single_bodies(py::module &m) {

    /********************/
	/* SingleBodyObject */
	/********************/
	py::class_<raisim::SingleBodyObject, raisim::Object>(m, "SingleBodyObject", "Raisim Single Object from which all single objects/bodies (such as box, sphere, etc) inherit from.")
	    .def(py::init<raisim::ObjectType>(), "Initialize the Object.", py::arg("object_type"))
	    .def("get_position", &raisim::SingleBodyObject::getPosition, R"mydelimiter(
	    Get the body's position with respect to the world frame.

	    Returns:
	        np.array[float[3]]: position in the world frame.
	    )mydelimiter")
	    .def("get_com_position", &raisim::SingleBodyObject::getComPosition, R"mydelimiter(
	    Get the body's center of mass position with respect to the world frame.

	    Returns:
	        np.array[float[3]]: center of mass position in the world frame.
	    )mydelimiter")
	    .def("get_linear_velocity", &raisim::SingleBodyObject::getLinearVelocity, R"mydelimiter(
	    Get the body's linear velocity with respect to the world frame.

	    Returns:
	        np.array[float[3]]: linear velocity in the world frame.
	    )mydelimiter")
	    .def("get_angular_velocity", &raisim::SingleBodyObject::getAngularVelocity, R"mydelimiter(
	    Get the body's angular velocity position with respect to the world frame.

	    Returns:
	        np.array[float[3]]: angular velocity in the world frame.
	    )mydelimiter")
	    .def("get_quaternion", py::overload_cast<>(&raisim::SingleBodyObject::getQuaternion), R"mydelimiter(
	    Get the body's orientation (expressed as a quaternion [w,x,y,z]) with respect to the world frame.

	    Returns:
	        np.array[float[4]]: quaternion [w,x,y,z].
	    )mydelimiter")
	    .def("get_rotation_matrix", py::overload_cast<>(&raisim::SingleBodyObject::getRotationMatrix), R"mydelimiter(
	    Get the body's orientation (expressed as a rotation matrix) with respect to the world frame.

	    Returns:
	        np.array[float[3,3]]: rotation matrix.
	    )mydelimiter")
	    .def("get_kinetic_energy", &raisim::SingleBodyObject::getKineticEnergy, R"mydelimiter(
	    Get the body's kinetic energy.

	    Returns:
	        float: kinetic energy.
	    )mydelimiter")
	    .def("get_potential_energy", &raisim::SingleBodyObject::getPotentialEnergy, R"mydelimiter(
	    Get the body's potential energy.

	    Returns:
	        float: potential energy.
	    )mydelimiter")
	    .def("get_energy", &raisim::SingleBodyObject::getEnergy, R"mydelimiter(
	    Get the body's total energy.

	    Returns:
	        float: total energy.
	    )mydelimiter")
	    .def("get_linear_momentum", &raisim::SingleBodyObject::getLinearMomentum, R"mydelimiter(
	    Get the body's linear momentum.

	    Returns:
	        np.array[float[3]]: linear momentum.
	    )mydelimiter")
	    .def("get_mass", &raisim::SingleBodyObject::getMass, R"mydelimiter(
	    Get the body's mass.

	    Returns:
	        float: mass (kg).
	    )mydelimiter")
	    .def("get_world_inertia_matrix", &raisim::SingleBodyObject::getInertiaMatrix_W, R"mydelimiter(
	    Get the body's inertia matrix expressed in the world frame.

	    Returns:
	        np.array[float[3,3]]: world inertia matrix.
	    )mydelimiter")
	    .def("get_body_inertia_matrix", &raisim::SingleBodyObject::getInertiaMatrix_B, R"mydelimiter(
	    Get the body's inertia matrix expressed in the body frame.

	    Returns:
	        np.array[float[3,3]]: body inertia matrix.
	    )mydelimiter")
	    .def("get_object_type", &raisim::SingleBodyObject::getObjectType, R"mydelimiter(
	    Get the body's type.

	    Returns:
	        raisim.ObjectType: object type (BOX, CYLINDER, CAPSULE, CONE, SPHERE, etc.)
	    )mydelimiter")
	    ;


    /*******/
	/* Box */
	/*******/
	py::class_<raisim::Box, raisim::SingleBodyObject>(m, "Box", "Raisim Box.")
	    .def(py::init<double, double, double, double>(),
	    "Initialize a box.\n\n"
	    "Args:\n"
	    "    x (float): length along the x axis.\n"
	    "    y (float): length along the y axis.\n"
	    "    z (float): length along the z axis.\n"
	    "    mass (float): mass of the box.",
	    py::arg("x"), py::arg("y"), py::arg("z"), py::arg("mass"))
	    .def("get_dimensions", [](raisim::Box &box) {
	        Vec<3> dimensions = box.getDim();
	        return convert_vec_to_np(dimensions);
	    }, R"mydelimiter(
	    Get the box's dimensions.

	    Returns:
	        tuple[float[3]]: dimensions along each axis.
	    )mydelimiter");


    /***********/
	/* Capsule */
	/***********/
	py::class_<raisim::Capsule, raisim::SingleBodyObject>(m, "Capsule", "Raisim Capsule.")
	    .def(py::init<double, double, double>(),
	    "Initialize a capsule.\n\n"
	    "Args:\n"
	    "    radius (float): radius of the capsule.\n"
	    "    height (float): height of the capsule.\n"
	    "    mass (float): mass of the capsule.",
	    py::arg("radius"), py::arg("height"), py::arg("mass"))
	    .def("get_radius", &raisim::Capsule::getRadius, R"mydelimiter(
	    Get the capsule's radius.

	    Returns:
	        float: radius of the capsule.
	    )mydelimiter")
	    .def("get_height", &raisim::Capsule::getHeight, R"mydelimiter(
	    Get the capsule's height.

	    Returns:
	        float: height of the capsule.
	    )mydelimiter");


    /************/
    /* Compound */
    /************/


    /********/
	/* Cone */
	/********/
	py::class_<raisim::Cone, raisim::SingleBodyObject>(m, "Cone", "Raisim Cone.")
	    .def(py::init<double, double, double>(),
	    "Initialize a cone.\n\n"
	    "Args:\n"
	    "    radius (float): radius of the cone.\n"
	    "    height (float): height of the cone.\n"
	    "    mass (float): mass of the cone.",
	    py::arg("radius"), py::arg("height"), py::arg("mass"))
	    .def("get_radius", &raisim::Cone::getRadius, R"mydelimiter(
	    Get the cone's radius.

	    Returns:
	        float: radius of the cone.
	    )mydelimiter")
	    .def("get_height", &raisim::Cone::getHeight, R"mydelimiter(
	    Get the cone's height.

	    Returns:
	        float: height of the cone.
	    )mydelimiter");


    /************/
    /* Cylinder */
    /************/
	py::class_<raisim::Cylinder, raisim::SingleBodyObject>(m, "Cylinder", "Raisim Cylinder.")
	    .def(py::init<double, double, double>(),
	    "Initialize a cylinder.\n\n"
	    "Args:\n"
	    "    radius (float): radius of the cylinder.\n"
	    "    height (float): height of the cylinder.\n"
	    "    mass (float): mass of the cylinder.",
	    py::arg("radius"), py::arg("height"), py::arg("mass"))
	    .def("get_radius", &raisim::Cylinder::getRadius, R"mydelimiter(
	    Get the cylinder's radius.

	    Returns:
	        float: radius of the cylinder.
	    )mydelimiter")
	    .def("get_height", &raisim::Cylinder::getHeight, R"mydelimiter(
	    Get the cylinder's height.

	    Returns:
	        float: height of the cylinder.
	    )mydelimiter");


    /********/
    /* Mesh */
    /********/
//	py::class_<raisim::Mesh, raisim::SingleBodyObject>(m, "Mesh", "Raisim Mesh.")
//	    .def(py::init<const std::string&, dSpaceID>(),
//	    "Initialize a Mesh.\n\n"
//	    "Args:\n"
//	    "    filename (str): path to the mesh file.\n"
//	    "    space (dSpaceID): space.",
//	    py::arg("filename"), py::arg("space"));


    /**********/
	/* Sphere */
	/**********/
	py::class_<raisim::Sphere, raisim::SingleBodyObject>(m, "Sphere", "Raisim Sphere.")
	    .def(py::init<double, double>(),
	    "Initialize a sphere.\n\n"
	    "Args:\n"
	    "    radius (float): radius of the sphere.\n"
	    "    mass (float): mass of the sphere.",
	    py::arg("radius"), py::arg("mass"))
	    .def("get_radius", &raisim::Sphere::getRadius, R"mydelimiter(
	    Get the sphere's radius.

	    Returns:
	        float: radius of the sphere.
	    )mydelimiter");

}