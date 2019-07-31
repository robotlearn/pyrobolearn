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

#include "raisim/object/Object.hpp"

namespace py = pybind11;
using namespace raisim;


void init_single_bodies(py::module &);
void init_articulated_system(py::module &);
void init_terrain(py::module &);


void init_object(py::module &m) {

    // create submodule
    py::module object_module = m.def_submodule("object", "RaiSim contact submodule.");


    /**************/
    /* ObjectType */
    /**************/
	// object type enum (from include/raisim/configure.hpp)
	py::enum_<raisim::ObjectType>(object_module, "ObjectType", py::arithmetic())
	    .value("SPHERE", raisim::ObjectType::SPHERE)
	    .value("BOX", raisim::ObjectType::BOX)
	    .value("CYLINDER", raisim::ObjectType::CYLINDER)
	    .value("CONE", raisim::ObjectType::CONE)
	    .value("CAPSULE", raisim::ObjectType::CAPSULE)
	    .value("MESH", raisim::ObjectType::MESH)
	    .value("HALFSPACE", raisim::ObjectType::HALFSPACE)
	    .value("COMPOUND", raisim::ObjectType::COMPOUND)
	    .value("HEIGHTMAP", raisim::ObjectType::HEIGHTMAP)
	    .value("ARTICULATED_SYSTEM", raisim::ObjectType::ARTICULATED_SYSTEM);


    /************/
    /* BodyType */
    /************/
	// body type enum (from include/raisim/configure.hpp)
	py::enum_<raisim::BodyType>(object_module, "BodyType", py::arithmetic())
	    .value("STATIC", raisim::BodyType::STATIC)
	    .value("KINEMATIC", raisim::BodyType::KINEMATIC)
	    .value("DYNAMIC", raisim::BodyType::DYNAMIC);


	/**********/
	/* Object */
	/**********/
	py::class_<raisim::Object>(object_module, "Object", "Raisim Object from which all other objects/bodies inherit from.")
	    .def_property("name", &raisim::Object::getName, &raisim::Object::setName)
	    .def("get_name", &raisim::Object::getName, "Get the object's name.")
	    .def("set_name", &raisim::Object::setName, "Set the object's name.", py::arg("name"))
	    .def("clear_per_object_contact", &raisim::Object::clearPerObjectContact)
	    .def("add_contact_to_per_object_contact", &raisim::Object::addContactToPerObjectContact)
	    ;


	// raisim.object.singleBodies
	init_single_bodies(object_module);

	// raisim.object.ArticulatedSystem
	init_articulated_system(object_module);

	// raisim.object.terrain
	init_terrain(object_module);

}