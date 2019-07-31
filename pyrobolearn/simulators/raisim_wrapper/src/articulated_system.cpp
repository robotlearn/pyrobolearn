/**
 * Python wrappers for raisim.object.ArticulatedSystem using pybind11.
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

#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, etc.
#include "raisim/World.hpp"

namespace py = pybind11;
using namespace raisim;


void init_articulated_system(py::module &m) {


    /****************/
    /* LoadFromMJCF */
    /****************/
//    py::class_<raisim::mjcf::LoadFromMJCF>(m, "LoadFromMJCF", "Load from MJCF file.");


    /*****************/
    /* LoadFromURDF2 */
    /*****************/
//    py::class_<raisim::urdf::LoadFromURDF2>(m, "LoadFromURDF2", "Load from URDF file.");


    /***************/
    /* ControlMode */
    /***************/
    py::enum_<raisim::ControlMode::Type>(m, "Type", py::arithmetic())
	    .value("FORCE_AND_TORQUE", raisim::ControlMode::Type::FORCE_AND_TORQUE)
	    .value("PD_PLUS_FEEDFORWARD_TORQUE", raisim::ControlMode::Type::PD_PLUS_FEEDFORWARD_TORQUE)
	    .value("VELOCITY_PLUS_FEEDFORWARD_TORQUE", raisim::ControlMode::Type::VELOCITY_PLUS_FEEDFORWARD_TORQUE);


    /***************************/
    /* ArticulatedSystemOption */
    /***************************/
    py::class_<raisim::ArticulatedSystemOption>(m, "ArticulatedSystemOption", "Articulated System Option.")
        .def_readwrite("do_not_collide_with_parent", &raisim::ArticulatedSystemOption::doNotCollideWithParent);


    /*********************/
    /* ArticulatedSystem */
    /*********************/
    py::class_<raisim::ArticulatedSystem, raisim::Object> system(m, "ArticulatedSystem", "Raisim Articulated System.");

    system.def(py::init<>(), "Initialize the Articulated System.")
          .def(py::init<const std::string &, const std::string &, std::vector<std::string>, raisim::ArticulatedSystemOption>(),
            "Initialize the Articulated System." )  // TODO: finish the doc
          .def("get_generalized_coordinate", [](raisim::ArticulatedSystem &self) {
            return ;
          })
          .def("update_kinematics", &raisim::ArticulatedSystem::updateKinematics,  R"mydelimiter(
          unnecessary to call this function if you are simulating your system. `integrate1` calls this function Call
          this function if you want to get kinematic properties but you don't want to integrate.
          )mydelimiter")
    ;

    py::enum_<raisim::ArticulatedSystem::Frame>(system, "Frame")
        .value("WORLD_FRAME", raisim::ArticulatedSystem::Frame::WORLD_FRAME)
        .value("PARENT_FRAME", raisim::ArticulatedSystem::Frame::PARENT_FRAME)
        .value("BODY_FRAME", raisim::ArticulatedSystem::Frame::BODY_FRAME);


}