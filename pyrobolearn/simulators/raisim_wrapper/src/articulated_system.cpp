/**
 * Python wrappers for raisim.object.ArticulatedSystem using pybind11.
 *
 * Copyright (c) 2019, jhwangbo (C++), Brian Delhaisse <briandelhaisse@gmail.com> (Python wrappers)
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
#include "raisim/object/ArticulatedSystem/loaders.hpp"
#include "raisim/object/ArticulatedSystem/JointAndBodies.hpp"
#include "raisim/object/ArticulatedSystem/ArticulatedSystem.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = pybind11;
using namespace raisim;


void init_articulated_system(py::module &m) {


    /****************/
    /* LoadFromMJCF */
    /****************/
    py::class_<raisim::mjcf::LoadFromMJCF>(m, "LoadFromMJCF", "Load from MJCF file.")
        .def(py::init<ArticulatedSystem &, std::string, std::vector<std::string>>(), "Initialize the MJCF loader.");


    /*****************/
    /* LoadFromURDF2 */
    /*****************/
    py::class_<raisim::urdf::LoadFromURDF2>(m, "LoadFromURDF2", "Load from URDF file.")
        .def(py::init<ArticulatedSystem &, std::string, std::vector<std::string>>(), "Initialize the URDF loader.");


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

    // From the `ArticulatedSystem.h` file:
    /* list of vocabs
     1. body: body here refers to only rotating bodies. Fixed bodies are optimized out.
              Position of a body refers to the position of the joint connecting the body and its parent.
     2. Coordinate frame: coordinate frames are defined on every joint (even at the fixed joints). If you want
                          to define a custom frame, define a fixed zero-mass object and a joint in the URDF */

    py::class_<raisim::ArticulatedSystem, raisim::Object> system(m, "ArticulatedSystem", "Raisim Articulated System.");

    py::enum_<raisim::ArticulatedSystem::Frame>(system, "Frame")
        .value("WORLD_FRAME", raisim::ArticulatedSystem::Frame::WORLD_FRAME)
        .value("PARENT_FRAME", raisim::ArticulatedSystem::Frame::PARENT_FRAME)
        .value("BODY_FRAME", raisim::ArticulatedSystem::Frame::BODY_FRAME);

    system.def(py::init<>(), "Initialize the Articulated System.")

        .def(py::init<const std::string &, const std::string &, std::vector<std::string>, raisim::ArticulatedSystemOption>(),
        "Initialize the Articulated System.\n\n"
        "Do not call this method yourself. use World class to create an Articulated system.\n\n"
        "Args:\n"
        "    filename (str): path to the robot description file (URDF, etc).\n"
        "    resource_directory (str): path the resource directory. If empty, it will use the robot description folder.\n"
        "    joint_order (list[str]): specify the joint order, if we want it to be different from the URDF file.\n"
        "    options (ArticulatedSystemOption): options.",
        py::arg("filename"), py::arg("resource_directory"), py::arg("joint_order"), py::arg("options"))


        .def("get_generalized_coordinate", [](raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getGeneralizedCoordinate());
        }, R"mydelimiter(
        Get the generalized coordinates of the system.

        Returns:
            np.array[float[n]]: generalized coordinates.
        )mydelimiter")


        .def("get_base_orientation", [](raisim::ArticulatedSystem &self) {
            Vec<4> quaternion;
            self.getBaseOrientation(quaternion);
            return convert_vec_to_np(quaternion);
        }, R"mydelimiter(
        Get the base orientation (expressed as a quaternion [w,x,y,z]).

        Returns:
            np.array[float[4]]: base orientation (expressed as a quaternion [w,x,y,z])
        )mydelimiter")


        .def("get_base_rotation_matrix", [](raisim::ArticulatedSystem &self) {
            Mat<3,3> rot;
            self.getBaseOrientation(rot);
            return convert_mat_to_np(rot);
        }, R"mydelimiter(
        Get the base orientation (expressed as a rotation matrix).

        Returns:
            np.array[float[3,3]]: rotation matrix
        )mydelimiter")


        .def("get_generalized_velocity", [](raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getGeneralizedVelocity());
        }, R"mydelimiter(
        Get the generalized velocities of the system.

        Returns:
            np.array[float[n]]: generalized velocities.
        )mydelimiter")


        .def("update_kinematics", &raisim::ArticulatedSystem::updateKinematics,  R"mydelimiter(
        unnecessary to call this function if you are simulating your system. `integrate1` calls this function Call
        this function if you want to get kinematic properties but you don't want to integrate.
        )mydelimiter")
    ;




}