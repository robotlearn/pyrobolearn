/**
 * Python wrappers for raisimOgre using pybind11.
 *
 * Copyright (c) 2019, jhwangbo (original C++), Brian Delhaisse <briandelhaisse@gmail.com> (Python wrappers)
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

// include Ogre related headers
#include "Ogre.h"
#include "OgreApplicationContext.h"
#include "OgreInput.h"

// include Raisim Ogre related headers
#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, etc.
#include "raisim/interfaceClasses.hpp"
#include "raisim/OgreVis.hpp"
//#include "visualizer/helper.hpp"
//#include "visualizer/guiState.hpp"
#include "visualizer/raisimBasicImguiPanel.hpp"   // for `imguiSetupCallback` and  `imguiRenderCallBack`
#include "visualizer/raisimKeyboardCallback.hpp"  // for `raisimKeyboardCallback`
#include "visualizer/visSetupCallback.hpp"        // for `setupCallback`

// include headers that allows to convert between raisim::Vec, raisim::Mat, raisim::VecDyn, raisim::Mat to np.arrays.
#include "converter.hpp"

namespace py = pybind11;
using namespace raisim;


/// \brief: Visualizer class which inherits from raisim::OgreVis. It adds other few functionalities that were missing.
//class Visualizer : public raisim::OgreVis {
//
//  public:
//
//    using MouseButtonCallback = std::function<bool(const MouseButtonEvent &)>;
//    using MouseMotionCallback = std::function<bool(const MouseMotionEvent &)>;
//    using MouseWheelCallback = std::function<bool(const MouseWheelEvent &)>;
//
//    /** set mouse callback. This callback is called for every mouse event */
//    void setMouseButtonCallback(MouseButtonCallback callback) { mouseButtonCallback_ = callback; }
//    void setMouseMotionCallback(MouseMotionCallback callback) { mouseMotionCallback_ = callback; }
//    void setMouseWheelCallback(MouseWheelCallback callback) { mouseWheelCallback_ = callback; }
//
//  private:
//
//    MouseButtonCallback mouseButtonCallback_ = nullptr;
//    MouseMotionCallback mouseMotionCallback_ = nullptr;
//    MouseWheelCallback mouseWheelCallback_ = nullptr;
//};


void init_visualizer(py::module &m) {

    // create submodule
    py::module visualizer_module = m.def_submodule("visualizer", "RaiSim visualizer submodule.");

    /*****************/
    /* GraphicObject */
    /*****************/

    py::class_<raisim::GraphicObject>(visualizer_module, "GraphicObject", "Graphic object represents the underlying object.")
        .def(py::init<>(), "Instantiate the Graphic Object by setting its orientation, scale, and offset position.");

    /****************/
    /* VisualObject */
    /****************/

    py::class_<raisim::VisualObject>(visualizer_module, "VisualObject", "Visual object is for visualization only")
        .def(py::init<>(), "Instantiate a visual object (by setting its orientation).");

    /***********/
    /* OgreVis */
    /***********/

    // py::nodedelete is because the destructor is non-public (it is private because of Singleton pattern)
    py::class_<raisim::OgreVis, std::unique_ptr<raisim::OgreVis, py::nodelete>> ogre_vis(visualizer_module, "OgreVis", "Raisim Ogre visualizer.");

    py::enum_<raisim::OgreVis::VisualizationGroup>(ogre_vis, "VisualizationGroup")
        .value("RAISIM_OBJECT_GROUP", raisim::OgreVis::VisualizationGroup::RAISIM_OBJECT_GROUP)
        .value("RAISIM_COLLISION_BODY_GROUP", raisim::OgreVis::VisualizationGroup::RAISIM_COLLISION_BODY_GROUP)
        .value("RAISIM_CONTACT_POINT_GROUP", raisim::OgreVis::VisualizationGroup::RAISIM_CONTACT_POINT_GROUP)
        .value("RAISIM_CONTACT_FORCE_GROUP", raisim::OgreVis::VisualizationGroup::RAISIM_CONTACT_FORCE_GROUP);

    ogre_vis.def(py::init([](raisim::World *world, uint32_t width=1280, uint32_t height=720,
        double fps=60, int anti_aliasing=2) {
                // get reference to the Ogre visualizer
                auto vis = raisim::OgreVis::get();

                // initialize (need to be called before initApp)
                vis->setWorld(world);
                vis->setWindowSize(width, height);
                vis->setImguiSetupCallback(imguiSetupCallback);
                vis->setImguiRenderCallback(imguiRenderCallBack);
                vis->setKeyboardCallback(raisimKeyboardCallback);
                vis->setSetUpCallback(setupCallback);
                vis->setAntiAliasing(anti_aliasing);

                // starts visualizer thread (this will call `setup()`)
                vis->initApp();

                // set desired FPS
                vis->setDesiredFPS(fps);

                // return the visualizer
                return vis;
            }),  R"mydelimiter(
            Instantiate the visualizer for the given world.

            Args:
                world (World): world instance.
                width (int): width of the window.
                height (int): height of the window.
                fps (double): the number of frames per second.
                anti_aliasing (int): anti aliasing.
            )mydelimiter",
            py::arg("world"), py::arg("width") = 1280, py::arg("height") = 720, py::arg("fps") = 60,
            py::arg("anti_aliasing") = 2)


        .def("get", &raisim::OgreVis::get, R"mydelimiter(
        Return a pointer to the singleton visualizer.

        Returns:
            OgreVis: reference to this class.
        )mydelimiter")


        .def("create_graphical_object", py::overload_cast<raisim::Sphere*, const std::string&,
            const std::string&>(&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add a sphere in the window.

            Args:
                sphere (Sphere): Raisim sphere instance.
                name (str): name of the sphere.
                material (str): material for visualization.
            )mydelimiter",
            py::arg("sphere"), py::arg("name"), py::arg("material"))


        .def("create_graphical_object", py::overload_cast<raisim::Ground *, double, const std::string&,
            const std::string&>(&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add a ground in the window.

            Args:
                ground (Ground): Raisim ground instance.
                dimension (double): the plane dimension.
                name (str): name of the ground.
                material (str): material for visualization.
            )mydelimiter",
            py::arg("ground"), py::arg("dimension"), py::arg("name"), py::arg("material"))


        .def("create_graphical_object", py::overload_cast<raisim::Box*, const std::string&,
            const std::string&>(&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add a box in the window.

            Args:
                box (Box): Raisim box instance.
                name (str): name of the box.
                material (str): material for visualization.
            )mydelimiter",
            py::arg("box"), py::arg("name"), py::arg("material"))


        .def("create_graphical_object", py::overload_cast<raisim::Cylinder*, const std::string&,
            const std::string&>(&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add a cylinder in the window.

            Args:
                cylinder (Cylinder): Raisim cylinder instance.
                name (str): name of the cylinder.
                material (str): material for visualization.
            )mydelimiter",
            py::arg("cylinder"), py::arg("name"), py::arg("material"))


        .def("create_graphical_object", py::overload_cast<raisim::Wire*, const std::string&,
            const std::string&>(&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add a wire in the window.

            Args:
                wire (Wire): Raisim wire instance.
                name (str): name of the wire.
                material (str): material for visualization.
            )mydelimiter",
            py::arg("wire"), py::arg("name"), py::arg("material") = "default")


        .def("create_graphical_object", py::overload_cast<raisim::Capsule*, const std::string&,
            const std::string&>(&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add a capsule in the window.

            Args:
                capsule (Capsule): Raisim capsule instance.
                name (str): name of the capsule.
                material (str): material for visualization.
            )mydelimiter",
            py::arg("capsule"), py::arg("name"), py::arg("material"))


        .def("create_graphical_object", py::overload_cast<raisim::ArticulatedSystem*, const std::string&>
            (&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add an articulated system in the window.

            Args:
                articulated_system (ArticulatedSystem): Raisim articulated system instance.
                name (str): name of the articulated system.
            )mydelimiter",
            py::arg("articulated_system"), py::arg("name"))


        .def("create_graphical_object", py::overload_cast<raisim::HeightMap*, const std::string&,
            const std::string&>(&raisim::OgreVis::createGraphicalObject), R"mydelimiter(
            Add a heightmap in the window.

            Args:
                heightmap (HeightMap): Raisim heightmap instance.
                name (str): name of the heightmap.
                material (str): material for visualization.
            )mydelimiter",
            py::arg("capsule"), py::arg("name"), py::arg("material") = "default")


        .def("sync", &raisim::OgreVis::sync, "Synchronize Raisim and Ogre.")


        .def("get_paused", &raisim::OgreVis::getPaused, R"mydelimiter(
        Return if the visualizer is paused or not.

        Returns:
            bool: True if the visualizer is paused.
        )mydelimiter")
        .def_property_readonly("paused", &raisim::OgreVis::getPaused, "Return if the visualizer is paused or not.")


        .def("remove", py::overload_cast<raisim::Object*>(&raisim::OgreVis::remove), R"mydelimiter(
        Remove an object from the visualizer.

        Args:
            obj (Object): Raisim object instance to be removed.
        )mydelimiter",
        py::arg("obj"))
        .def("remove", py::overload_cast<const std::string&>(&raisim::OgreVis::remove), R"mydelimiter(
        Remove an object from the visualizer.

        Args:
            name (str): name of the object to be removed.
        )mydelimiter",
        py::arg("name"))


        .def("get_selected", &raisim::OgreVis::getSelected, R"mydelimiter(
        Return the current selected item.

        Returns:
            Object: Raisim object instance.
            int: index.
        )mydelimiter")


        .def("get_selected_graphical_object", &raisim::OgreVis::getSelectedGraphicalObject, R"mydelimiter(
        Return the current selected graphical object item.

        Returns:
            GraphicObject: Raisim graphic object instance.
        )mydelimiter")


        .def("select", &raisim::OgreVis::select, R"mydelimiter(
        Select the given graphic object item.

        Args:
            obj (GraphicObject): Raisim graphic object instance.
            highlight (bool): if we should highlight the graphical object in the visualizer.
        )mydelimiter",
        py::arg("obj"), py::arg("highlight") = true)


        .def("deselect", &raisim::OgreVis::deselect, "Deselect the current selected object.")


        // TODO: wrap Ogre::SceneNode
//        .def("get_raisim_object", &raisim::OgreVis::getRaisimObject, "get Raisim object.")


        // TODO: wrap Ogre::SceneNode
//        .def("get_graphic_object", &raisim::OgreVis::getGraphicObject, "get the graphic object.")


        .def("is_recording", &raisim::OgreVis::isRecording, R"mydelimiter(
        Return if the visualizer is recording or not.

        Returns:
            bool: True if the visualizer is recording.
        )mydelimiter")


        .def("start_recording_video", &raisim::OgreVis::startRecordingVideo, R"mydelimiter(
        Initiate a video recording session.

        Returns:
            filename (str): filename for the recorded video.
        )mydelimiter",
        py::arg("filename"))


        .def("stop_recording_video_and_save", &raisim::OgreVis::stopRecordingVideoAndSave,
            "Stop the recording of the video and save it in the previous given filename.")


        .def("set_desired_fps", &raisim::OgreVis::setDesiredFPS, R"mydelimiter(
        Set the desired frame per second.

        Args:
            fps (double): frame per second.
        )mydelimiter",
        py::arg("fps"))


        .def("set_visibility_mask", &raisim::OgreVis::setVisibilityMask, R"mydelimiter(
        Set the visibility mask.

        Args:
            mask (unsigned long int): mask (it is a bitfield).
        )mydelimiter",
        py::arg("mask"))


        .def("get_visual_object_list", &raisim::OgreVis::getVisualObjectList, R"mydelimiter(
        Return the list of visual objects.

        Returns:
            dict[str:VisualObject]: dictionary mapping names to visual objects.
        )mydelimiter")


        .def("set_contact_visual_object_size", &raisim::OgreVis::setContactVisObjectSize, R"mydelimiter(
        Set the contact visual object size.

        Args:
            point_size (float): point size.
            force_arrow_length (float): force size corresponding to the maximum impulse.
        )mydelimiter",
        py::arg("point_size"), py::arg("force_arrow_length"))


        .def("get_real_time_factor_reference", &raisim::OgreVis::getRealTimeFactorReference, R"mydelimiter(
        Get the real time factor reference.

        Returns:
            float: real time factor reference.
        )mydelimiter")


        .def("set_remote_mode", &raisim::OgreVis::setRemoteMode, R"mydelimiter(
        Set the remote mode.

        Args:
            mode (bool): True if we are in a remote mode.
        )mydelimiter",
        py::arg("mode"))


        .def("remote_run", &raisim::OgreVis::remoteRun, "Run in remote mode.")


        .def("add_visual_object", [](raisim::OgreVis &self, const std::string &name, const std::string &mesh_name,
            const std::string &material, py::array_t<double> scale, bool cast_shadow = true,
            unsigned long int group = raisim::OgreVis::VisualizationGroup::RAISIM_OBJECT_GROUP |
            raisim::OgreVis::VisualizationGroup::RAISIM_COLLISION_BODY_GROUP) {
                // convert np.array to vec<3>
                raisim::Vec<3> scale_ = convert_np_to_vec<3>(scale);
                self.addVisualObject(name, mesh_name, material, scale_, cast_shadow, group);
            }, R"mydelimiter(
        Add a visual object.

        Args:
            name (str): name of the visual object.
            mesh_name (str): name of the material.
            material (str): material.
            scale (np.array[float[3]]): scale.
            cast_shadow (bool): if we should cast shadow or not.
            group (unsigned long int): group. You can select between {RAISIM_OBJECT_GROUP, RAISIM_COLLISION_BODY_GROUP,
                RAISIM_CONTACT_POINT_GROUP, RAISIM_CONTACT_FORCE_GROUP}, or any combination using bit operations.
        )mydelimiter",
        py::arg("name"), py::arg("mesh_name"), py::arg("material"), py::arg("scale"), py::arg("cast_shadow") = true,
        py::arg("group") = raisim::OgreVis::VisualizationGroup::RAISIM_OBJECT_GROUP |
            raisim::OgreVis::VisualizationGroup::RAISIM_COLLISION_BODY_GROUP)


        .def("clear_visual_object", &raisim::OgreVis::clearVisualObject, "Clear all the visual objects.")


        .def("build_height_map", &raisim::OgreVis::buildHeightMap, R"mydelimiter(
        Build the heigthmap.

        Args:
            name (str): the heightmap name.
            x_samples (int): the number of samples in x.
            x_size (float): the size in the x direction.
            x_center (float): the x center of the heightmap in the world.
            y_samples (int): the number of samples in y.
            y_size (float): the size in the y direction.
            y_center (float): the y center of the heightmap in the world.
            height (list[float]): list of desired heights.
        )mydelimiter",
        py::arg("name"), py::arg("x_samples"), py::arg("x_size"), py::arg("x_center"), py::arg("y_samples"),
        py::arg("y_size"), py::arg("y_center"), py::arg("height"))
    ;

//    py::class_<Visualizer, raisim::OgreVis> visualizer(visualizer_module, "Visualizer", "The visualizer class allows you to create a "
//        "window from which you can visualize the simulation.");
//    py::class_<Visualizer, raisim::OgreVis> visualizer(m, "Visualizer", "Visualizer");

//    visualizer.def("__init__", [](Visualizer &self, uint32_t width, uint32_t height, double fps=60) {
//                // initialize
//
//                // starts visualizer thread (this will call `setup()`)
//                self.initApp();
//
//                // set desired FPS
//                self.setDesiredFPS(fps)
//            }, "Instantiate the visualizer.")
//            .def("get", &Visualizer::get, "Return a pointer to the singleton visualizer.");
////            .def("create_graphical_object", [](Visualizer &self, const std::string &name, const std::string &meshname,
////                const std::string &material, const py::array_t<double> &scale, const py::array_t<double> &offset,
////                const py::array_t<double> &rotation_matrix, size_t local_idx, bool cast_shadow=true, bool selectable=true,
////                unsigned long int group = ));


}