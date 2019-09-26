3D objects and Meshes
=====================

Creating worlds and environments that contains various 3d objects is important for many robotic tasks. However, it can
be quite hard to find these objects. In this ``README`` file, I will go over the basic tools you need, where you can
find 3D models for free, which ones you can redistribute without any legal problems, and which Python libraries you 
can use to get useful information from those meshes and how to import/export them in different format. 

Note that in this repository, you already have access to many free and open-source 3D models. You can check them in
the various subfolders in this ``meshes`` folder.


Licenses
~~~~~~~~

First, let's talk about the legal aspect.

In order to redistribute 3D models (if you are not the author), you have to make sure that these models are licensed
under:

- the Creative Commons (CC) License: https://creativecommons.org/licenses/
    - an example is the `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_ license
- the Free Art License: https://artlibre.org/licence/lal/en/

Other licenses allow you to use them for personal projects but do not allow you to redistribute them, unless you have
the specific authorization from the authors. This is the case for:

- the Personal Use License, as it can be found on websites such as Free3D: https://free3d.com/
- the General Model License, as it can be found on websites such as 3D Warehouse:
  https://legacy-3dwarehouse.sketchup.com/tos.html#license

Note that the Royalty Free License (which can be found here: https://free3d.com/royalty-free-license ) does not allow
you to redistribute downloaded models (see section 7.c). Note that purchased models (more than 0$) can not be 
redistributed as well. Anyway, always check the corresponding license. If a 3D model does not have a license,
you have to ask the original author(s) for permission.

**Note**: the license of each 3D mesh used in PRL is inside the corresponding folder.


Tools
~~~~~

Now, let's talk about tools that are useful to visualize, edit, manipulate, and convert 3D models.
I personally know two free and open-source tools that can be used on Linux and could interest users:

- `Meshlab <http://www.meshlab.net/>`_: "an open source system for processing and editing 3D triangular meshes. It
  provides a set of tools for editing, cleaning, healing, inspecting, rendering, texturing and converting meshes. It
  offers features for processing raw data produced by 3D digitization tools/devices and for preparing models for 3D
  printing".
- `Blender <https://www.blender.org/>`_: "a free and open source 3D creation suite. It supports the entirety of the
  3D pipeline—modeling, rigging, animation, simulation, rendering, compositing and motion tracking, even video editing
  and game creation".

There is a certain learning curve to learn how to use these tools (especially Blender) but once you master them they 
can become very handy and powerful tools.

Note that you can also use Blender in Python (>=3.3, not 2.* since Blender 2.5).

The following formats are supported (and thus can be converted between them):

- Meshlab: 3ds, ply, stl, obj (with corresponding mtl), off, wrl, dxf, dae, ctm, xyz, gts, json, m, u3d, idtf, x3m
- Blender: 3ds, fbx, bvh, ply, obj, x3d/wrl, stl, svg

If you need to split or export specific parts of a 3D model, use Blender. In Blender, you can select an object with a
right-click, multiple objects with 'shift + right click', all of them with 'a'. You can delete the selected objects
with 'x'. You can export the selected objects by going to 'File > Export' and then select 'Selection Only' on the
left panel (you should also set 'Path Mode' to 'Copy' to export the textures). If you need to transform the object
(by translating, rotating, or rescaling it), you can bring the transform panel by selecting the object and typing 'n'.
If you need to rescale the entire scene, you can select all the objects by typing 'a', then 's' (for scale), and
directly write the scaling factor, "0.01" for instance. To rotate, type 'r' and to translate type 'g'.
To move the camera, you can use the scroll wheel to zoom, and if you press on it, you can rotate it. If you need to
recenter the object, press 'ctrl + shift + alt + c' and click on 'origin to geometry'. You can press 'ctrl + z' to
undo something. You can also separate objects if necessary, please check the following tutorial on Youtube:
https://www.youtube.com/watch?v=U3J-oYFdyqQ


Websites
~~~~~~~~

Here are few websites that provide 3D free models.

- Under a CC or Free Art License
    - Sweet Home 3D: http://www.sweethome3d.com/freeModels.jsp
    - Blend swap: https://www.blendswap.com/
    - Sketchfab (check the license of the model to be sure): https://sketchfab.com/3d-models?features=downloadable
    - Gazebo models: https://bitbucket.org/osrf/gazebo_models/src/default/  and
      http://data.nvision2.eecs.yorku.ca/3DGEMS/
    - Bullet models: https://github.com/bulletphysics/bullet3/tree/master/data

- Under a Personal Use or Royalty Free License
    - Free3D: https://free3d.com/3d-models/
    - Sketchfab: https://sketchfab.com/features/download
    - CGTrader: https://www.cgtrader.com/free-3d-models
    - TurboSquid: https://www.turbosquid.com/Search/3D-Models/free
    - GrabCAD: https://grabcad.com/library
    - 3D warehouse: https://3dwarehouse.sketchup.com/?hl=en


Use 3D models with a Simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Bullet: it only accepts to load ``OBJ`` files. You can also load collada (``DAE``) and ``STL`` files through a URDF/SDF.
- MuJoCo: it only accepts to load ``STL`` files. Note that ``STL`` does not contain any texture information (compared 
  to others like ``OBJ`` and ``DAE``).


Create a URDF/SDF file
~~~~~~~~~~~~~~~~~~~~~~

Loaded models in the simulator might not have the correct collision shape, mass, inertia, or other physical properties.
You can set them in a `URDF (Unified Robot Description Format) <http://wiki.ros.org/urdf>`_, `SDF <http://sdformat.org/>`_, 
or `XML MuJoCo` files.


- Tutorials on URDF:
    - `URDF Tutorials <http://wiki.ros.org/urdf/Tutorials>`_
    - `Create a URDF for an Industrial Robot <http://wiki.ros.org/Industrial/Tutorials/Create%20a%20URDF%20for%20an%20Industrial%20Robot>`_
- Tutorials on SDF:
    - `Make a model (from Gazebo) <http://gazebosim.org/tutorials?tut=build_model>`_
    - `SDF Tutorials <http://sdformat.org/tutorials?cat=get_started&>`_

In order to set correct inertia values (if you don't have access to CAD models), please have a look at the following
tutorials:

- `Adding Physical and Collision Properties to a URDF Model <http://wiki.ros.org/urdf/Tutorials/Adding%20Physical%20and%20Collision%20Properties%20to%20a%20URDF%20Model>`_
- `Inertial parameters of triangle meshes <http://gazebosim.org/tutorials?tut=inertia&cat=build_robot>`_

Based on the 2 above tutorials, you will understand how to compute inertia matrices based on the mass and the volume
of the mesh (if uniform density) computed using MeshLab, for instance.


Python libraries
~~~~~~~~~~~~~~~~

In Python (2.* or 3.*), you can:

- get useful information from a mesh (such as the center of mass, volume, moment of inertia, and others), using the `trimesh <https://github.com/mikedh/trimesh>`_ library.
- load and export in different formats, using the `pyassimp <https://github.com/assimp/assimp/blob/master/port/PyAssimp/README.md>`_ library (which is a Python wrapper around the `assimp <https://github.com/assimp/assimp>`_ library).
- In Python 3, you can use the blender python library: `bpy <https://docs.blender.org/api/current/index.html>`_.


That's all folks!