# To run the code `H3DLoad

import H3DInterface as h3d


# print(dir(h3d))
# ['AutoUpdate', 'Console', 'Field', 'H3DConsole', 'INITIALIZE_ONLY', 'INPUT_ONLY', 'INPUT_OUTPUT', 'LogLevel',
# 'MFBOOL', 'MFBool', 'MFCOLOR', 'MFCOLORRGBA', 'MFColor', 'MFColorRGBA', 'MFDOUBLE', 'MFDouble', 'MFFLOAT',
# 'MFFloat', 'MFINT32', 'MFInt32', 'MFMATRIX3D', 'MFMATRIX3F', 'MFMATRIX4D', 'MFMATRIX4F', 'MFMatrix3d',
# 'MFMatrix3f', 'MFMatrix4d', 'MFMatrix4f', 'MFNODE', 'MFNode', 'MFQUATERNION', 'MFQuaternion', 'MFROTATION',
# 'MFRotation', 'MFSTRING', 'MFString', 'MFTIME', 'MFTime', 'MFVEC2D', 'MFVEC2F', 'MFVEC3D', 'MFVEC3F', 'MFVEC4D',
# 'MFVEC4F', 'MFVec2d', 'MFVec2f', 'MFVec3d', 'MFVec3f', 'MFVec4d', 'MFVec4f', 'MField', 'MFieldBack', 'MFieldClear',
# 'MFieldEmpty', 'MFieldErase', 'MFieldFront', 'MFieldPopBack', 'MFieldPushBack', 'MFieldSize', 'Matrix3d',
# 'Matrix3f', 'Matrix4d', 'Matrix4f', 'Node', 'OUTPUT_ONLY', 'PeriodicUpdate', 'Quaternion', 'RGB', 'RGBA',
# 'Rotation', 'SFBOOL', 'SFBool', 'SFCOLOR', 'SFCOLORRGBA', 'SFColor', 'SFColorRGBA', 'SFDOUBLE', 'SFDouble',
# 'SFFLOAT', 'SFFloat', 'SFINT32', 'SFInt32', 'SFMATRIX3D', 'SFMATRIX3F', 'SFMATRIX4D', 'SFMATRIX4F', 'SFMatrix3d',
# 'SFMatrix3f', 'SFMatrix4d', 'SFMatrix4f', 'SFNODE', 'SFNode', 'SFQUATERNION', 'SFQuaternion', 'SFROTATION',
# 'SFRotation', 'SFSTRING', 'SFString', 'SFStringGetValidValues', 'SFStringIsValidValue', 'SFTIME', 'SFTime',
# 'SFVEC2D', 'SFVEC2F', 'SFVEC3D', 'SFVEC3F', 'SFVEC4D', 'SFVEC4F', 'SFVec2d', 'SFVec2f', 'SFVec3d', 'SFVec3f',
# 'SFVec4d', 'SFVec4f', 'SField', 'TypedField', 'UNKNOWN_X3D_TYPE', 'Vec2d', 'Vec2f', 'Vec3d', 'Vec3f', 'Vec4d',
# 'Vec4f', '_ConsoleStderr', '_ConsoleStdout', '__builtins__', '__doc__', '__name__', '__package__',
# 'addProgramSetting', 'addURNResolveRule', 'auto_update_classes', 'createField', 'createNode',
# 'createVRMLFromString', 'createVRMLFromURL', 'createVRMLNodeFromString', 'createVRMLNodeFromURL',
# 'createX3DFromString', 'createX3DFromURL', 'createX3DNodeFromString', 'createX3DNodeFromURL', 'eventSink',
# 'exportGeometryAsSTL', 'fieldGetAccessType', 'fieldGetFullName', 'fieldGetName', 'fieldGetOwner',
# 'fieldGetRoutesIn', 'fieldGetRoutesOut', 'fieldGetTypeName', 'fieldGetValue', 'fieldGetValueAsString',
# 'fieldHasRouteFrom', 'fieldIsAccessCheckOn', 'fieldIsUpToDate', 'fieldReplaceRoute', 'fieldReplaceRouteNoEvent',
# 'fieldRoute', 'fieldRouteNoEvent', 'fieldRoutesTo', 'fieldSetAccessCheck', 'fieldSetAccessType', 'fieldSetName',
# 'fieldSetOwner', 'fieldSetValue', 'fieldSetValueFromString', 'fieldTouch', 'fieldUnroute', 'fieldUnrouteAll',
# 'fieldUpToDate', 'findNodes', 'getActiveBackground', 'getActiveBindableNode', 'getActiveDeviceInfo', 'getActiveFog',
# 'getActiveGlobalSettings', 'getActiveNavigationInfo', 'getActiveStereoInfo', 'getActiveViewpoint', 'getCPtr',
# 'getCurrentScenes', 'getHapticsDevice', 'getNamedNode', 'getNrHapticsDevices', 'log', 'mfield_types',
# 'periodic_update_classes', 'resolveURLAsFile', 'resolveURLAsFolder', 'sfield_types', 'sys', 't',
# 'takeScreenshot', 'throwQuitAPIException', 'time', 'typed_field_classes', 'writeNodeAsX3D']


num_devices = h3d.getNrHapticsDevices()
print("Number of devices: {}".format(num_devices))

device = None
info = h3d.getActiveDeviceInfo()
if info:
    device = info.device.getValue()[0]

if device is not None:
    print(device.trackerPosition)
    print(device.trackerOrientation)
    print(device.mainButton)
    print(device.secondaryButton)
