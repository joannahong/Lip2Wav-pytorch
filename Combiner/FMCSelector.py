import FaceLandmarks.face_mesh_collections as fmc


FMCSelector = {
    "LipsEdge": fmc.FMesh_Lips_Edge,
    "LipsContent": fmc.FMesh_Lips_Content,
    "Lips": fmc.FMesh_Lips,

    "LeftEye": fmc.FMesh_Left_Eye,
    "LeftEyebrow": fmc.FMesh_Left_Eyebrow,
    "LeftIris": fmc.FMesh_Left_Iris,

    "RigthEye": fmc.FMesh_Right_Eye,
    "RightEyebrow": fmc.FMesh_Right_Eyebrow,
    "RightIris": fmc.FMesh_Right_Iris,

    "Eyes": fmc.FMesh_Eyes,
    "Eyebrows": fmc.FMesh_Eyebrows,
    "Irises": fmc.FMesh_Irises,

    "FaceEdge": fmc.FMesh_Face_Edge,
    "Forehead": fmc.FMesh_Forehead,
    "Contours": fmc.FMesh_Features_Contours,
    "Tesselation": fmc.FMesh_Tesselation,
    "WholeFace": fmc.FMesh_Whole_Face
}