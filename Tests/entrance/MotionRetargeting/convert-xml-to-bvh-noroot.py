from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.Loader.XMLCharacterLoader import XMLCharacterLoader
from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH


def main():
    scene = ODEScene()
    ch_fname = "../../CharacterData/StdHuman_New.xml"
    joint_fname = "../../CharacterData/JointControlParams_Kp500_Kd50.xml"
    loader = XMLCharacterLoader(scene, ch_fname, joint_fname)
    loader.load(True, False)
    scene.create_floor()

    # bvh_fname = ch_fname[:-3] + "bvh"
    bvh_fname = "test.bvh"
    writer = CharacterTOBVH(scene.character0)
    # writer.to_bvh(bvh_fname)
    writer.to_bvh_by_no_root(bvh_fname)


if __name__ == "__main__":
    main()