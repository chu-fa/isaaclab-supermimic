"""Convert the pinned A3 URDF with Isaac Lab's importer in an existing runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.physics import PhysicsCfg
    from isaaclab.utils.version import standalone_importers_available

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    arguments = parser.parse_args()
    arguments.require_kit = not standalone_importers_available()
    arguments.physics = "isaacsim_physx" if arguments.require_kit else "newton_mjwarp"
    root = arguments.asset_root.resolve()
    urdf_path = root / "a3_description/urdf/model.urdf"
    profile_path = root / "a3_description/runtime_profile.json"
    manifest = json.loads((root / "official_asset_manifest.json").read_text())
    official_root = root / "official/robot_descriptions/models"
    for entry in manifest["files"]:
        if digest(official_root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Official source changed: {entry['path']}")
    if urdf_path.read_bytes() != (official_root / "a3_t2d5/urdf/model.urdf").read_bytes():
        raise ValueError("Conversion input must be the unchanged official URDF")
    if (root / "a3_usd_physx_v2").exists() or (root / "a3_usd_newton_v1").exists():
        raise FileExistsError("Select a fresh asset root; existing conversions are not overwritten")
    model = ElementTree.parse(urdf_path).getroot()
    profile = json.loads(profile_path.read_text())
    active_joints = {joint.get("name"): joint for joint in model.findall("joint") if joint.get("type") != "fixed"}
    if set(active_joints) != set(profile["joint_names"]):
        raise ValueError("Runtime profile and official URDF joint sets differ")
    for name, joint in active_joints.items():
        limits = joint.find("limit")
        motor = profile["joints"][name]
        if motor["effort_limit_sim"] > float(limits.get("effort")) or motor["velocity_limit_sim"] > float(limits.get("velocity")):
            raise ValueError(f"Runtime motor limits exceed official URDF: {name}")
    fixed_links = {
        joint.find("child").get("link"): joint.find("parent").get("link")
        for joint in model.findall("joint") if joint.get("type") == "fixed"
    }
    with launch_simulation(PhysicsCfg(), arguments):
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdGeom, UsdPhysics

        configuration = UrdfConverterCfg(
            asset_path=str(urdf_path), usd_dir=str(root / "a3_usd_physx_v2"),
            fix_base=False, merge_fixed_joints=True,
            collision_from_visuals=False, collision_type="Convex Decomposition",
            self_collision=True, robot_type="Humanoid", force_usd_conversion=True,
            ros_package_paths=[{"name": "robot_descriptions", "path": str(official_root.parent)}],
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
        converted_path = Path(UrdfConverter(configuration).usd_path)
        stage = Usd.Stage.Open(str(converted_path))
        stage.SetEditTarget(stage.GetRootLayer())
        root_prim = stage.GetDefaultPrim()
        variants = root_prim.GetVariantSet("Physics")
        available_variants = variants.GetVariantNames()
        if "physx" not in available_variants or "mujoco" not in available_variants:
            raise ValueError(f"Importer did not produce both physics variants: {available_variants}")
        variants.SetVariantSelection("physx")
        stage.GetRootLayer().Save()
        variants.SetVariantSelection("mujoco")
        # Newton uses convex hulls. PhysX keeps the importer's decomposition.
        # Uninstance before editing descendants so geometry is actually authored.
        for prim in list(stage.Traverse()):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        newton_path = root / "a3_usd_newton_v1/model.usda"
        newton_path.parent.mkdir()
        stage.Flatten().Export(str(newton_path))
        # The variant switch and hull edits above are intentionally unsaved.
        # The PhysX source on disk remains on its own variant and approximation.
        rigid_links = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        total_mass = sum(UsdPhysics.MassAPI(prim).GetMassAttr().Get() or 0.0 for prim in rigid_links)
        expected_mass = sum(float(link.find("inertial/mass").get("value")) for link in model.findall("link"))
        if len(rigid_links) != len(active_joints) + 1 or abs(total_mass - expected_mass) > 1e-4:
            raise AssertionError(f"Fixed-joint merge changed body count or mass: {len(rigid_links)}, {total_mass}, {expected_mass}")
        record = {
            "official_commit": manifest["commit"], "urdf_sha256": digest(urdf_path),
            "profile_sha256": digest(profile_path), "fixed_links_merged_by_importer": fixed_links,
            "active_joint_count": len(active_joints), "rigid_body_count": len(rigid_links),
            "urdf_mass_kg": expected_mass, "usd_mass_kg": total_mass,
            "newton_collision_approximation": "convexHull",
            "physx_collision_approximation": "convexDecomposition",
            "outputs_sha256": {
                str(path.relative_to(root)): digest(path)
                for directory in (root / "a3_usd_physx_v2", root / "a3_usd_newton_v1")
                for path in sorted(directory.rglob("*")) if path.is_file()
            },
        }
        (root / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
