import mujoco
import mujoco.viewer

#in the terminal run: 'onshape-to-robot config.json to download the assests to set the path
xml_path = r"onshape-stuff\scene.xml"
 
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
 
mujoco.viewer.launch(model, data)