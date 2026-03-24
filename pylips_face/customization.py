from pylips.speech import RobotFace

face = RobotFace()

face.set_appearance({
    "brow_width": 90,     # default ≈ 240 → smaller
    "mouth_width": 250     # default ≈ 520 → smaller
})
