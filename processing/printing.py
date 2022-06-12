
def printing_results(frame_number, position, frame_offset, frame_duration, velocity, x, z):
    print("______________________")
    print("Frame: ", frame_number)
    print("Position in space", position)
    print("Velocity calculate for last ", (frame_offset * frame_duration), " s")
    print("Velocity Vector: ", velocity)
    print("Landing coordinates x:", x, "z", z)
    print("______________________")